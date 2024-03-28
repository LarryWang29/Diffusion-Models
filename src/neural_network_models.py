import torch
import torch.nn as nn
import numpy as np


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)),
            act()
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, 128), act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed


class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 input_shape, time_emb_dim=32):
        super().__init__()

        self.shallow_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, in_channels)
        )

        self.ds_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7,
                                 padding=3, groups=in_channels)

        self.ln = nn.LayerNorm((in_channels, *input_shape))
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3,
                               padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3,
                               padding=1)
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        depth_conv = self.ds_conv(x)

        time_embedding = self.shallow_mlp(t)[:, :, None, None]
        depth_conv = depth_conv + time_embedding

        ln = self.ln(depth_conv)
        conv1 = self.conv1(ln)
        act = self.act(conv1)
        conv2 = self.conv2(act)

        res_conv = self.res_conv(x)
        return conv2 + res_conv


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        hidden_dims=[32, 64, 128],
        time_embeddings=32,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv_block1 = ConvNextBlock(in_channels, hidden_dims[0],
                                         (28, 28), time_emb_dim=32)
        self.maxpool_block1 = self.maxpool_block(2, 2)
        self.conv_block2 = ConvNextBlock(hidden_dims[0], hidden_dims[1],
                                         (14, 14), time_emb_dim=32)
        self.maxpool_block2 = self.maxpool_block(2, 2)
        self.conv_block3 = ConvNextBlock(hidden_dims[1], hidden_dims[2],
                                         (7, 7), time_emb_dim=32)
        self.maxpool_block3 = self.maxpool_block(2, 2)

        self.middle_block = ConvNextBlock(hidden_dims[2], hidden_dims[2]*2,
                                          (3, 3), time_emb_dim=32)
        self.transpose_block1 = self.transpose_block(hidden_dims[2]*2, hidden_dims[2], 3, 2)

        self.upconv_block1 = ConvNextBlock(hidden_dims[2]*2, hidden_dims[2],
                                           (7, 7), time_emb_dim=32)
        self.transpose_block2 = self.transpose_block(hidden_dims[1]*2, hidden_dims[1], 2, 2)
        self.upconv_block2 = ConvNextBlock(hidden_dims[1]*2, hidden_dims[1],
                                           (14, 14), time_emb_dim=32)
        self.transpose_block3 = self.transpose_block(hidden_dims[1], hidden_dims[0], 2, 2)
        self.upconv_block3 = ConvNextBlock(hidden_dims[1], hidden_dims[0],
                                           (28, 28), time_emb_dim=32)

        self.final = self.final_block(hidden_dims[0], out_channels, 1, 1)

        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, hidden_dims[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)

    def maxpool_block(self, kernel_size, stride):
        return nn.MaxPool2d(kernel_size, stride)

    def transpose_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                  stride)

    def final_block(self, in_channels, out_channels, kernel_size, stride):
        final = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        return final

    def forward(self, x, t):
        t = self.time_encoding(t)

        conv1 = self.conv_block1(x, t)
        maxpool1 = self.maxpool_block1(conv1)
        conv2 = self.conv_block2(maxpool1, t)
        maxpool2 = self.maxpool_block2(conv2)
        conv3 = self.conv_block3(maxpool2, t)
        maxpool3 = self.maxpool_block3(conv3)

        middle = self.middle_block(maxpool3, t)

        transpose1 = self.transpose_block1(middle)
        upconv1 = self.upconv_block1(torch.cat([transpose1, conv3], dim=1), t)
        transpose2 = self.transpose_block2(upconv1)
        upconv2 = self.upconv_block2(torch.cat([transpose2, conv2], dim=1), t)
        transpose3 = self.transpose_block3(upconv2)
        upconv3 = self.upconv_block3(torch.cat([transpose3, conv1], dim=1), t)

        final = self.final(upconv3)

        return final
