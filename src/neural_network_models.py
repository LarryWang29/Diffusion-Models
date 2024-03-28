"""!@file src/neural_network_models.py
@brief Implementation of the neural network models used in DDPM, 
Cold Diffusion models.

@details This file contains the implementation of the neural network models
and blocks used in both DDPM and Cold Diffusion models.

@author Larry Wang
@Date 22/03/2024
"""
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
        """
        !@brief Initialise the CNN block used in DDPM model.

        @details This function initialises the CNN block used in the DDPM model.
        The CNN block makes up the neural network model that predicts the "error term"
        of the image. It consists of a convolutional layer, a layer normalisation layer,
        and an activation function.

        @param in_channels The number of input channels to the CNN block.
        @param out_channels The number of output channels from the CNN block.
        @param expected_shape The expected shape of the input tensor.
        @param act The activation function used in the CNN block. The default is the GELU
        activation function.
        @param kernel_size The kernel size of the convolutional layer. The default is 7.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)),
            act()
        )

    def forward(self, x):
        """
        !@brief Forward pass of the CNN block.

        @param x The input tensor to the CNN block.

        @return The output tensor from the CNN block.
        """
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
        """
        !@brief Initialise the CNN model used in DDPM model.

        @details This function initialises the CNN model used in the DDPM model.
        The CNN model is used to predict the "error term" of the image. The architecture
        of the CNN model consists of a series of CNN blocks, each consisting of a convolutional
        layer, a layer normalisation layer, and an activation function. The CNN model also
        includes a time embedding layer that provides information about the time along the
        diffusion process.

        @param in_channels The number of input channels to the CNN model.
        @param expected_shape The expected shape of the input tensor.
        @param n_hidden The number of hidden channels in the hidden CNN blocks.
        @param kernel_size The kernel size of the convolutional layer. The default is 7.
        @param last_kernel_size The kernel size of the final convolutional layer. The default is 3.
        @param time_embeddings The dimension of time embeddings used in the time embedding layer.
        @param act The activation function used in the CNN model. The default is the GELU activation function.

        @return None
        """
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
        """
        !@brief Encode the time information using a variation of sinusoidal embeddings
        from the starter notebook

        @details This function encodes the time information using sinusoidal embeddings.
        The sinusoidal embeddings are used to provide information about the time along
        the diffusion process.

        @param t The time tensor to encode.
        
        @return The encoded time tensor.
        """
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
        """
        !@brief Initialise the ConvNextBlock used in the UNet model.

        @details This function initialises the ConvNextBlock used in the UNet model.
        The ConvNextBlock consists of a series of convolutional layers, layer normalisation
        layers, and activation functions. IT also includes a shallow MLP that ensures the 
        precomputed time embeddings are in the same dimensions as the input tensor.

        @param in_channels The number of input channels to the ConvNextBlock.
        @param out_channels The number of output channels from the ConvNextBlock.
        @param input_shape The expected shape of the input tensor.
        @param time_emb_dim The dimension of the inputted precomputed time embedding.
        """
        super().__init__()

        # A shallow MLP to ensure the precomputed time embeddings are in the same
        # dimensions as the input tensor
        self.shallow_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, in_channels)
        )

        # Depthwise separable convolution across channels
        self.ds_conv = nn.Conv2d(in_channels, in_channels, kernel_size=7,
                                 padding=3, groups=in_channels)

        # Layer normalisation
        self.ln = nn.LayerNorm((in_channels, *input_shape))

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size=3,
                               padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3,
                               padding=1)
        
        # Residual convolution
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        !@brief Forward pass of the ConvNextBlock.

        @details This function performs the forward pass of the ConvNextBlock.
        
        @param x The input tensor to the ConvNextBlock.
        @param t The precomputed time embedding tensor, to be used in the ConvNextBlock.
        """
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
        """
        !@brief Initialise the UNet model used in the Cold Diffusion model.
        
        @details This function initialises the UNet model used in the Cold Diffusion model.
        The UNet model is used to model the diffusion process of an image. The architecture
        of the UNet model consists of a series of ConvNextBlocks, each consisting of a series
        of convolutional layers, layer normalisation layers, and activation functions. The time
        embedding used are the same as the ones used in the starter DDPM model. The UNet model
        also includes a series of transpose convolutional layers to upsample the image.

        @param in_channels The number of input channels to the UNet model.
        @param out_channels The number of output channels from the UNet model.
        @param hidden_dims The number of hidden channels in the hidden ConvNextBlocks.
        @param time_embeddings The dimension of time embeddings.
        @param act The activation function used in the UNet model. The default is the GELU activation 
        function.

        @return None
        """
        super().__init__()
        # Downward path
        self.conv_block1 = ConvNextBlock(in_channels, hidden_dims[0],
                                         (28, 28), time_emb_dim=32)
        self.maxpool_block1 = self.maxpool_block(2, 2)
        self.conv_block2 = ConvNextBlock(hidden_dims[0], hidden_dims[1],
                                         (14, 14), time_emb_dim=32)
        self.maxpool_block2 = self.maxpool_block(2, 2)
        self.conv_block3 = ConvNextBlock(hidden_dims[1], hidden_dims[2],
                                         (7, 7), time_emb_dim=32)
        self.maxpool_block3 = self.maxpool_block(2, 2)

        # Middle block
        self.middle_block = ConvNextBlock(hidden_dims[2], hidden_dims[2]*2,
                                          (3, 3), time_emb_dim=32)
        
        # Upward path
        self.transpose_block1 = self.transpose_block(hidden_dims[2]*2, hidden_dims[2], 3, 2)

        self.upconv_block1 = ConvNextBlock(hidden_dims[2]*2, hidden_dims[2],
                                           (7, 7), time_emb_dim=32)
        self.transpose_block2 = self.transpose_block(hidden_dims[1]*2, hidden_dims[1], 2, 2)
        self.upconv_block2 = ConvNextBlock(hidden_dims[1]*2, hidden_dims[1],
                                           (14, 14), time_emb_dim=32)
        self.transpose_block3 = self.transpose_block(hidden_dims[1], hidden_dims[0], 2, 2)
        self.upconv_block3 = ConvNextBlock(hidden_dims[1], hidden_dims[0],
                                           (28, 28), time_emb_dim=32)

        # Final block
        self.final = self.final_block(hidden_dims[0], out_channels, 1, 1)

        # Time embedding block, same as in the starter DDPM model
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
        """
        !@brief Encode the time information using a variation of sinusoidal embeddings
        from the starter notebook

        @details This function encodes the time information using sinusoidal embeddings.
        The sinusoidal embeddings are used to provide information about the time along
        the diffusion process.

        @param t The time tensor to encode.
        
        @return The encoded time tensor.
        """
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)

    def maxpool_block(self, kernel_size, stride):
        """
        !@ brief Maxpooling block used in the UNet model. Used to downsample the image
        by a factor of 2 and increase the number of channels by a factor of 2.

        @param kernel_size The kernel size of the maxpooling layer.
        @param stride The stride of the maxpooling layer.

        @return The maxpooled output.
        """
        return nn.MaxPool2d(kernel_size, stride)

    def transpose_block(self, in_channels, out_channels, kernel_size, stride):
        """
        !@brief Transpose convolutional block used in the UNet model. Used to upsample the image
        by a factor of 2 and decrease the number of channels by a factor of 2.

        @param in_channels The number of input channels to the transpose convolutional layer.
        @param out_channels The number of output channels from the transpose convolutional layer.

        @return The upsampled output.
        """
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                  stride)

    def final_block(self, in_channels, out_channels, kernel_size, stride):
        """
        !@brief Final convolutional block used in the UNet model. Used to output the final
        image.

        @param in_channels The number of input channels to the final convolutional layer.
        @param out_channels The number of output channels from the final convolutional layer.
        @param kernel_size The kernel size of the final convolutional layer.
        @param stride The stride of the final convolutional layer.
        
        @return The final output.
        """
        final = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        return final

    def forward(self, x, t):
        """
        !@brief Forward pass of the UNet model.

        @details This function performs the forward pass of the UNet model.

        @param x The input tensor to the UNet model.
        @param t The time tensor to feed into the UNet model.

        @return The output tensor from the UNet model.
        """
        t = self.time_encoding(t)

        # Downward path
        conv1 = self.conv_block1(x, t)
        maxpool1 = self.maxpool_block1(conv1)
        conv2 = self.conv_block2(maxpool1, t)
        maxpool2 = self.maxpool_block2(conv2)
        conv3 = self.conv_block3(maxpool2, t)
        maxpool3 = self.maxpool_block3(conv3)

        # Middle block
        middle = self.middle_block(maxpool3, t)

        # Upward path
        transpose1 = self.transpose_block1(middle)
        upconv1 = self.upconv_block1(torch.cat([transpose1, conv3], dim=1), t)
        transpose2 = self.transpose_block2(upconv1)
        upconv2 = self.upconv_block2(torch.cat([transpose2, conv2], dim=1), t)
        transpose3 = self.transpose_block3(upconv2)
        upconv3 = self.upconv_block3(torch.cat([transpose3, conv1], dim=1), t)

        final = self.final(upconv3)

        return final
