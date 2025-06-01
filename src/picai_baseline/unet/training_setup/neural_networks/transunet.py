#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm


class ConvBlock(nn.Module):
    """
    Double 3D convolution block with batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels, spatial_dims=3, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = Convolution(
            dimensions=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            act=Act.RELU,
            norm=Norm.BATCH
        )
        self.conv2 = Convolution(
            dimensions=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            act=Act.RELU,
            norm=Norm.BATCH
        )
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block to capture global contextual information
    Memory-efficient implementation with patch-based processing
    """
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=2048, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        # Reduce number of heads for smaller VRAM usage
        actual_heads = min(heads, max(1, dim // 32))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=actual_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm2 = nn.LayerNorm(dim)
        
        # Max sequence length for memory efficiency
        self.max_seq_len = 512 # 4096  # Can be adjusted based on VRAM availability

    def forward(self, x):
        # Store original shape
        batch, channels, depth, height, width = x.shape
        
        # Process in patches if sequence is too long
        if depth * height * width > self.max_seq_len:
            # Use smaller chunks with center-focused attention
            center_d, center_h, center_w = depth // 2, height // 2, width // 2
            chunk_d, chunk_h, chunk_w = min(depth, 8), min(height, 64), min(width, 64)
            
            # Extract center region
            d_start = max(0, center_d - chunk_d // 2)
            h_start = max(0, center_h - chunk_h // 2)
            w_start = max(0, center_w - chunk_w // 2)
            
            d_end = min(depth, d_start + chunk_d)
            h_end = min(height, h_start + chunk_h)
            w_end = min(width, w_start + chunk_w)
            
            # Create center chunk
            center_chunk = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
            
            # Process center chunk with transformer
            chunk_flat = center_chunk.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
            chunk_norm = self.norm1(chunk_flat)
            chunk_transformed = self.transformer(chunk_norm)
            chunk_transformed = self.norm2(chunk_transformed)
            
            # Reshape back
            chunk_reshaped = chunk_transformed.reshape(
                batch, d_end-d_start, h_end-h_start, w_end-w_start, channels
            ).permute(0, 4, 1, 2, 3)
            
            # Copy transformed chunk back to original tensor
            result = x.clone()
            result[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += chunk_reshaped
            
            return result
        else:
            # Reshape for transformer: [batch, d*h*w, channels]
            x_flat = x.permute(0, 2, 3, 4, 1).reshape(batch, depth*height*width, channels)
            
            # Apply layer normalization and transformer
            x_norm = self.norm1(x_flat)
            x_transformed = self.transformer(x_norm)
            x_transformed = self.norm2(x_transformed)
            
            # Reshape back to original shape
            x_reshaped = x_transformed.reshape(batch, depth, height, width, channels).permute(0, 4, 1, 2, 3)
            
            # Residual connection
            return x + x_reshaped


class DownBlock(nn.Module):
    """
    Downsampling block with max pooling and convolution
    """
    def __init__(self, in_channels, out_channels, spatial_dims=3, dropout_p=0.0):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if spatial_dims == 3 else nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, spatial_dims, dropout_p)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block with transposed convolution and concatenation
    """
    def __init__(self, in_channels, out_channels, spatial_dims=3, dropout_p=0.0):
        super(UpBlock, self).__init__()
        if spatial_dims == 3:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, spatial_dims, dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle potential size mismatch for concatenation
        if x.shape != skip.shape:
            # Calculate padding based on dimensionality
            if x.dim() == 5:  # 3D (B, C, D, H, W)
                diffD = skip.size()[2] - x.size()[2]
                diffH = skip.size()[3] - x.size()[3]
                diffW = skip.size()[4] - x.size()[4]
                
                # Apply padding if necessary
                x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                            diffH // 2, diffH - diffH // 2,
                            diffD // 2, diffD - diffD // 2])
            else:  # 4D (B, C, H, W)
                diffH = skip.size()[2] - x.size()[2]
                diffW = skip.size()[3] - x.size()[3]
                
                # Apply padding if necessary
                x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                            diffH // 2, diffH - diffH // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


class UNetTransformer(nn.Module):
    """
    3D UNet with Transformer blocks for prostate cancer detection in MRI
    """
    def __init__(
        self, 
        spatial_dims=3, 
        in_channels=3, 
        out_channels=2, 
        channels=[16, 32], # , 64], #, 128], 
        strides=None, 
        dropout_p=0.2
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions (2 for 2D, 3 for 3D)
            in_channels: number of input channels
            out_channels: number of output channels
            channels: list of feature dimensions for each level
            strides: list of convolutional strides (not used in this implementation but kept for compatibility)
            dropout_p: dropout probability
        """
        super(UNetTransformer, self).__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = channels
        self.dropout_p = dropout_p
        
        # Initial convolution block
        self.encoder1 = ConvBlock(in_channels, channels[0], spatial_dims, dropout_p)
        
        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        for i in range(len(channels)-1):
            self.down_blocks.append(DownBlock(channels[i], channels[i+1], spatial_dims, dropout_p))
        
        # Transformer block at bottleneck
        self.transformer = TransformerBlock(dim=channels[-1], dropout=dropout_p)
        
        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels)-1, 0, -1):
            self.up_blocks.append(UpBlock(channels[i], channels[i-1], spatial_dims, dropout_p))
        
        # Final convolution
        if spatial_dims == 3:
            self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        skip_connections = []
        
        # Initial block
        x = self.encoder1(x)
        skip_connections.append(x)
        
        # Down blocks
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
        
        # Remove the last skip connection (bottleneck)
        skip_connections = skip_connections[:-1]
        
        # Apply transformer at bottleneck
        try:
            x = self.transformer(x)
        except RuntimeError as e:
            # If transformer fails (e.g., out of memory), skip it with a warning
            print(f"Warning: Transformer block failed with error: {e}")
            print("Skipping transformer and continuing with the decoder")
        
        # Decoder with skip connections (in reverse order)
        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

    def get_detection_map(self, x):
        """
        Generate detection map with sigmoid activation
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_case_level_prediction(self, detection_map):
        """
        Generate case-level prediction by taking maximum value
        from detection map (as suggested in PI-CAI evaluation)
        """
        return torch.max(detection_map)
