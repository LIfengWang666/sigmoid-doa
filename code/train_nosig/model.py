import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import DropPath

## Convolution module prepended to ViTs
class Conv_Layers(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, (1, 3), stride=1, padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels * 4, in_channels, (1, 3), stride=1, padding=(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return out

        
## Split the input into patches, convert each patch to an embedding vector, and add positional encoding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size_x: int = 1, patch_size_y: int = 6, emb_size: int = 64, img_size: int = 120, num_cls_tokens: int = 6):
        self.patch_size_x = patch_size_x
        self.patch_size_y = patch_size_y
        self.num_cls_tokens = num_cls_tokens  # Number of CLS tokens to include
        self.positions_num = (img_size // patch_size_y) + num_cls_tokens  # Adjust positional encoding size based on the number of CLS tokens
        
        super().__init__()
        
        # Define convolutional layers and positional encoding
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_x, patch_size_y), stride=(patch_size_x, patch_size_y)),
            nn.BatchNorm2d(emb_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )       

        # Add num_cls_tokens CLS tokens
        self.cls_token_a = nn.Parameter(torch.randn(1, num_cls_tokens//2, emb_size)) 
        self.cls_token_b = nn.Parameter(torch.randn(1, num_cls_tokens//2, emb_size)) 
        
        # Positional encoding, considering CLS token positions
        self.positions = nn.Parameter(torch.randn(self.positions_num, emb_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)  # Convert the original image into patch embeddings
        
        cls_tokens_a = repeat(self.cls_token_a, '() n e -> b n e', b=b)  # Generate CLS tokens and extend their dimensions to match batch size
        cls_tokens_b = repeat(self.cls_token_b, '() n e -> b n e', b=b)  # Generate CLS tokens and extend their dimensions to match batch size
        x = torch.cat([cls_tokens_a, x], dim=1)  # Concatenate CLS tokens and patches along the 2nd dimension (patches)
        x = torch.cat([x, cls_tokens_b], dim=1)  # Concatenate CLS tokens after patches along the 2nd dimension
        
        x += self.positions  # Add positional encoding
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 64, num_heads: int = 8):
        super().__init__()
        self.emb_size = emb_size  # Save embedding size
        self.num_heads = num_heads  # Save the number of heads for multi-head attention
        self.qkv = nn.Linear(emb_size, emb_size * 3)  # Define a linear layer to map input to QKV
        self.projection = nn.Linear(emb_size, emb_size)  # Define a linear layer to map multi-head attention output to the original dimension
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        ## Compute QKV
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h = self.num_heads, qkv = 3)  # Compute QKV and rearrange dimensions
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # Extract Q, K, and V

        ## Compute attention scores
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # Define multiplication method and compute dot product between Q and K
        scaling = self.emb_size ** (1/2)  # Compute scaling factor, typically the square root of embedding dimension
        att = F.softmax(energy, dim=-1) / scaling  # Apply softmax to attention scores and normalize

        ## Compute attention output
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  # Apply attention weights to values
        out = rearrange(out, "b h n d -> b n (h d)")  # Concatenate multi-head outputs
        out = self.projection(out)  # Map multi-head attention output back to the original dimension
        return out
    
## Implements a residual connection
class ResidualAdd(nn.Module):
    ## fn is a sub-network module or a sequence of operations, essentially any callable
    ## Save the passed fn as a class attribute for use in the forward method
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    ## Define the forward propagation method for the residual block
    def forward(self, x, **kwargs):
        res = x  # Save input as residual
        x = self.fn(x, **kwargs)  # Call fn and compute output
        x += res  # Add residual to output
        return x

## A feed-forward neural network block used in the Transformer encoder
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4):  # expansion: expansion factor for feed-forward network
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),  # Linear layer to map input to a higher dimension
            nn.GELU(),  # Activation function, GELU instead of ReLU
            nn.Linear(expansion * emb_size, emb_size),  # Linear layer to map back to the original dimension
        )

## A fundamental component of the Transformer encoder, stacked to form the encoder
## Includes a multi-head attention module and a feed-forward module
class TransformerEncoderBlock(nn.Sequential):
    ## forward_expansion: expansion factor for the feed-forward module, indicating how much larger the intermediate layer is compared to the input layer
    def __init__(self,
                 emb_size: int = 64,
                 forward_expansion: int = 4,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),  # Layer normalization
                MultiHeadAttention(emb_size, **kwargs),  # Multi-head attention module
                DropPath(0.2)  # Add DropPath
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),  # Layer normalization
                FeedForwardBlock(emb_size, expansion=forward_expansion),    # Feed-forward module
                DropPath(0.2)  # Add DropPath
            )
            ))

## Inherits from nn.Sequential, stacks multiple TransformerEncoderBlocks to form a TransformerEncoder
class TransformerEncoder(nn.Sequential):
    ## depth defines the number of TransformerEncoderBlocks to stack
    def __init__(self, depth: int = 4, emb_size = 64, **kwargs):
        ## __init__ calls the parent constructor, combining multiple TransformerEncoderBlocks
        ## Uses a list comprehension to create a list containing depth TransformerEncoderBlocks, which is unpacked and passed to the parent constructor
        super().__init__(
            nn.Dropout(0.25),
            *[TransformerEncoderBlock(emb_size, **kwargs) for _ in range(depth)],
            nn.LayerNorm(emb_size)
        )

## The final layer of the model, maps Transformer output to class labels
class ClassificationHead(nn.Module):
    # def __init__(self, emb_size=64, num_classes_per_token=[60, 60, 60, 60, 60, 60, 60, 60], dropout_rate=0.25):
    def __init__(self, emb_size=64, num_classes_per_token=[60, 60, 60, 60, 60, 60, 60, 60], dropout_rate=0.25):
        """
        :param emb_size: Embedding dimension
        :param num_classes_per_token: List of the number of classes for each CLS token
        :param dropout_rate: Dropout rate
        """
        super(ClassificationHead, self).__init__()
        self.num_cls_tokens = len(num_classes_per_token)  # Determine the number of CLS tokens based on the list

        # Define an independent classification network for each CLS token
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, emb_size * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(emb_size * 4, emb_size * 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(emb_size * 2, num_classes)  # Final classification layer
            ) for num_classes in num_classes_per_token
        ])

    def forward(self, x):
        cls_token_first = x[:, :self.num_cls_tokens//2, :]  # shape: (batch_size, num_cls_tokens, emb_size)
        cls_token_last = x[:, -self.num_cls_tokens//2:, :]  # shape: (batch_size, num_cls_tokens, emb_size)
        cls_tokens = torch.cat((cls_token_first, cls_token_last), dim=1) # shape: (batch_size, num_cls_tokens * emb_size)
        outputs = []

        # Iterate over the classification task for each CLS token
        for i, head in enumerate(self.heads):
            outputs.append(head(cls_tokens[:, i, :]))  # Classification output for each CLS token
        combined_outputs = torch.cat([output for output in outputs], dim=-1)

        return combined_outputs


class cnn_trans(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,  # Number of input channels
                patch_size_x: int = 1,  # Patch size in x-direction
                patch_size_y: int = 6,  # Patch size in y-direction
                emb_size: int = 128,  # Embedding dimension
                img_size: int = 120,  # Original image size
                depth: int = 4,  # Transformer encoder depth
                n_classes: int = 121,  # Number of classification categories
                num_cls_tokens: int = 8,  # Parameter to increase the number of CLS tokens
                **kwargs):
        super().__init__(
            Conv_Layers(in_channels),  # Convolution module
            PatchEmbedding(in_channels, patch_size_x, patch_size_y, emb_size, img_size, num_cls_tokens=num_cls_tokens),  # Patch embedding module
            TransformerEncoder(depth, emb_size = emb_size, **kwargs),  # Transformer encoder
            ClassificationHead(emb_size)  # Classification head
        )