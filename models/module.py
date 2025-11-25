

import itertools

import torch
from torch import nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])
         
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads = 4, dropout = 0., attn_bias=False, resolution=(5, 5)):
        super().__init__()
        
        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        
        dim_head = emb_dim // heads 
        project_out = not (heads == 1) 

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias = False) 

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.attn_bias = attn_bias
        if attn_bias:
            points = list(itertools.product(
                range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(
                torch.zeros(heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs',
                                torch.LongTensor(idxs).view(N, N),
                                persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        if self.attn_bias:
            super().train(mode)
            if mode and hasattr(self, 'ab'):
                del self.ab
            else:
                self.ab = self.attention_biases[:, self.attention_bias_idxs]
        
    def forward(self, x, mask = None, return_attn=False):
        # qkv = self.to_qkv(x) # b x n x d*3
        
        # qkv = rearrange(qkv, 'b n (h d a) -> b n a h d', h = self.heads, a=3)
        # out = flash_attn_qkvpacked_func(qkv, self.drop_p, softmax_scale=None, causal=False)
        # out = rearrange(out, 'b n h d -> b n (h d)')
        
        # return self.to_out(out)
        
        qkv = self.to_qkv(x).chunk(3, dim = -1) 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) 

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        if self.attn_bias:
            qk += (self.attention_biases[:, self.attention_bias_idxs]
            if self.training else self.ab)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float16).min
            ind_mask = mask.shape[-1]
            qk[:,:,-ind_mask:,-ind_mask:] = qk[:,:,-ind_mask:,-ind_mask:].masked_fill(mask==0, fill_value)

        attn_weights = self.attend(qk) # b h n n
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)
        
        out = torch.matmul(attn_weights, v) 
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        if return_attn:
            return self.to_out(out), attn_weights_averaged[:,0]
        else:
            return self.to_out(out)
        

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads = 4, dropout = 0., attn_bias=False):
        super().__init__()
        
        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        
        dim_head = emb_dim // heads 
        project_out = not (heads == 1) 

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        
        self.to_q = nn.Linear(emb_dim, emb_dim, bias = False) 
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias = False) 

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x_q, x_kv, mask = None, return_attn=False):
        # qkv = self.to_qkv(x) # b x n x d*3
        
        # qkv = rearrange(qkv, 'b n (h d a) -> b n a h d', h = self.heads, a=3)
        # out = flash_attn_qkvpacked_func(qkv, self.drop_p, softmax_scale=None, causal=False)
        # out = rearrange(out, 'b n h d -> b n (h d)')
        
        # return self.to_out(out)
        
        q = self.to_q(x_q)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        kv = self.to_kv(x_kv).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv) 

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        
        if mask is not None:
            fill_value = torch.finfo(torch.float16).min
            ind_mask = mask.shape[-1]
            qk[:,:,-ind_mask:,-ind_mask:] = qk[:,:,-ind_mask:,-ind_mask:].masked_fill(mask==0, fill_value)

        attn_weights = self.attend(qk) # b h n n
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)
        
        out = torch.matmul(attn_weights, v) 
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        if return_attn:
            return self.to_out(out), attn_weights_averaged[:,0]
        else:
            return self.to_out(out)


class CustomMultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0., attn_bias=False):
        super().__init__()
        
        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'
        
        dim_head = emb_dim // heads 
        project_out = not (heads == 1)  # Only project output if heads > 1

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5  # Scaling factor for attention scores
        self.attend = nn.Softmax(dim=-1)  # Softmax over the last dimension
        
        # Linear layers for query and key/value projections
        self.to_q = nn.Linear(emb_dim, emb_dim, bias=attn_bias) 
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias=attn_bias) 

        # Output projection layer (if needed)
        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, x_q, x_kv, return_attn=False):
        """
        Forward pass for multi-head cross-attention without mask.
        
        Args:
            x_q (torch.Tensor): Query input, shape [batch, 250, emb_dim].
            x_kv (torch.Tensor): Key/Value input, shape [batch, emb_dim].
            return_attn (bool): If True, return attention weights alongside output.
        
        Returns:
            torch.Tensor: Output tensor, shape [batch, 250, emb_dim].
            (torch.Tensor, torch.Tensor): If return_attn=True, returns (output, attn_weights), 
                                          where attn_weights has shape [batch, 250, 1].
        """
        # Query processing: [batch, 250, emb_dim] -> [batch, 250, emb_dim]
        q = self.to_q(x_q)
        # Reshape for multi-head: [batch, 250, emb_dim] -> [batch, heads, 250, dim_head]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        # Key/Value processing: [batch, emb_dim] -> [batch, 1, emb_dim]
        x_kv = x_kv.unsqueeze(1)  # Add sequence dimension: [batch, 1, emb_dim]
        # Linear projection: [batch, 1, emb_dim] -> [batch, 1, emb_dim * 2]
        kv = self.to_kv(x_kv)
        # Split into key and value: two tensors of shape [batch, 1, emb_dim]
        k, v = kv.chunk(2, dim=-1)
        # Reshape for multi-head: [batch, 1, emb_dim] -> [batch, heads, 1, dim_head]
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)

        # Compute attention scores: [batch, heads, 250, dim_head] @ [batch, heads, dim_head, 1] -> [batch, heads, 250, 1]
        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale 
        
        # Apply softmax to get attention weights: [batch, heads, 250, 1]
        attn_weights = self.attend(qk)
        
        # Compute output: [batch, heads, 250, 1] @ [batch, heads, 1, dim_head] -> [batch, heads, 250, dim_head]
        out = torch.matmul(attn_weights, v)
        # Reshape back to original dimension: [batch, heads, 250, dim_head] -> [batch, 250, emb_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        # Return attention weights if requested
        if return_attn:
            # Average over heads: [batch, heads, 250, 1] -> [batch, 250, 1]
            attn_weights_averaged = attn_weights.mean(dim=1)
            return self.to_out(out), attn_weights_averaged
        else:
            return self.to_out(out)

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., attn_bias=False, resolution=(5,5)):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads = heads, dropout = dropout, attn_bias=attn_bias, resolution=resolution)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x, mask=mask, return_attn=return_attn)
                x += attn_out # residual connection after attention      
                x = ff(x) + x # residual connection after feed forward net
                
            else:
                x = attn(x, mask=mask) + x # residual connection after attention      
                x = ff(x) + x # residual connection after feed forward net
            
        if return_attn:
            return x, attn_weights
        else:
            return x


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., attn_bias=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads = heads, dropout = dropout, attn_bias=attn_bias)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x_q, x_kv, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, mask=mask, return_attn=return_attn)
                x_q += attn_out # residual connection after attention      
                x_q = ff(x_q) + x_q # residual connection after feed forward net
            else:
                x_q = attn(x_q, x_kv=x_kv, mask=mask) + x_q
                x_q = ff(x_q) + x_q # residual connection after feed forward net

        if return_attn:
            return x_q, attn_weights
        else:
            return x_q


class CustomCrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., attn_bias=False):
        """
        Initialize the CrossEncoder with multiple layers of cross-attention and feed-forward networks.
        
        Args:
            emb_dim (int): Embedding dimension.
            depth (int): Number of layers.
            heads (int): Number of attention heads.
            mlp_dim (int): Hidden dimension of the feed-forward network.
            dropout (float): Dropout rate, default 0.
            attn_bias (bool): Whether to use bias in attention linear layers, default False.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, CustomMultiHeadCrossAttention(emb_dim, heads=heads, dropout=dropout, attn_bias=attn_bias)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x_q, x_kv, return_attn=False):
        """
        Forward pass through the CrossEncoder.
        
        Args:
            x_q (torch.Tensor): Query input, shape [batch, 250, emb_dim].
            x_kv (torch.Tensor): Key/Value input, shape [batch, emb_dim].
            return_attn (bool): If True, return the attention weights from the last layer.
        
        Returns:
            torch.Tensor: Output tensor, shape [batch, 250, emb_dim].
            (torch.Tensor, torch.Tensor): If return_attn=True, returns (output, attn_weights),
                                          where attn_weights has shape [batch, 250, 1].
        """
        attn_weights = None  # To store attention weights if return_attn is True
        
        for attn, ff in self.layers:
            if return_attn:
                # Apply attention with residual connection and get attention weights
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, return_attn=True)
                x_q = x_q + attn_out  # Residual connection after attention
                x_q = ff(x_q) + x_q   # Residual connection after feed-forward
            else:
                # Apply attention with residual connection
                x_q = attn(x_q, x_kv=x_kv) + x_q
                x_q = ff(x_q) + x_q   # Residual connection after feed-forward

        if return_attn:
            return x_q, attn_weights
        else:
            return x_q


class PEGH(nn.Module):
    def __init__(self, dim=512, kernel_size=3):
        super(PEGH, self).__init__()
        
        self.proj1 = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, bias=True, groups=dim)
        
    def forward(self, x, pos):

        pos = pos - pos.min(0)[0]
        x_sparse = torch.sparse_coo_tensor(pos.T , x.squeeze())
        x_dense = x_sparse.to_dense().permute(2,1,0).unsqueeze(dim=0)
        
        x_pos = self.proj1(x_dense)
            
        mask = (x_dense.sum(dim=1) != 0.)
        x_pos = x_pos.masked_fill(~mask, 0.) + x_dense
        x_pos_sparse = x_pos.squeeze().permute(2,1,0).to_sparse(2)
        x_out = x_pos_sparse.values().unsqueeze(dim=0)
        
        return x_out


class GlobalEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., kernel_size=3):
        super().__init__()      
        
        self.pos_layer = PEGH(dim=emb_dim, kernel_size=kernel_size) 
        
        self.layer1 = TransformerEncoder(emb_dim, 1, heads, mlp_dim, dropout)
        self.layer2 = TransformerEncoder(emb_dim, depth-1, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)
        
    def foward_features(self, x, pos):
        # Translayer x1
        x = self.layer1(x) #[B, N, 384]

        # PEGH
        x = self.pos_layer(x, pos) #[B, N, 384]
        
        # Translayer x (depth-1)
        x = self.layer2(x) #[B, N, 384]        
        x = self.norm(x) 
        
        return x
        
    def forward(self, x, position):    
        x = self.foward_features(x, position) # 1 x N x 384
    
        return x
    
    
class NeighborEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout = 0., resolution=(5,5)):
        super().__init__()      
        
        self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, attn_bias=True, resolution=resolution)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            
        # Translayer
        x = self.layer(x, mask=mask) #[B, N, 512]
        x = self.norm(x)
        
        return x


class FusionEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()      
                
        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x_t=None, x_n=None, x_g=None, mask=None):
            
        if mask != None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            
        # Target token
        fus1 = self.fusion_layer(x_g.unsqueeze(1), x_t) 
            
        # Neighbor token
        fus2 = self.fusion_layer(x_g.unsqueeze(1), x_n, mask=mask) 
                
        fusion = (fus1 + fus2).squeeze(1)            
        fusion = self.norm(fusion)
        
        return fusion


class ExpressionEncoder(nn.Module):
    def __init__(
        self,
        num_genes,
        embed_dim,
        dropout=0.
    ):
        super().__init__()
        self.projection = nn.Linear(num_genes, embed_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)

        return x