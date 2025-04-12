import torch
import torch.nn as nn

class Cross_Attention_DDG(nn.Module):
    
    def __init__(self, base_module, **transf_parameters):
        super().__init__()
        self.base_ddg = base_module(**transf_parameters).to(device)
    
    def forward(self, x_wild, x_mut, length):

        delta_x = x_wild - x_mut
        output_TCA = self.base_ddg(delta_x, x_wild, length)

        # inv Janus
        delta_x_inv = x_mut -x_wild
        output_TCA_inv = self.base_ddg(delta_x_inv, x_mut, length)
        
        return (output_TCA - output_TCA_inv)/2


def apply_masked_pooling(position_attn_output, padding_mask):

    # Convert mask to float for element-wise multiplication
    padding_mask = padding_mask.float()

    # Global Average Pooling (GAP) - Exclude padded tokens
    # Sum only over valid positions (padding_mask is False for valid positions)
    sum_output = torch.sum(position_attn_output * (1 - padding_mask.unsqueeze(-1)), dim=1)  # (batch_size, feature_dim)
    valid_count = torch.sum((1 - padding_mask).float(), dim=1)  # (batch_size,)
    gap = sum_output / valid_count.unsqueeze(-1)  # Divide by number of valid positions

    # Global Max Pooling (GMP) - Exclude padded tokens
    # Set padded positions to -inf so they don't affect the max computation
    position_attn_output_masked = position_attn_output * (1 - padding_mask.unsqueeze(-1)) + (padding_mask.unsqueeze(-1) * (- 1e10))
    gmp, _ = torch.max(position_attn_output_masked, dim=1)  # (batch_size, feature_dim)

    return gap, gmp


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=3700):    

        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegression(nn.Module):
    def __init__(self, input_dim=1280, num_heads=8, dropout_rate=0., f_activation = nn.ReLU(), kernel_size=20):
        
        super(TransformerRegression, self).__init__()

        self.embedding_dim = input_dim
        self.act = f_activation
        self.max_len = 3700 
        out_channels = 128  #num filtri conv 1D
        kernel_size = 20
        padding = 0
        
        self.conv1d = nn.Conv1d(in_channels=self.embedding_dim, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, 
                                             padding=padding) 
        
        self.conv1d_wild = nn.Conv1d(in_channels=self.embedding_dim, 
                                             out_channels=out_channels, 
                                             kernel_size=kernel_size, 
                                             padding=padding)

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Cross-attention layers
        self.positional_encoding = SinusoidalPositionalEncoding(out_channels, 3700)
        self.speach_att_type = True
        self.multihead_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout_rate, batch_first=True )
        self.inverse_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, dropout=dropout_rate, batch_first =True)
        
        dim_position_wise_FFN = out_channels*2

        self.norm3 = nn.LayerNorm(dim_position_wise_FFN)
        self.router = nn.Linear(dim_position_wise_FFN, 1)

        self.pw_ffnn = nn.Sequential(
            nn.Linear(dim_position_wise_FFN, 512),
            self.act,
            nn.Linear(512, dim_position_wise_FFN)
            )
        
        self.Linear_ddg = nn.Linear(dim_position_wise_FFN*2, 1)

    
    def create_padding_mask(self, length, seq_len, batch_size):
        """
        Create a padding mask for multihead attention.
        length: Tensor of shape (batch_size,) containing the actual lengths of the sequences.
        seq_len: The maximum sequence length.
        batch_size: The number of sequences in the batch.
        
        Returns a padding mask of shape (batch_size, seq_len).
        """
        mask = torch.arange(seq_len, device=length.device).unsqueeze(0) >= length.unsqueeze(1)
        return mask

    
    def forward(self, delta_w_m, x_wild, length):
            
            delta_w_m = delta_w_m.transpose(1, 2)  # (batch_size, feature_dim, seq_len) -> (seq_len, batch_size, feature_dim)
            C_delta_w_m = self.conv1d(delta_w_m)
            C_delta_w_m = C_delta_w_m.transpose(1, 2)  # (seq_len, batch_size, feature_dim) -> (batch_size, seq_len, feature_dim)
            C_delta_w_m = self.positional_encoding(C_delta_w_m)
            
            x_wild = x_wild.transpose(1, 2)  # (batch_size, feature_dim, seq_len) -> (seq_len, batch_size, feature_dim)
            C_x_wild = self.conv1d_wild(x_wild)
            C_x_wild = C_x_wild.transpose(1, 2)  # (seq_len, batch_size, feature_dim) -> (batch_size, seq_len, feature_dim)
            C_x_wild = self.positional_encoding(C_x_wild)            
            
            batch_size, seq_len, feature_dim = C_x_wild.size()

            padding_mask = self.create_padding_mask(length, seq_len, batch_size)        
                    
            if self.speach_att_type:
                print('ATTENTION TYPE: Dual cross Attention\n q = wild , k = delta, v = delta and q = delta , k = wild, v = wild \n ----------------------------------')
                self.speach_att_type = False
                
            direct_attn_output, _ = self.multihead_attention(C_x_wild, C_delta_w_m, C_delta_w_m, key_padding_mask=padding_mask)
            direct_attn_output += C_delta_w_m 
            direct_attn_output = self.norm1(direct_attn_output)                        
            
            inverse_attn_output, _ = self.inverse_attention(C_delta_w_m, C_x_wild, C_x_wild, key_padding_mask=padding_mask)                   
            inverse_attn_output += C_x_wild  
            inverse_attn_output = self.norm2(inverse_attn_output)
            
            attn_output = torch.cat([direct_attn_output, inverse_attn_output], dim=-1)

            output = self.pw_ffnn(attn_output)
    
            position_attn_output = attn_output + output
    
            position_attn_output = self.norm3(position_attn_output)
    
            gap, gmp = apply_masked_pooling(position_attn_output, padding_mask)
    
            # Concatenate GAP and GMP
            pooled_output = torch.cat([gap, gmp], dim=-1)  # (batch_size, 2 * feature_dim)
    
            # Pass through FFNN to predict DDG
            x = self.Linear_ddg(pooled_output)        
            
            return x.squeeze(-1)