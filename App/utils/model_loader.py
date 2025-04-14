import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
from einops import rearrange

# Helper components
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Shortcut connection
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.activation(x + residual)
        return self.dropout(x)

# Main Model
class EnhancedFusionModel(nn.Module):
    def __init__(self, tab_dim, img_dim, text_dim):
        super().__init__()
        
        # Define common embedding dimension for attention compatibility
        self.common_dim = 64
        
        # Enhanced Branches
        self.tabular_block = nn.Sequential(
            ResidualBlock(tab_dim, 128),
            ResidualBlock(128, self.common_dim)
        )
        
        self.image_block = nn.Sequential(
            ResidualBlock(img_dim, 128),
            ResidualBlock(128, self.common_dim)
        )
        
        # New text branch with IndoBert embeddings
        self.text_block = nn.Sequential(
            ResidualBlock(text_dim, 128),
            ResidualBlock(128, self.common_dim)
        )
        
        # Cross-modal attention between all modalities
        # Tabular to others
        self.tab2img_attn = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        self.tab2text_attn = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        
        # Image to others
        self.img2tab_attn = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        self.img2text_attn = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        
        # Text to others
        self.text2tab_attn = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        self.text2img_attn = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=4)
        
        # Normalization layers
        self.attn_norm_tab = nn.LayerNorm(self.common_dim)
        self.attn_norm_img = nn.LayerNorm(self.common_dim)
        self.attn_norm_text = nn.LayerNorm(self.common_dim)
        
        # Pairwise Bilinear Interactions
        self.bilinear_tab_img = nn.Bilinear(self.common_dim, self.common_dim, 32)
        self.bilinear_tab_text = nn.Bilinear(self.common_dim, self.common_dim, 32)
        self.bilinear_img_text = nn.Bilinear(self.common_dim, self.common_dim, 32)
        
        # Normalization for interactions
        self.interaction_norm_tab_img = nn.LayerNorm(32)
        self.interaction_norm_tab_text = nn.LayerNorm(32)
        self.interaction_norm_img_text = nn.LayerNorm(32)
        
        # Gated Fusion with expanded dimensionality
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.common_dim*3 + 32*3, 6),  # 3 modalities + 3 interactions
            nn.Sigmoid()
        )
        
        # Final Prediction with Skip
        self.final = nn.Sequential(
            nn.Linear(self.common_dim*3 + 32*3, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            ResidualBlock(128, 64),
            nn.Linear(64, 1)
        )

    def forward(self, tab, img, text):
        # Process branches
        tab_feat = self.tabular_block(tab)
        img_feat = self.image_block(img)
        text_feat = self.text_block(text)
        
        # Prepare for attention - shape: [seq_len, batch_size, embedding_dim]
        tab_q = tab_feat.unsqueeze(0)  # [1, batch_size, embed_dim]
        img_q = img_feat.unsqueeze(0)  # [1, batch_size, embed_dim]
        text_q = text_feat.unsqueeze(0)  # [1, batch_size, embed_dim]
        
        # Cross attention from tabular to other modalities
        tab2img, _ = self.tab2img_attn(query=tab_q, key=img_q, value=img_q)
        tab2text, _ = self.tab2text_attn(query=tab_q, key=text_q, value=text_q)
        
        # Cross attention from image to other modalities
        img2tab, _ = self.img2tab_attn(query=img_q, key=tab_q, value=tab_q)
        img2text, _ = self.img2text_attn(query=img_q, key=text_q, value=text_q)
        
        # Cross attention from text to other modalities
        text2tab, _ = self.text2tab_attn(query=text_q, key=tab_q, value=tab_q)
        text2img, _ = self.text2img_attn(query=text_q, key=img_q, value=img_q)
        
        # Combine cross-attention features with residual connections
        tab_attn = self.attn_norm_tab(tab_feat + 0.5 * (tab2img.squeeze(0) + tab2text.squeeze(0)))
        img_attn = self.attn_norm_img(img_feat + 0.5 * (img2tab.squeeze(0) + img2text.squeeze(0)))
        text_attn = self.attn_norm_text(text_feat + 0.5 * (text2tab.squeeze(0) + text2img.squeeze(0)))
        
        # Compute pairwise bilinear interactions
        tab_img_inter = F.gelu(self.interaction_norm_tab_img(
            self.bilinear_tab_img(tab_attn, img_attn)))
        
        tab_text_inter = F.gelu(self.interaction_norm_tab_text(
            self.bilinear_tab_text(tab_attn, text_attn)))
        
        img_text_inter = F.gelu(self.interaction_norm_img_text(
            self.bilinear_img_text(img_attn, text_attn)))
        
        # Concatenate all features and interactions for fusion
        combined = torch.cat([
            tab_attn, img_attn, text_attn,
            tab_img_inter, tab_text_inter, img_text_inter
        ], dim=1)
        
        # Dynamic fusion with gates
        gate_weights = self.fusion_gate(combined)
        
        # Apply gates to features and combine
        tab_weight = gate_weights[:, 0:1].expand_as(tab_attn)
        img_weight = gate_weights[:, 1:2].expand_as(img_attn)
        text_weight = gate_weights[:, 2:3].expand_as(text_attn)
        tab_img_weight = gate_weights[:, 3:4].expand_as(tab_img_inter)
        tab_text_weight = gate_weights[:, 4:5].expand_as(tab_text_inter)
        img_text_weight = gate_weights[:, 5:6].expand_as(img_text_inter)
        
        # Apply weights and concatenate for final prediction
        weighted_combined = torch.cat([
            tab_attn * tab_weight,
            img_attn * img_weight,
            text_attn * text_weight,
            tab_img_inter * tab_img_weight,
            tab_text_inter * tab_text_weight,
            img_text_inter * img_text_weight
        ], dim=1)
        
        return self.final(weighted_combined)