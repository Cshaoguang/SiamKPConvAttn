import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from einops.layers import torch as ELT
from backbone import KPConvFPN

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ff_dim=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)   
        x = self.layernorm1(x)    
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layernorm2(x)    
        return x
    
class EnhancedFeatureFusion(nn.Module):
    def __init__(self, in_features):
        super(EnhancedFeatureFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),    
            nn.Tanh(),                              
            nn.Linear(in_features, 1),            
            nn.Softmax(dim=0)                     
        )
        self.expand = nn.Linear(1, 1024)  
        
    def forward(self, x):                      
        weights = self.attention(x)                
        weighted_sum = (x * weights).sum(dim=0, keepdim=True)              
        return self.expand(weighted_sum) 

class ChangeNetwork(nn.Module):
  def __init__(self, cfg):
    super(ChangeNetwork, self).__init__()
    self.backbone =  KPConvFPN(cfg.backbone.input_dim,
                               cfg.backbone.output_dim,
                               cfg.backbone.init_dim,
                               cfg.backbone.kernel_size,
                               cfg.backbone.init_radius,
                               cfg.backbone.init_sigma,
                               cfg.backbone.group_norm)  
    self.transformer = TransformerBlock(cfg.backbone.output_dim)
    self.global_max_pooling = ELT.Reduce("N d -> d", "max")
    self.feature_fusion = EnhancedFeatureFusion(cfg.backbone.output_dim)
    self.sequential = nn.Sequential(nn.Linear(cfg.backbone.output_dim, 512),
                                    nn.Dropout(p=0.2),
                                    nn.LeakyReLU(0.01), 
                                    nn.Linear(512, 256),
                                    nn.Dropout(p=0.2),
                                    nn.LeakyReLU(0.01), 
                                    nn.Linear(256, 256)) 
    self.final_decision = nn.Linear(256, 5)
   
  def forward(self, data_dict):
    output_dict = {}
    feats = data_dict['features'].detach()
    ref_length = data_dict['lengths'][-1][0].item()
    feats = self.backbone(feats, data_dict)
    feats = self.transformer(feats.unsqueeze(0)).squeeze(0)
    ref_feats = feats[:ref_length]
    src_feats = feats[ref_length:]
    ref_feats = self.global_max_pooling(ref_feats)
    src_feats = self.global_max_pooling(src_feats)
    diff = (ref_feats - src_feats)
    diff = self.feature_fusion(diff)
    out = self.sequential(diff)
    out = torch.flatten(out)
    out = self.final_decision(out)
    output_dict['output'] = out
    return output_dict

def create_model(cfg):
    model = ChangeNetwork(cfg)
    return model

def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)

if __name__ == '__main__':
    main()
