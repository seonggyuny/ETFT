import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import os 
from timm.models.layers import trunc_normal_
import numpy as np
from segm.model.vit import PatchEmbedding
from segm.model.blocks import Block, FeedForward, CABlock
from segm.model.utils import init_weights
import mmcv
import torch.nn.functional as F
from mmseg.ops import resize

from icecream import ic

def generate_random_orthogonal_matrix(feat_in, num_classes):
        rand_mat = np.random.random(size=(feat_in, num_classes))
        orth_vec, _ = np.linalg.qr(rand_mat)
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
            "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
        return orth_vec
    
def gumbel_sigmoid(logits, tau):
    r"""Straight-through gumbel-sigmoid estimator
    """
    gumbels = -torch.empty_like(logits,
                                memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.sigmoid()
    
    # Straight through.
    y_hard = (y_soft > 0.5).long()
    ret = y_hard - y_soft.detach() + y_soft
    
    return ret    
    
    
    

class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

class ETFT(nn.Module): 
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
      

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)
        
        
        self.proj_classifier = nn.Sequential(
          nn.Linear(2 * d_model, d_model),
          nn.BatchNorm1d(num_features=self.n_cls),
          nn.ReLU(),
          nn.Linear( d_model, d_model),
          nn.BatchNorm1d(num_features=self.n_cls),
          nn.ReLU(),
        )
        
        self.apply(init_weights)
        
        file_path = f"/home/sg/project/experiments/etf_{d_model}_{n_cls}_class.pth"
        
        if os.path.exists(file_path):
            self.etf = torch.load(f"/home/sg/project/experiments/etf_{d_model}_{n_cls}_class.pth", map_location='cpu')
            print("ETF file exists. Loaded ETF.")
          
            
        else : 
            print("Creating ETF...")
            orth_vec= generate_random_orthogonal_matrix(d_model,n_cls)
            i_nc_nc = torch.eye(n_cls)
            one_nc_nc: torch.Tensor = torch.mul(torch.ones(n_cls, n_cls), (1 / n_cls))
            self.etf = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(n_cls / (n_cls - 1)))
            torch.save(self.etf,f"/home/sg/project/experiments/etf_{d_model}_{n_cls}_class.pth")
            
        
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.etf.detach().expand(x.shape[0],self.d_model,self.n_cls).to('cuda')
        cls_emb = cls_emb.permute(0,2,1)
        
        
        x = torch.cat((x, cls_emb), 1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, learnable_classifier = x[:, : -self.n_cls], x[:, -self.n_cls : ]
        patches = patches @ self.proj_patch
        patches = patches / patches.norm(dim=-1, keepdim=True)
        
        fixed_classifier = cls_emb
        mixing_classifier = torch.cat((learnable_classifier, fixed_classifier), dim = 2)
        mixing_classifier = self.proj_classifier(mixing_classifier)   
        mask = patches @ mixing_classifier.permute(0,2,1)
        masks = self.mask_norm(mask)
      
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
     
        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.etf.detach().expand(x.shape[0],self.d_model,self.n_cls).to('cuda')
        cls_emb = cls_emb.permute(0,2,1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

