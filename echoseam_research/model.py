"""Untrained global/local localization research prototype, not a detector."""
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,channels,3,padding=1),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1),nn.GELU(),nn.Conv2d(channels,1,1))
    def forward(self,x):
        return self.net(x)

class GlobalLocalPrototype(nn.Module):
    """Inputs: one RGB tensor (1,3,H,W). Outputs: score logit and dense logits.

    All tiles are evaluated, ensuring coverage; this is a research control for
    testing proposal-based crop selection later. Parameters are untrained.
    """
    def __init__(self, global_size=256, tile_size=256, stride=192, channels=32):
        super().__init__()
        if not 0 < stride <= tile_size: raise ValueError('stride must be in (0, tile_size]')
        self.global_size,self.tile_size,self.stride=global_size,tile_size,stride
        self.global_net=Encoder(channels)
        self.local_net=Encoder(channels)
        self.score_head=nn.Linear(2,1)
    @staticmethod
    def starts(length,size,stride):
        if length <= size: return [0]
        points=list(range(0,length-size+1,stride))
        if points[-1]!=length-size: points.append(length-size)
        return points
    def forward(self,x):
        if x.ndim!=4 or x.shape[0]!=1 or x.shape[1]!=3: raise ValueError('Expected (1,3,H,W)')
        h,w=x.shape[-2:]
        if not h or not w: raise ValueError('Empty image')
        g=F.interpolate(x,size=(self.global_size,self.global_size),mode='bilinear',align_corners=False)
        global_logits=F.interpolate(self.global_net(g),size=(h,w),mode='bilinear',align_corners=False)
        accum=torch.zeros_like(global_logits); counts=torch.zeros_like(global_logits)
        th,tw=min(h,self.tile_size),min(w,self.tile_size)
        boxes=[]
        for y in self.starts(h,th,self.stride):
            for z in self.starts(w,tw,self.stride):
                local=self.local_net(x[:,:,y:y+th,z:z+tw])
                accum[:,:,y:y+th,z:z+tw]=accum[:,:,y:y+th,z:z+tw]+local
                counts[:,:,y:y+th,z:z+tw]=counts[:,:,y:y+th,z:z+tw]+1
                boxes.append((z,y,z+tw,y+th))
        local_logits=accum/counts.clamp_min(1)
        fused=(global_logits+local_logits)/2
        features=torch.stack((global_logits.mean((1,2,3)),local_logits.amax((1,2,3))),dim=1)
        return {'score_logit':self.score_head(features),'mask_logits':fused,'global_logits':global_logits,'local_logits':local_logits,'coverage':counts,'boxes':boxes}
