import math
from functools import partial
from typing import List, Tuple,  Optional, Dict
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from torch_uncertainty.layers import PackedLinear
from bayesian_torch.layers import LinearReparameterization
from itertools import combinations
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, VideoMAEConfig


class AttentionFusionModule(nn.Module):
    """
    Fuse embeddings through weighted sum of the corresponding linear projections.
    Linear layer for learning the weights.
    Copied from: https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/attention.py

    Args:
        channel_to_encoder_dim: mapping of channel name to the encoding dimension
        encoding_projection_dim: common dimension to project the encodings to.
        defaults to min of the encoder dim if not set

    """
    def __init__(
        self,
        channel_to_encoder_dim: Dict[str, int],
        encoding_projection_dim: Optional[int] = None,
    ):
        super().__init__()
        attn_in_dim = sum(channel_to_encoder_dim.values())
        self.attention = nn.Sequential(
            nn.Linear(attn_in_dim, len(channel_to_encoder_dim)),
            nn.Softmax(-1),
        )
        if encoding_projection_dim is None:
            encoding_projection_dim = min(channel_to_encoder_dim.values())

        encoding_projection = {}
        for channel in sorted(channel_to_encoder_dim.keys()):
            encoding_projection[channel] = nn.Linear(
                channel_to_encoder_dim[channel], encoding_projection_dim
            )
        self.encoding_projection = nn.ModuleDict(encoding_projection)

    def forward(self, embeddings: Dict[str, Tensor]) -> Tensor:
        concatenated_in = torch.cat(
            [embeddings[k] for k in sorted(embeddings.keys())], dim=-1
        )
        attention_weights = self.attention(concatenated_in)
        projected_embeddings: List[Tensor] = []
        for channel, projection in self.encoding_projection.items():
            projected_embedding = projection(embeddings[channel])
            projected_embeddings.append(projected_embedding)

        for i in range(len(projected_embeddings)):
            projected_embeddings[i] = (
                attention_weights[:, i].unsqueeze(-1) * projected_embeddings[i]
            )

        fused = torch.sum(torch.stack(projected_embeddings), dim=0)
        return fused


class LRUQ_VMAE(torch.nn.Module):
    def __init__(self, 
                 args,
                 model_dim=768,
                 num_estimators = 5,
                 head_type = "linear",
                 num_projections:int =1,
                 dist="cos",
                 label2id = None,
                 id2label = None,):
        super().__init__()
        self.model_dim = model_dim
        self.num_projections = num_projections
        self.model_type = args.model_type
        self.num_classes = args.num_classes
        self.num_estimators = num_estimators
        self.id2label = id2label
        self.label2id = label2id
        self.args = args
        self.head_type = head_type
        self.dist = dist
        self.projections = nn.ModuleList([nn.Linear(self.mask_dim, model_dim) for _ in range(num_projections)])
        self.activation = nn.ReLU()  # nn.LeakyReLU(negative_slope=0.01)
        
        if self.args.exterior and self.args.incabin:
            self.model_ex = self._load_model()
            self.model_in = self._load_model()
            self.fusion = AttentionFusionModule({"in":self.model_dim, 
                                                 "ex":self.model_dim}, 
                                                 self.model_dim)
        else:
            self.model = self._load_model()
        
        self.head = torch.nn.Linear(self.model_dim, self.num_classes)
        
    def _load_model(self,
                    model_ckpt:str="MCG-NJU/videomae-base-finetuned-kinetics"):
        model = VideoMAEForVideoClassification.from_pretrained(
                        model_ckpt, label2id=self.label2id, id2label=self.id2label, ignore_mismatched_sizes=True,)
        model.classifier =  torch.nn.Identity()
        return model
                    
    def forward(self, x):
        if self.args.exterior and self.args.incabin: # multi-modal
            x_in  = x["incabin_video"].permute(0,2,1,3,4)
            x_in = x_in.to(self.args.device)
            x_in = self.model_in(x_in).logits 
            
            x_ex = x["exterior_video"].permute(0,2,1,3,4)
            x_ex = x_ex.to(self.args.device.replace("0", "1")) # just it dump to the second GPU
            x_ex = self.model_ex(x_ex).logits 
            x_ex = x_ex.to(self.args.device)
            z = self.fusion({"in": x_in, "ex": x_ex})

        else: # uni-modal
            x = x.permute(0,2,1,3,4)
            z = self.model(x).logits         
        
        z_representations = [z] 
        y = self.head(z)
        y_projections = []
        
        for i, proj in enumerate(self.projections):
            z_p = self.activation(proj(z))
            z_representations.append(z_p)
            y_projections.append(self.head(z_p))

        if self.training:
            # apply softmax to the original y and the y_projections
            y_softmax =  [F.softmax(x, dim=-1) for x in [y]+y_projections]
            
            return  y, y_projections
        else:
            return y, y_projections
