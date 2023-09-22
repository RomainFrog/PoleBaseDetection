from PIL import Image
import requests
import matplotlib.pyplot as plt
# config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import torch.nn.functional as F

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETROAP(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, backbone, transformer, num_classes=1, num_queries=100, aux_loss=False):
        super().__init__()
        self.backbone = backbone
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        self.num_queries = num_queries
        self.query_embded = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.points_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.aux_loss = aux_loss

    def forward(self, inputs: NestedTensor):
    
        # Check the data type of inputs and convert to NestedTensor if necessary
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Get the features from the backbone (a.k.a. Joiner of features and position)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None, 'mask should not be None'
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]


        

        # construct positional encodings
        bs, _, H, W = h.shape[-4:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        pos = pos.expand(pos.shape[0], bs, pos.shape[-1])
        # upscale pos to match batch size
        print(f"{pos.shape=}")


        # propagate through the transformer
        print(f"{h.flatten(2).permute(2, 0, 1).shape=}")
        print(f"{pos.shape=}")
        print(f"{self.query_pos.unsqueeze(1).transpose(0,1).shape=}")
        # h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)).transpose(0, 1)
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1).expand(bs,100,256).transpose()).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_points': self.linear_point(h).sigmoid()}
    

def build_DETR(args):
    num_classes = 1 # only one class: pole
    model = DETROAP(num_classes=num_classes, hidden_dim=args.hidden_dim, 
                    nheads=args.nheads, num_encoder_layers=args.enc_layers, 
                    num_decoder_layers=args.dec_layers)
    
    return model
