import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import vision_transformer as vits
from vision_transformer import DINOHead
from torch.nn import init
import argparse


import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots
    
class decoder(nn.Module):
    def __init__(self,patch_size,embed_dim,din_hidden,dim_out) -> None:
        super().__init__()
        self.hw =224//patch_size
        self.dim_out =dim_out
        self.pos_embed = nn.Parameter(torch.zeros(1,self.hw,self.hw,embed_dim))  # learned positional encoding
        self.mlp = nn.Sequential(nn.Linear(embed_dim,din_hidden),nn.ReLU(inplace=True),nn.Linear(din_hidden,din_hidden),
                                 nn.Linear(din_hidden,din_hidden),nn.Linear(din_hidden,dim_out+1))#token-wise mlp
        
    def forward(self, slots):
        #slots的维度：B K D       B N*D   slots_tokens  B N KD  #每个slot(D) 由n个tokens组成  DN
        B,K,D = slots.shape#K=embed_dim
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1,self.hw,self.hw, 1))#`slots` shape: [batch_size*num_slots, width_init：14, height_init：14, slot_size].
        pos_embed = self.pos_embed.expand(B*K,self.hw,self.hw,D) #shape: [batch_size*num_slots, width_init：14, height_init：14, slot_size].
        slots = slots +pos_embed #shape: [batch_size*num_slots, width_init：14, height_init：14, slot_size].
        y_k = self.mlp(slots)  #shape: [batch_size*num_slots, width_init：14, height_init：14, dim_out+1].
        recons, masks = y_k .reshape(B, -1, y_k.shape[1], y_k.shape[2], y_k.shape[3]).split([768,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].
        return recon_combined      
    
# slots = torch.rand(1,3,256)
# d = decoder(196,256,1024,768)
# y =d(slots)
# print(y.shape)
# Vit 输出 BND  torch.Size([1, 768, 14, 14])

class dinosaur(nn.Module):
    def __init__(self, encoder_t,encoder_s, args):
        super().__init__()
        self.num_iterations = args.num_iterations
        self.num_prototypes = args.num_prototypes#聚类中心数量的初始化
        self.dim_hidden = args.dim_hidden
        self.patch_size =args.patch_size
        # self.teacher_momentum = args.teacher_momentum

        if args.arch in ('resnet18', 'resnet34'):
            self.dim_out = 512
        elif args.arch in ('vit_tiny'):
            self.dim_out = 192
        elif args.arch in ('vit_small'):
            self.dim_out = 384
        elif args.arch in ('vit_base'):
            self.dim_out = 768
        else:
            self.dim_out = 2048
        
        self.encoder_s = encoder_s(self.patch_size)
        self.encoder_t = encoder_t(self.patch_size)
        self.decoder_s = decoder(self.patch_size,self.dim_out,self.dim_hidden,self.dim_out)
        self.slotattention = SlotAttention(self.num_prototypes, self.dim_out,self.num_iterations,hidden_dim= self.dim_hidden)
        
    def forward(self, input):
        with torch.no_grad():
            patch_Features = self.encoder_t(input)#B  ND(n个slot，每个维度是d)
            hw = int(patch_Features.size(1)**0.5)#得到map的h与w的大小，为后续恢复feature map
            patch_Features = patch_Features.transpose(2, 1)
            patch_Features = patch_Features.view(patch_Features.size(0),patch_Features.size(1),hw,hw)#bd h*w  ---bdhw
            x = self.encoder_s(input)
        Slots = self.slotattention(x)
        rec_patch_Features = self.decoder_s(Slots)
        loss =  nn.MSELoss()
        loss = loss(rec_patch_Features,patch_Features)
        return loss
def get_parser():
    parser = argparse.ArgumentParser('dinosaur')

    # dataset
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=1024, help='d')
    parser.add_argument('--num_prototypes', type=int, default=10, help='')
    parser.add_argument('--arch', type=str, default='vit_base', help='')
    parser.add_argument('--patch_size', type=int, default='16', help='')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd', help='optimizer choice')

    args = parser.parse_args()
    return args 
def build_model(args):
    teacher = vits.__dict__[args.arch]
    if args.arch in vits.__dict__.keys():
        encoder_s = vits.__dict__[args.arch]
    model =dinosaur(teacher,encoder_s,args)#用的ViT，encoder输出的是c h*w d，需要将其还原回到hw
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            # lr=args.batch_size * args.world_size / 256 * args.base_lr,
            lr =0.1,
            momentum=0.9,
            weight_decay=0.01)
    else:
        raise NotImplementedError
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    return model, optimizer

if __name__ == '__main__':
    args = get_parser()
    # teacher =torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    # student =torch.hub.load('facebookresearch/dino:main', 'dino_vits16')   
    x = torch.rand(1,3,224,224)
    model, optimizer = build_model(args)
    y =model(x)
    print(y.item())
