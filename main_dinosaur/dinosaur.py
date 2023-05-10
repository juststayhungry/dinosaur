import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import vision_transformer as vits
from vision_transformer import DINOHead
from torch.nn import init
import argparse

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):

        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device, dtype = dtype)

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
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
    
class decoder(nn.Module):
    def __init__(self,num_patches,embed_dim,din_hidden,dim_out) -> None:
        super().__init__()
        self.num_patches =num_patches
        self.dim_out =dim_out
        self.map_alpha = nn.Parameter(torch.zeros(num_patches,dim_out))#signifies where the slot is active
        self.pos_embed = nn.Parameter(torch.zeros(num_patches,embed_dim))  # learned positional encoding
        init.xavier_uniform_(self.map_alpha)

        self.mlp = nn.Sequential(nn.Linear(embed_dim,din_hidden),nn.ReLU(inplace=True),nn.Linear(din_hidden,dim_out))#token-wise mlp
        
    def forward(self, slots):
        #slots的维度：B K D       B N*D   slots_tokens  B N KD  #每个slot(D) 由n个tokens组成  DN
        B,K,D = slots.shape
        slots_tokens = slots.repeat(1,1,self.num_patches).reshape(B,K,self.num_patches,D)
        # slots_tokens = slots.expand(self.num_patches,B,K,D).reshape(B,K,self.num_patches,D)
        pos_embed = self.pos_embed.expand(B,K,self.num_patches,D) #BKND
        slots_tokens = slots_tokens +pos_embed #BKND
        y_k = self.mlp(slots_tokens)  #BKNdim_out
        map_alpha = self.map_alpha.expand(B,K,self.num_patches,self.dim_out)  #BKNdim_out
        m_k = map_alpha.softmax(dim=1)#BKNdim_out
        y = torch.einsum('bknd,bknd->bnd',y_k,m_k)
        return y       #bnd
# slots = torch.rand(1,3,256)
# d = decoder(196,256,512,768)
# y =d(slots)
# print(y.shape)
#Vit 输出 BND  1 196 768

class dinosaur(nn.Module):
    def __init__(self, encoder_t,encoder_s, args):
        super().__init__()
        self.num_iterations = args.num_iterations
        self.num_prototypes = args.num_prototypes#聚类中心数量的初始化
        self.dim_hidden = args.dim_hidden
        self.num_patches =args.num_patches
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
        self.decoder_s = decoder(self.num_patches,self.dim_out,self.dim_hidden,self.dim_out)
        self.slotattention = SlotAttention(self.num_prototypes, self.dim_out,self.num_iterations,hidden_dim= self.dim_hidden)
        
    def forward(self, input):
        with torch.no_grad():
            patch_Features = self.encoder_t(input)#B  ND(n个slot，每个维度是d)
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
    parser.add_argument('--dim_hidden', type=int, default=512, help='d')
    parser.add_argument('--num_patches', type=int, default=196, help='')
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
