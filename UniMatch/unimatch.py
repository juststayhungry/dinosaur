import argparse
import logging
import os
import pprint
import ipdb
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from collections import Counter
from copy import deepcopy
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
'''
定义selected_label为[u_num,h*w]用于存储像素级别达到阈值的类别，初始化为全-1
通过统计每个batch中样本结果代码如下，其中mask_u_w_f为BHW的达到阈值的类别矩阵(即预测达到阈值的像素显示类别id，未达到阈值位置显示-1)；
            for i,j in enumerate(x_ulb_idx):#每图片单独统计
                selected_label[j] = (mask_u_w_f[i]).flatten()
'''
# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
# parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=False)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    img_u_w_h = cfg['crop_size']
    img_u_w_w =cfg['crop_size']
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    total_u_num = len(trainset_u)
    total_u_pixel_num =total_u_num*img_u_w_h*img_u_w_w

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)


    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        
        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        selected_label = torch.ones((total_u_num,img_u_w_h*img_u_w_w), dtype=torch.long, ) * -1
        
        selected_label = selected_label.cuda()
        class_acc = torch.zeros((cfg['nclass'],)).cuda()
        '''
        selected_label初始化为[-1,-1,-1...]#K
        pred_u_w_counter = Counter(selected_label.tolist())
        统计每类超出阈值的样本数量pred_u_w_counter
        '''
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2,x_ulb_idx),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _,_)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]#

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)#
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])#将preds继续划分为lb与ulb
            pred_u_w_fp = preds_fp[num_lb:]#取unlabel的样本

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)#B K H W
            #两个强增强

            pred_u_w = pred_u_w.detach()#teacher分支需要detach #b k h w
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]#置信度(后续滤噪以及fiexmatch要用)
            mask_u_w = pred_u_w.argmax(dim=1)#将图像分割预测的结果当作mask，像素级别的分类label

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)
            '''
            像素级别的分类
            用了cutmix增强导致两个强增强的伪标签不同
            '''
            # loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            # loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            # loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            # loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            # loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            # loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            # loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            # loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            # loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()
            # selected_label = torch.ones((num_ulb,), dtype=torch.long, ) * -1#tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])类比-1，更新时会为一个类
            
            pred_u_w_counter = Counter(selected_label.flatten().tolist())#得到所有类别的出现的次数，次数越多越容易，阈值越高。[类1:次数]键值对
            p_cutoff = 0.95#初始阈值
            select = conf_u_w.ge(p_cutoff).long()
            if max(pred_u_w_counter.values()) < (total_u_pixel_num+1):  #
                wo_negative_one = deepcopy(pred_u_w_counter)
                if -1 in wo_negative_one.keys():
                    wo_negative_one.pop(-1)
                wo_negative_one.update({-1: 1})
                
                for i in range(cfg['nclass']):                    
                    class_acc[i] = pred_u_w_counter[i] / max(wo_negative_one.values())
                    # class_acc[i] = pred_u_w_counter[i] / max((total_u_pixel_num - sum(pred_u_w_counter.values())),max(pred_u_w_counter.values()))

            T = p_cutoff * (class_acc[mask_u_w.flatten(1)] / (2. - class_acc[mask_u_w.flatten(1)])).reshape(-1,img_u_w_h,img_u_w_w) # convex
            mask = conf_u_w.ge(T) 

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1*(mask& (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * (mask& (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()


            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * (mask & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0
            '''
            已知select == 1矩阵为初始阈值ge得到的bool矩阵 BHW
            mask_u_w矩阵是类别id矩阵
            '''
            mask_u_w = mask_u_w +1#防止类别0作用
            mask_u_w_f = (mask_u_w*select)
            mask_u_w_f = mask_u_w_f -1#恢复原始类别
            # print("阈值处理后的类别矩阵",mask_u_w_f)
            for i,j in enumerate(x_ulb_idx):#每图片单独统计
                # ipdb.set_trace()
                selected_label[j] = (mask_u_w_f[i]).flatten()
 
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = (mask & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
