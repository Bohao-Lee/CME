import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from core.utils import *
from core.cfg import cfg
from numbers import Number
from random import random


def neg_filter(pred_boxes, target, withids=False):
    assert pred_boxes.size(0) == target.size(0)
    if cfg.neg_ratio == 'full':
        inds = list(range(pred_boxes.size(0)))
    elif isinstance(cfg.neg_ratio, Number):
        flags = torch.sum(target, 1) != 0
        flags = flags.cpu().data.tolist()
        ratio = cfg.neg_ratio * sum(flags) * 1. / (len(flags) - sum(flags))
        if ratio >= 1:
            inds = list(range(pred_boxes.size(0)))
        else:
            flags = [0 if f == 0 and random() > ratio else 1 for f in flags]
            inds = np.argwhere(flags).squeeze()
            pred_boxes, target = pred_boxes[inds], target[inds]
    else:
        raise NotImplementedError('neg_ratio not recognized')
    if withids:
        return pred_boxes, target, inds
    else:
        return pred_boxes, target

def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
    
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    anchor_step = int(len(anchors)/num_anchors)
    
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW)
    ty         = torch.zeros(nB, nA, nH, nW)
    tw         = torch.zeros(nB, nA, nH, nW)
    th         = torch.zeros(nB, nA, nH, nW)
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW)
    
    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    
    for b in range(nB):
        
        cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(cfg.max_boxes):
            ####  当没有目标时
            if target[b][t*5+1] == 0:
                break
            gx = target[b][t*5+1]*nW
            gy = target[b][t*5+2]*nH
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            
            cur_gt_boxes = torch.FloatTensor([gx,gy,gw,gh]).repeat(nAnchors,1).t()
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        conf_mask[b][torch.reshape(cur_ious, (nA, nH, nW)) > sil_thresh] = 0
    if seen < 12800:
       if anchor_step == 4:
           tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
           ty = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([2])).view(1,nA,1,1).repeat(nB,1,nH,nW)
       else:
           tx.fill_(0.5)
           ty.fill_(0.5)
       tw.zero_()
       th.zero_()
       coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    
    for b in range(nB):
        # pdb.set_trace()
        for t in range(50):
            if target[b][t*5+1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t*5+1] * nW
            gy = target[b][t*5+2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t*5+3]*nW
            gh = target[b][t*5+4]*nH
            gt_box = [0, 0, gw, gh]
            
            for n in range(nA):
                aw = anchors[anchor_step*n]
                ah = anchors[anchor_step*n+1]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if anchor_step == 4:
                    ax = anchors[anchor_step*n+2]
                    ay = anchors[anchor_step*n+3]
                    dist = pow(((gi+ax) - gx), 2) + pow(((gj+ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step==4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = [gx, gy, gw, gh]
            pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
            
            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            
            tx[b][best_n][gj][gi] = target[b][t*5+1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t*5+2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw/anchors[anchor_step*best_n])
            th[b][best_n][gj][gi] = math.log(gh/anchors[anchor_step*best_n+1])
            gt_box = [i.to(torch.float32) for i in gt_box]
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t*5]
            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):

        if target.dim() == 3:
            target = target.view(-1, target.size(-1))
        bef = target.size(0)
        output, target = neg_filter(output, target)
        # print("{}/{}".format(target.size(0), bef))

        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        # [nB, nA, nC, nW, nH] | (bs, 5, 1, 13, 13)
        cls  = output.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)

        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(4, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.data + grid_x
        pred_boxes[1] = y.data + grid_y
        pred_boxes[2] = torch.exp(w.data) * anchor_w
        pred_boxes[3] = torch.exp(h.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask = (cls_mask == 1)
        if cfg.metayolo:
            tcls.zero_()
        nProposals = int((conf > 0.25).float().sum().data[0])

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  

        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect, nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss

class RegionLossV2(nn.Module):
    """
    Yolo region loss + Softmax classification across meta-inputs
    """
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLossV2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        print('class_scale', self.class_scale)

    def forward(self, output, target):
        bs = target.size(0)
        cs = target.size(1)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls = output.view(output.size(0), nA, (5+nC), nH, nW)
        cls = cls.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda())).squeeze()
        cls = cls.view(bs, cs, nA*nC*nH*nW).transpose(1,2).contiguous().view(bs*nA*nC*nH*nW, cs)

        target = target.view(-1, target.size(-1))
        output, target, inds = neg_filter(output, target, withids=True)
        counts, _ = np.histogram(inds, bins=bs, range=(0, bs*cs))
        t0 = time.time()
        nB = output.data.size(0)
        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        t1 = time.time()
        pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        anchor_w = torch.Tensor(self.anchors)
        anchor_w = anchor_w.view(nA, int(self.anchor_step))
        anchor_w = anchor_w.index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1,
                                                                                           torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        
        pred_boxes[0] = torch.reshape(x.data, (1, nB * nA * nH * nW)) + grid_x
        pred_boxes[1] = torch.reshape(y.data, (1, nB * nA * nH * nW)) + grid_y
        pred_boxes[2] = torch.reshape(torch.exp(w.data), (1, nB * nA * nH * nW)) * anchor_w
        pred_boxes[3] = torch.reshape(torch.exp(h.data), (1, nB * nA * nH * nW)) * anchor_h
        
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_num = torch.sum(cls_mask)
        idx_start = 0
        cls_mask_list = []
        tcls_list = []
        for i in range(len(counts)):
            if counts[i] == 0:
                cur_mask = torch.zeros(nA, nH, nW)
                cur_tcls = torch.zeros(nA, nH, nW)
            else:
                cur_mask = torch.sum(cls_mask[idx_start:idx_start+counts[i]], dim=0)
                cur_tcls = torch.sum(tcls[idx_start:idx_start+counts[i]], dim=0)
            cls_mask_list.append(cur_mask)
            tcls_list.append(cur_tcls)
            idx_start += counts[i]
        cls_mask = torch.stack(cls_mask_list)
        tcls = torch.stack(tcls_list)

        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).float().sum().item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls        = cls[Variable(cls_mask.view(-1, 1).repeat(1,cs).cuda())].view(-1, cs)  
        tcls = tcls[cls_mask == 1].view(-1).long().cuda()
        ClassificationLoss = nn.CrossEntropyLoss(size_average=False)
       
        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * ClassificationLoss(cls, tcls)
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
        self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
        loss_conf.item(), loss_cls.item(), loss.item()))
      
        return loss

class RegionLossV3(nn.Module):
    """
    Yolo region loss + Softmax classification across meta-inputs + max margin loss
    """
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLossV3, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)/num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        print('class_scale', self.class_scale)

    def forward(self, output, target, dynamic_weights, target_cls_ids):
        dynamic_weights = dynamic_weights[0]
        #max margin:
        #VOC:
        vector_store = [[] for _ in range(20)]
        cnt = [0.0] * 20
        enews = [0.0] * 20
        
        #COCO:
        #vector_store = [[] for _ in range(80)]
        #cnt = [0.0] * 80
        #enews = [0.0] * 80
        count = 0
        same_class_distance = 0
        loss_class = nn.MSELoss(size_average=False)
        for c in target_cls_ids:
            c = int(c)
            enews[c] = enews[c] * cnt[c] / (cnt[c] + 1) + dynamic_weights[count] / (cnt[c] + 1)
            cnt[c] += 1
            count += 1
        count = 0
        
        different_class_distance = 0
        for i in range(len(enews)):
            current_class_max_distance = 0
            if(type(enews[i]) is type(float(0))):
                continue
            for j in range(len(enews)):
                if(i == j):
                    continue
                if(type(enews[j]) is type(float(0))):
                    continue
                if(current_class_max_distance == 0):
                    current_class_max_distance = loss_class(enews[i],enews[j])
                else:
                    current_class_max_distance = min(current_class_max_distance,loss_class(enews[i],enews[j]))
            different_class_distance += current_class_max_distance
            
        for c in target_cls_ids:
            c = int(c)
            same_class_distance += loss_class(dynamic_weights[count] ,enews[c])
            count += 1
            
        bs = target.size(0)
        cs = target.size(1)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls = output.view(output.size(0), nA, (5+nC), nH, nW)
        cls = cls.index_select(2, Variable(torch.linspace(5,5+nC-1,nC).long().cuda())).squeeze()
        cls = cls.view(bs, cs, nA*nC*nH*nW).transpose(1,2).contiguous().view(bs*nA*nC*nH*nW, cs)
        target = target.view(-1, target.size(-1))
        output, target, inds = neg_filter(output, target, withids=True)
        counts, _ = np.histogram(inds, bins=bs, range=(0, bs*cs))
        t0 = time.time()
        nB = output.data.size(0)
        output   = output.view(nB, nA, (5+nC), nH, nW)
        x    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y    = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w    = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        h    = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW))
        t1 = time.time()
        pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        anchor_w = torch.Tensor(self.anchors)
        anchor_w = anchor_w.view(nA, int(self.anchor_step))
        anchor_w = anchor_w.index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, int(self.anchor_step)).index_select(1,torch.LongTensor([1])).cuda() 
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
     
        pred_boxes[0] = torch.reshape(x.data, (1, nB * nA * nH * nW)) + grid_x
        pred_boxes[1] = torch.reshape(y.data, (1, nB * nA * nH * nW)) + grid_y
        pred_boxes[2] = torch.reshape(torch.exp(w.data), (1, nB * nA * nH * nW)) * anchor_w
        pred_boxes[3] = torch.reshape(torch.exp(h.data), (1, nB * nA * nH * nW)) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4))
        t2 = time.time()     
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf,tcls = build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
                                                               nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_num = torch.sum(cls_mask)
        idx_start = 0
        cls_mask_list = []
        tcls_list = []
        for i in range(len(counts)):
            if counts[i] == 0:
                cur_mask = torch.zeros(nA, nH, nW)
                cur_tcls = torch.zeros(nA, nH, nW)
            else:
                cur_mask = torch.sum(cls_mask[idx_start:idx_start+counts[i]], dim=0)
                cur_tcls = torch.sum(tcls[idx_start:idx_start+counts[i]], dim=0)
            cls_mask_list.append(cur_mask)
            tcls_list.append(cur_tcls)
            idx_start += counts[i]
        cls_mask = torch.stack(cls_mask_list)
        tcls = torch.stack(tcls_list)

        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).float().sum().item())

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda().sqrt())
        cls        = cls[Variable(cls_mask.view(-1, 1).repeat(1,cs).cuda())].view(-1, cs)  

        tcls = tcls[cls_mask == 1].view(-1).long().cuda()
        ClassificationLoss = nn.CrossEntropyLoss(size_average=False)
       
        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x*coord_mask, tx*coord_mask)/2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y*coord_mask, ty*coord_mask)/2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w*coord_mask, tw*coord_mask)/2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h*coord_mask, th*coord_mask)/2.0
        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
        loss_cls = self.class_scale * ClassificationLoss(cls, tcls)
        loss_class_contrast = same_class_distance / different_class_distance
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls + loss_class_contrast
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, class_contrast %f, total %f' % (
        self.seen, nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
        loss_conf.item(), loss_cls.item(), loss_class_contrast.item(), loss.item()))
        return loss



def select_classes(pred, tgt, ids):
    # convert tgt to numpy
    tgt = tgt.cpu().data.numpy()
    new_tgt = [(tgt == d) * i  for i, d in enumerate(ids)]
    new_tgt = np.max(np.stack(new_tgt), axis=0)
    idxes = np.argwhere(new_tgt > 0).squeeze()
    new_pred = pred[idxes]
    new_pred = new_pred[:, ids]
    new_tgt = new_tgt[idxes]
    return new_pred, new_tgt
