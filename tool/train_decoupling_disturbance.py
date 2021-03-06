from __future__ import print_function
import sys

sys.path.append("Your Project dir")

if len(sys.argv) != 6:
    print('Usage:')
    print('python train.py datacfg darknetcfg weightfile')
    exit()

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from core import dataset
from core.utils import *
from core.cfg import parse_cfg, cfg
from tool.darknet.darknet_decoupling import Darknet

# Training settings
datacfg       = sys.argv[1]
darknetcfg    = parse_cfg(sys.argv[2])
learnetcfg    = parse_cfg(sys.argv[3])
weightfile    = sys.argv[4]

data_options  = read_data_cfg(datacfg)
net_options   = darknetcfg[0]
meta_options  = learnetcfg[0]

# Configure options
cfg.config_data(data_options)
cfg.config_meta(meta_options)
cfg.config_net(net_options)

# Parameters 
metadict      = data_options['meta']
trainlist     = data_options['train']

testlist      = data_options['valid']
backupdir     = data_options['backup']
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
use_cuda      = True
#seed          = int(time.time())
seed = 1
eps           = 1e-5
dot_interval  = 70  # batches
# save_interval = 10  # epoches

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

## --------------------------------------------------------------------------
## MAIN
#backupdir = cfg.backup
backupdir    = sys.argv[5]

print('logging to ' + backupdir)
if not os.path.exists(backupdir):
    os.mkdir(backupdir)

torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet(darknetcfg, learnetcfg)
region_loss = model.loss

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(weightfile))

model.load_weights(weightfile)
model.print_network()


###################################################
### Meta-model parameters
region_loss.seen  = model.seen
processed_batches = 0 if cfg.tuning else model.seen/batch_size
trainlist         = dataset.build_dataset(data_options)
nsamples          = len(trainlist)
init_width        = model.width
init_height       = model.height
init_epoch        = 0 if cfg.tuning else model.seen/nsamples
max_epochs        = max_batches*batch_size/nsamples+1
max_epochs        = int(math.ceil(cfg.max_epoch*1./cfg.repeat)) if cfg.tuning else max_epochs 
print(cfg.repeat, nsamples, max_batches, batch_size)
print(num_workers)

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                        shuffle=False,
                        transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

test_metaset = dataset.MetaDataset(metafiles=metadict, train=True)
test_metaloader = torch.utils.data.DataLoader(
    test_metaset,
    batch_size=test_metaset.batch_size,
    shuffle=False,
    num_workers=num_workers//2,
    pin_memory=True
)

# Adjust learning rate
factor = len(test_metaset.classes)
if cfg.neg_ratio == 'full':
    factor = 15.
elif cfg.neg_ratio == 1:
    factor = 3.0
elif cfg.neg_ratio == 0:
    factor = 1.5
elif cfg.neg_ratio == 5:
    factor = 8.0

print('factor:', factor)
learning_rate /= factor

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model.cuda(), device_ids=[0,1])
    else:
        model = model.cuda()

optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate/batch_size,
                      momentum=momentum,
                      dampening=0,
                      weight_decay=decay*batch_size*factor)

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def Inverted_gradient(feature, ratio, mask):
    """
        feature: rpn features
        ratio: how many to be inverted
    """
    mask_all = []
    mask_count = []
    for rpn_feat in feature:
        num_batch = rpn_feat.shape[0]
        num_channel = rpn_feat.shape[1]
        num_height = rpn_feat.shape[2]
        num_width = rpn_feat.shape[3]
        
        rpn_feat = torch.abs(rpn_feat)
        rpn_feat = rpn_feat * mask
        for batch in range(num_batch):
            mask_count.append(torch.sum(mask[batch]))
        feat_channel_mean = torch.mean(rpn_feat.view(num_batch, num_channel, -1), dim=2)
        feat_channel_mean = feat_channel_mean.view(num_batch, num_channel, 1, 1)
        cam = torch.sum(rpn_feat * feat_channel_mean, 1) #[B 1 H W]
        cam_tmp = cam.view(num_batch, num_height*num_width)
        cam_tmp_sort, cam_tmp_indice = cam_tmp.sort(dim = 1,descending = True)
        for batch in range(num_batch):
            th_idx = int(ratio * mask_count[batch])
            
            thresold = cam_tmp_sort[batch][th_idx - 1]
            thresold_map = thresold * torch.ones(1, num_height, num_width).cuda()
            mask_all_cuda = torch.where(cam[batch] > thresold_map, torch.zeros(cam[batch].shape).cuda(),
                                torch.ones(cam[batch].shape).cuda())
            mask_all_cuda = mask_all_cuda.view(1,1,num_height,num_width)
            if batch == 0:
                result = mask_all_cuda.detach()
            else:
                result = torch.cat((result,mask_all_cuda.detach()),0)
        mask_all.append(result)

    return mask_all

def train(epoch, repeat_time, mask_ratio):
    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                            shuffle=False,
                            transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]),
                            train=True,
                            seen=cur_model.seen,
                            batch_size=batch_size,
                            num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    metaset = dataset.MetaDataset(metafiles=metadict, train=True, with_ids=True)
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=metaset.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    metaloader = iter(metaloader)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d/%d, processed %d samples, lr %f' % (epoch, max_epochs, epoch * len(train_loader.dataset), lr))

    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    
    novel_id = cfg['novel_ids']
    for batch_idx, (data, target) in enumerate(train_loader):
        metax, mask, target_cls_ids = metaloader.next()
        
        novel_cls_flag = torch.zeros(len(target_cls_ids))
        for index,j in enumerate(target_cls_ids):
            #print(index)
            if j in novel_id:
                #print("flag",index)
                novel_cls_flag[int(index)] = 0
            else:
                novel_cls_flag[int(index)] = 1
                
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        if use_cuda:
            data = data.cuda()
            metax = metax.cuda()
            mask = mask.cuda()
            target_cls_ids = target_cls_ids.cuda()
            #target= target.cuda()
        t3 = time.time()
        data, target = Variable(data), Variable(target)
        metax, mask, target_cls_ids = Variable(metax,requires_grad=True), Variable(mask), Variable(target_cls_ids)
        t4 = time.time()
        for i in range(repeat_time):
            optimizer.zero_grad()
            t5 = time.time()
            metax_disturbance = metax
            if i == 0:
                mask_disturbance = mask
            else:
                for index, flag_each in enumerate(novel_cls_flag):
                    if flag_each == 0:
                        mask_disturbance[index] = mask[index]
                    elif flag_each == 1:
                        mask_disturbance[index] = mask[index] * metax_mask[0][index]
                    else:
                        print("error")
            output, dynamic_weights = model(data, metax_disturbance, mask_disturbance)
            t6 = time.time()
            region_loss.seen = region_loss.seen + data.data.size(0)
            if i == 0:
                loss = region_loss(output, target, dynamic_weights, target_cls_ids)
                dynamic_weights_store = dynamic_weights
                target_cls_ids_store = target_cls_ids
                dynamic_weight_buffer = dynamic_weights
            else:
                with torch.no_grad():
                    for index, flag_each in enumerate(novel_cls_flag):
                        if flag_each == 1:
                            dynamic_weights_store = [torch.cat((dynamic_weights_store[0],dynamic_weights[0][index].unsqueeze(0)),dim = 0)]
                        else:
                            continue
                    for num in range(int(torch.sum(novel_cls_flag) // len(novel_id))):
                        Tensor_novel_id = torch.Tensor(novel_id).long().cuda()
                        target_cls_ids_store = torch.cat((target_cls_ids_store,Tensor_novel_id),0)
                loss = region_loss(output, target, dynamic_weights_store, target_cls_ids_store)
                
            t7 = time.time()
            loss.backward()
            metax_mask = Inverted_gradient([metax.grad], mask_ratio, mask)
            
            t8 = time.time()
            optimizer.step()
            t9 = time.time()
        t1 = time.time()
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))

    if (epoch+1) % cfg.save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))


def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    num_classes = cur_model.num_classes
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    _test_metaloader = iter(test_metaloader)
    for batch_idx, (data, target) in enumerate(test_loader):
        metax, mask = _test_metaloader.next()
        if use_cuda:
            data = data.cuda()
            metax = metax.cuda()
            mask = mask.cuda()
        data = Variable(data, volatile=True)
        metax = Variable(metax, volatile=True)
        mask = Variable(mask, volatile=True)
        output = model(data, metax, mask).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
     
            total = total + num_gts
    
            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))



evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    for epoch in range(int(init_epoch), int(max_epochs)):
        if cfg.tuning:
            if "1shot" in metadict:
                shot_num = 1
            elif "2shot" in metadict:
                shot_num = 2
            elif "3shot" in metadict:
                shot_num = 3
            elif "5shot" in metadict:
                shot_num = 5
            elif "10shot" in metadict:
                shot_num = 10
            else:
                print("error!")
            repeat_time = 13 - shot_num
            mask_ratio = 0.15
        else:
            repeat_time = 1
            mask_ratio = 0
        
        train(epoch,repeat_time,mask_ratio)