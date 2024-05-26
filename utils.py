'''
Description     : utils implementation of Surgical VQLA model
Paper           : CAT-ViL: Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
Acknowledgement : Code adopted from the official implementation of 
                  Surgical VQA (https://github.com/huggingface/transformers.git) and
                  DETR (https://github.com/facebookresearch/detr).
'''

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from torchvision.ops.boxes import box_area
import torch.nn.functional as F
from torch import nn
import math
import statistics
class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, metrics, is_best, final_args):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param model: model
    :param optimizer: optimizer to update model's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'metrics': metrics,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)

def save_clf_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, Acc, final_args):
    """
    Saves model checkpoint.
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'Acc': Acc,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def calc_acc(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc


def calc_classwise_acc(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    classwise_acc = matrix.diagonal()/matrix.sum(axis=1)
    return classwise_acc


def calc_map(y_true, y_scores):
    mAP = average_precision_score(y_true, y_scores,average=None)
    return mAP

def calc_precision_recall_fscore(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division = 1)
    return(precision, recall, fscore)

        
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, l1g=False, eps=1e-7):    
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    src_boxes = box1
    target_boxes = box2
    (num_boxes, n) = src_boxes.shape
    loss_l1= F.l1_loss(src_boxes, target_boxes, reduction='none')
    losses_l1 = loss_l1.sum() / num_boxes

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or l1g:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return 1 - (iou - (rho2 / c2 + v * alpha))  # L1+CIoU
            return 1 - (iou - rho2 / c2) # L1+DIoU
        c_area = cw * ch + eps  # convex area
        if l1g:
            return 1 - (iou - (c_area - union) / c_area) + losses_l1  # L1+GIoU
        return 1 - (iou - (c_area - union) / c_area)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    else: return iou #L1



def DualFocalLoss(input, target):
    gamma = 1.0
    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1,1)

    logp_k = F.log_softmax(input, dim=1)
    softmax_logits = logp_k.exp()
    logp_k = logp_k.gather(1, target)
    logp_k = logp_k.view(-1)
    p_k = logp_k.exp()  # p_k: probility at target label
    p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
    p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

    loss = -1 * (1 - p_k + p_j) ** gamma * logp_k

    return loss.mean()


def giou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()

def mIoU_xyxy(box_a, box_b):
    # inter = intersection(box_a, box_b)
    assert box_a.shape == box_b.shape
    (m, n) = box_a.shape
    iou_sum = 0
    iou_list = []
    for i in range(m):
        x1 = max(box_a[i, 0], box_b[i, 0])
        y1 = max(box_a[i, 1], box_b[i, 1])
        x2 = min(box_a[i, 2], box_b[i, 2])
        y2 = min(box_a[i, 3], box_b[i, 3])
        if x1 >= x2 or y1 >= y2:
            inter = 0.0
        inter = float((x2 - x1 + 1) * (y2 - y1 + 1))
        box_a_area = (box_a[i, 2] - box_a[i, 0] + 1) * (box_a[i, 3] - box_a[i, 1] + 1)
        box_b_area = (box_b[i, 2] - box_b[i, 0] + 1) * (box_b[i, 3] - box_b[i, 1] + 1)
        union = box_a_area + box_b_area - inter
        iou = inter / float(max(union, 1))
        iou_sum = iou_sum + iou
        iou_list.append(iou)

    std_iou = statistics.stdev(iou_list)
    m_iou = iou_sum / m
    return m_iou, std_iou

def mIoU_single(box_a, box_b):
    # inter = intersection(box_a, box_b)
    assert box_a.shape == box_b.shape
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x1 >= x2 or y1 >= y2:
        inter = 0.0
    inter = float((x2 - x1 + 1) * (y2 - y1 + 1))
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = box_a_area + box_b_area - inter
    iou = inter / float(max(union, 1))
    return iou

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def loss_giou_l1(outputs, targets):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """

    src_boxes = outputs
    target_boxes = targets
    (num_boxes, n) = src_boxes.shape
    loss_l1= F.l1_loss(src_boxes, target_boxes, reduction='none')
    losses_l1 = loss_l1.sum() / num_boxes
    loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
    losses_giou = loss_giou.sum() / num_boxes
    losses = losses_l1 + losses_giou
    
    return losses

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class AutomaticWeightedLoss(nn.Module):                   

    def __init__(self, num=3):
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
        # loss = 0.5 / (self.params[0] ** 2) * loss_class + torch.log(1 + self.params[0] ** 2) + \
        #     0.5 / (self.params[1] ** 2) * loss_bbox + torch.log(1 + self.params[1] ** 2) + \
        #     0.5 / (self.params[2] ** 2) * (loss_bbox_aux + loss_class_aux * 0.01) + torch.log(1 + self.params[2] ** 2)

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自文本，i为初始文本的embedding,j为添加扰动后的embedding
        # print("Size of mlp_outputs: ", emb_i.size())
        # print("Size of mlp_perturbed_outputs: ", emb_j.size())
        # print("Size of batch:", self.batch_size)
        z_i = nn.functional.normalize(emb_i, dim=1)
        z_j = nn.functional.normalize(emb_j, dim=1)

        # representations = torch.cat([z_i, z_j], dim=0)
        # similarity_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
        #                                                     dim=2)

        # sim_ij = torch.diag(similarity_matrix, self.batch_size)
        # sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        # positives = torch.cat([sim_ij, sim_ji], dim=0)

        # nominator = torch.exp(positives / self.temperature)
        # denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # loss = torch.sum(loss_partial) / (2 * self.batch_size)
        # return loss


        representations = torch.cat([z_i, z_j], dim=0)
        sim_matrix = nn.functional.cosine_similarity(representations.unsqueeze(1)/self.temperature, representations.unsqueeze(0),
                                                            dim=2)
        # 计算对角线元素的softmax
        sim_diag = torch.diag(sim_matrix)
        sim_diag = sim_diag.view(-1, 1)
        sim_matrix = torch.exp(sim_matrix - sim_diag)
        # 计算对抗性损失
        loss_adv = -torch.log(torch.diag(sim_matrix) / torch.sum(sim_matrix, dim=1))
        loss_adv = torch.mean(loss_adv)

        return loss_adv


