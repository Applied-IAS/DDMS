import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.autograd import Variable
from custom_functional import compute_grad_mag


def clip_by_tensor(probs, probs_min, probs_max):
    """
    clip_by_tensor
    :param probs: tensor
    :param probs_min: min
    :param probs_max: max
    :return: cliped tensor
    """
    probs = probs.float()
    probs_min = probs_min.float()
    probs_max = probs_max.float()

    result = (probs >= probs_min).float() * probs + (probs < probs_min).float() * probs_min
    result = (result <= probs_max).float() * result + (result > probs_max).float() * probs_max
    return result

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # 这是根据lable取值作为下标，把label值替换为alpha[label_val_ij]
        alpha = self.alpha[ids.data.view(-1)]

        # 这里就是取的相应类中softmax后的概率，因为class_mask只有在label对应类中为1，其他类为0
        probs = (P*class_mask).sum(1).view(-1,1)
        hh, ww = probs.size()
        probs = clip_by_tensor(probs, probs.data.new(hh, ww).fill_(0.005), probs.data.new(hh, ww).fill_(1.0))
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
        # 对于二分类正样本来说，其对应alpha[1]，正样本位置的probs要概率尽可能接近与1，
        # 所以增大alpha[1],则是正样本被误判为负样本的loss会变大，反之亦然，
        # 而同时alpha[0]变小，则其被误判为正样本的loss会变小
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def perturbate_input_(input, n_elements=200):
    N, C, H, W = input.shape
    assert N == 1
    c_ = np.random.random_integers(0, C - 1, n_elements)
    h_ = np.random.random_integers(0, H - 1, n_elements)
    w_ = np.random.random_integers(0, W - 1, n_elements)
    for c_idx in c_:
        for h_idx in h_:
            for w_idx in w_:
                input[0, c_idx, h_idx, w_idx] = 1
    return input


def _sample_gumbel(shape, eps=1e-10):
    """
    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).cuda()
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    assert logits.dim() == 3
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, 1)


def _one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes).cuda()
    return y[labels].permute(0, 3, 1, 2)


class DualTaskLoss(nn.Module):
    def __init__(self, num_class, cuda=False):
        super(DualTaskLoss, self).__init__()
        self._cuda = cuda
        self.num_class = num_class
        return

    def forward(self, input_logits, gts, ignore_pixel=255):
        """
        :param input_logits: NxCxHxW
        :param gt_semantic_masks: NxCxHxW
        :return: final loss
        """
        N, C, H, W = input_logits.shape
        # print(input_logits.shape, gts.shape, gts.min(), gts.max())
        th = 1e-8  # 1e-10
        eps = 1e-10
        ignore_mask = (gts == self.num_class).detach()
        input_logits = torch.where(ignore_mask.view(N, 1, H, W).expand(N, self.num_class, H, W),
                                   torch.zeros(N, C, H, W).cuda(),
                                   input_logits)
        gt_semantic_masks = gts.detach()
        gt_semantic_masks = torch.where(ignore_mask, torch.zeros(N, H, W).long().cuda(), gt_semantic_masks)
        gt_semantic_masks = _one_hot_embedding(gt_semantic_masks, self.num_class).detach()

        g = _gumbel_softmax_sample(input_logits.view(N, C, -1), tau=0.5)
        g = g.reshape((N, C, H, W))
        g = compute_grad_mag(g, cuda=self._cuda)

        g_hat = compute_grad_mag(gt_semantic_masks, cuda=self._cuda)

        g = g.view(N, -1)
        g_hat = g_hat.view(N, -1)
        loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return total_loss


class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255,
                 norm=False, upper_bound=1.0, mode='train',
                 edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        if mode == 'train':
            self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()
        elif mode == 'val':
            self.seg_loss = CrossEntropyLoss2d(size_average=True,
                                               ignore_index=ignore_index).cuda()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

        self.dual_task = DualTaskLoss(classes)
        self.bce_loss = nn.BCELoss(reduce=False)

    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, size_average=True)
        # loss = F.binary_cross_entropy_with_logits(log_p, target_t, None, size_average=True)
        return loss

    def edge_attention(self, input, target, edge):
        n, c, h, w = input.size()
        filler = torch.ones_like(target) * 255
        return self.seg_loss(input,
                             torch.where(edge.max(1)[0] > 0.8, target, filler))

    def forward(self, inputs, targets):
        segin, edgein = inputs
        segmask, edgemask = targets

        # losses = {}

        # losses['seg_loss'] = self.seg_weight * self.seg_loss(segin, segmask)
        # losses['edge_loss'] = self.edge_weight * 20 * self.bce2d(edgein, edgemask)
        # losses['att_loss'] = self.att_weight * self.edge_attention(segin, segmask, edgein)
        # losses['dual_loss'] = self.dual_weight * self.dual_task(segin, segmask)

        # return losses
        # return self.bce2d(edgein, edgemask), self.edge_attention(segin, segmask, edgein),self.dual_task(segin, segmask)
        # self.seg_loss(segin, segmask),
        # return self.bce_loss(edgein, edgemask), self.edge_attention(segin, segmask, edgein), self.dual_task(segin, segmask)
        return self.seg_loss(segin, segmask), self.bce_loss(edgein, edgemask), self.edge_attention(segin, segmask, edgein), self.dual_task(segin,
                                                                                                            segmask)

# Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()

            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                  targets[i].unsqueeze(0))
        return loss


# Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        # print(logits.size())
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


if __name__ == "__main__":
    # input = torch.rand([5,2]).cuda()
    # print(input)
    # target = torch.Tensor([1, 1, 0, 0, 1]).cuda()
    # # print(input.data)
    # # input = input.view(-1,1)
    # # print(input)
    # class_mask = input.data.new(5,2).fill_(0)
    # class_mask = Variable(class_mask)
    # # print(class_mask)
    # ids = target.view(-1, 1)
    # class_mask.scatter_(1, ids.data.long(), 1.)
    # print(class_mask)
    # alpha = torch.Tensor([[0.5], [1]]).cuda()
    # print(ids.data.view(-1))
    # alpha = alpha[ids.data.view(-1).long()]
    # print(alpha)
    # print(input*class_mask)
    # print((input*class_mask).sum(1))

    # joint_loss = JointEdgeSegLoss(classes=2)
    # seg = torch.rand((2, 2, 768, 768)).cuda()
    # edge = torch.rand((2, 1, 768, 768)).cuda()
    # label = torch.rand((2, 768, 768)).cuda()
    # label_edge = torch.rand((2, 1, 768, 768)).cuda()
    # l_edge, l_att, l_dual =joint_loss((seg.float(), edge.float()), (label.long(), label_edge.float()))
    # print(l_edge, l_att, l_dual)

    dice_loss = SoftDiceLoss()
    seg = torch.zeros((1, 2, 4, 4)).cuda()
    # a = seg.cpu().numpy()
    # print(seg)
    # a = torch.argmax(seg, dim=1)
    # print(a)

    back = seg[0, 0, :, :].cpu().numpy()
    forward = seg[0, 1, :, :].cpu().numpy()
    res = np.zeros([4, 4], dtype=np.float)
    # print(back)
    # print(forward)
    index = np.where(back >= forward)
    # # print(index)
    res[index] = 1-back[index]
    # print(res)
    index = np.where(back < forward)
    res[index] = forward[index]
    # print(res)


    label = torch.zeros((1, 4, 4)).cuda()
    score = dice_loss(seg, label)
    print(score)