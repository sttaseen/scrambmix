# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta
from scipy import stats

from .builder import BLENDINGS

__all__ = ['BaseMiniBatchBlending', 'MixupBlending', 'CutmixBlending']


class BaseMiniBatchBlending(metaclass=ABCMeta):
    """Base class for Image Aliasing."""

    def __init__(self, num_classes):
        self.num_classes = num_classes

    @abstractmethod
    def do_blending(self, imgs, label, **kwargs):
        pass

    def __call__(self, imgs, label, **kwargs):
        """Blending data in a mini-batch.

        Images are float tensors with the shape of (B, N, C, H, W) for 2D
        recognizers or (B, N, C, T, H, W) for 3D recognizers.

        Besides, labels are converted from hard labels to soft labels.
        Hard labels are integer tensors with the shape of (B, 1) and all of the
        elements are in the range [0, num_classes - 1].
        Soft labels (probablity distribution over classes) are float tensors
        with the shape of (B, 1, num_classes) and all of the elements are in
        the range [0, 1].

        Args:
            imgs (torch.Tensor): Model input images, float tensor with the
                shape of (B, N, C, H, W) or (B, N, C, T, H, W).
            label (torch.Tensor): Hard labels, integer tensor with the shape
                of (B, 1) and all elements are in range [0, num_classes).
            kwargs (dict, optional): Other keyword argument to be used to
                blending imgs and labels in a mini-batch.

        Returns:
            mixed_imgs (torch.Tensor): Blending images, float tensor with the
                same shape of the input imgs.
            mixed_label (torch.Tensor): Blended soft labels, float tensor with
                the shape of (B, 1, num_classes) and all elements are in range
                [0, 1].
        """
        one_hot_label = F.one_hot(label, num_classes=self.num_classes)

        mixed_imgs, mixed_label = self.do_blending(imgs, one_hot_label,
                                                   **kwargs)

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class MixupBlending(BaseMiniBatchBlending):
    """Implementing Mixup in a mini-batch.

    This module is proposed in `mixup: Beyond Empirical Risk Minimization
    <https://arxiv.org/abs/1710.09412>`_.
    Code Reference https://github.com/open-mmlab/mmclassification/blob/master/mmcls/models/utils/mixup.py # noqa

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = self.beta.sample()
        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)

        mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
        mixed_label = lam * label + (1 - lam) * label[rand_index, :]

        return mixed_imgs, mixed_label

@BLENDINGS.register_module()
class Scrambmix(BaseMiniBatchBlending):
    """Implementing Scrambmix in a mini-batch.

    Args:
        num_classes (int): The number of classes.
        num_frames (int): The number of frames.
        alpha (float): Parameters for Beta Binomial distribution.
    """

    def __init__(self, num_classes, num_frames, alpha=5):
        super().__init__(num_classes=num_classes)
        self.num_frames = num_frames
        self.beta_binom = stats.betabinom(num_frames-1, alpha, alpha, loc=0)
        
    def rand_bbox(self, img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # This is uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with scrambmix."""

        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        batch_size = imgs.size(0)
        
        epsilon = self.beta_binom.rvs() + 1
        interval = round(self.num_frames/epsilon)
        rand_index = torch.randperm(batch_size)

        mask = torch.arange(self.num_frames) % interval == 0
        lam = mask.sum()/self.num_frames
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)

        A = imgs
        B = A.clone()[rand_index, ...]
        
        # Apply the masks
        A[..., ~mask, bby1:bby2, bbx1:bbx2], B[..., mask, bby1:bby2, bbx1:bbx2] = \
            B[..., ~mask, bby1:bby2, bbx1:bbx2], A[..., mask, bby1:bby2, bbx1:bbx2]
        
        A[..., mask, :, :] = 0
        B[..., ~mask, :, :] = 0

        # Calculate the areas
        bbox_area = (bby2-bby1) * (bbx2-bbx1)
        w = imgs.size(-1)
        h = imgs.size(-2)
        total_area = w*h

        # Proportion of frame occupied by the bounding box
        theta = bbox_area/total_area
        
        # MixUp the frames and encodings
        mixed_imgs = A + B
        mixed_label = lam * theta * label + (1 - lam) * theta * label[rand_index, :] + \
            lam * (1-theta) * label[rand_index, :] + (1-lam) * (1-theta) * label

        return mixed_imgs, mixed_label
    

@BLENDINGS.register_module()
class ReverseScrambmix(BaseMiniBatchBlending):
    """Implementing Scrambmix in a mini-batch.

    Args:
        num_classes (int): The number of classes.
        num_frames (int): The number of frames.
        alpha (float): Parameters for Beta Binomial distribution.
    """

    def __init__(self, num_classes, num_frames, alpha=5):
        super().__init__(num_classes=num_classes)
        self.num_frames = num_frames
        self.beta_binom = stats.betabinom(num_frames-1, alpha, alpha, loc=0)
        
    def rand_bbox(self, img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # This is uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with scrambmix."""

        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        batch_size = imgs.size(0)
        
        epsilon = self.beta_binom.rvs() + 1
        interval = round(self.num_frames/epsilon)
        rand_index = torch.randperm(batch_size)

        mask = torch.arange(self.num_frames) % interval == 0
        lam = mask.sum()/self.num_frames
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)

        A = imgs
        B = A.clone()[rand_index, ...]
        
        # Apply the masks
        A[..., ~mask, bby1:bby2, bbx1:bbx2], B[..., mask, bby1:bby2, bbx1:bbx2] = B[..., ~mask, bby1:bby2, bbx1:bbx2], A[..., mask, bby1:bby2, bbx1:bbx2]
        
        A[..., mask, :, :] = 0
        B[..., ~mask, :, :] = 0
        
        # MixUp the frames and encodings
        mixed_imgs = A + B
        mixed_label = lam * label + (1 - lam) * label[rand_index, :] # Mistake fixed
        # mixed_label = (1 - lam) * label +  lam * label[rand_index, :]

        return mixed_imgs, mixed_label
    
@BLENDINGS.register_module()
class Scrambmix_v2(BaseMiniBatchBlending):
    """Implementing Scrambmix in a mini-batch.

    Args:
        num_classes (int): The number of classes.
        num_frames (int): The number of frames.
        alpha (float): Parameters for Beta Binomial distribution.
    """

    def __init__(self, num_classes, num_frames, alpha=5):
        super().__init__(num_classes=num_classes)
        self.num_frames = num_frames
        self.beta_binom = stats.betabinom(num_frames-1, alpha, alpha, loc=0)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with scrambmix."""

        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        batch_size = imgs.size(0)
        
        epsilon = self.beta_binom.rvs() + 1
        interval = round(self.num_frames/epsilon)
        lam = 1/interval
        rand_index = torch.randperm(batch_size)

        mask = torch.arange(self.num_frames) % interval == 0

        A = imgs
        B = A.clone()[rand_index, ...]

        A[..., mask, :, :] = 0
        B[..., ~mask, :, :] = 0

        mixed_imgs = A + B
        mixed_label = (1 - lam) * label +  lam * label[rand_index, :]

        return mixed_imgs, mixed_label
    
    
@BLENDINGS.register_module()
class Scrambmix_v1(BaseMiniBatchBlending):
    """Implementing Scrambmix in a mini-batch.

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with mixup."""
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'

        lam = 0.5
        batch_size = imgs.size(0)
        frames = imgs.size(3)
        rand_index = torch.randperm(batch_size)

        batch_list = []
        for i in range(batch_size):
            A = imgs[i]
            B = imgs[rand_index[i]]
            
            shuffle = []
            for i in range(0, frames, 2):
                shuffle.append(A[:, :, i, :])
                shuffle.append(B[:, :, i+1, :])
            
            batch_list.append(torch.stack(shuffle))
            
        mixed_imgs = torch.stack(batch_list).permute(0, 2, 3, 1, 4, 5)

        mixed_label = (1 - lam)  * label + lam * label[rand_index, :]

        return mixed_imgs, mixed_label


@BLENDINGS.register_module()
class CutmixBlending(BaseMiniBatchBlending):
    """Implementing Cutmix in a mini-batch.

    This module is proposed in `CutMix: Regularization Strategy to Train Strong
    Classifiers with Localizable Features <https://arxiv.org/abs/1905.04899>`_.
    Code Reference https://github.com/clovaai/CutMix-PyTorch

    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, num_classes, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.beta = Beta(alpha, alpha)

    @staticmethod
    def rand_bbox(img_size, lam):
        """Generate a random boudning box."""
        w = img_size[-1]
        h = img_size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.tensor(int(w * cut_rat))
        cut_h = torch.tensor(int(h * cut_rat))

        # uniform
        cx = torch.randint(w, (1, ))[0]
        cy = torch.randint(h, (1, ))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w)
        bby1 = torch.clamp(cy - cut_h // 2, 0, h)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w)
        bby2 = torch.clamp(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def do_blending(self, imgs, label, **kwargs):
        """Blending images with cutmix."""
        assert len(kwargs) == 0, f'unexpected kwargs for cutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
        imgs[:, ..., bby1:bby2, bbx1:bbx2] = imgs[rand_index, ..., bby1:bby2,
                                                  bbx1:bbx2]
        lam = 1 - (1.0 * (bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.size()[-1] * imgs.size()[-2]))

        label = lam * label + (1 - lam) * label[rand_index, :]

        return imgs, label
    
@BLENDINGS.register_module()
class FloatFrameCutmix(BaseMiniBatchBlending):
    """Implementing FloatFrameCutMix"""

    def __init__(self, num_classes, num_frames, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.num_frames = num_frames
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, labels, **kwargs):
        """
        Blending images with FloatFrameCutMixup.

        """
        assert len(kwargs) == 0, f'unexpected kwargs for floatframecutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        sequence_length = round(self.num_frames * lam.item())

        adj = torch.randint(0, 2, (1,)) * 2 - 1  # gives -1 or 1 randomly
        adj = adj * torch.rand(1) * min(lam, 1.0 - lam)
        fade = torch.linspace(lam.item() - adj.item(), lam.item() + adj.item(), steps=sequence_length)
        ones = torch.ones(self.num_frames - sequence_length, dtype=torch.float32)
        weights = torch.cat((fade, ones)).view(1, self.num_frames, 1, 1).to('cuda')

        A = imgs
        B = A.clone()[rand_index, ...]
        A = A * weights
        B = B * (1 - weights)

        mixed_imgs = A + B

        label_ratio = (torch.sum(weights).item()) / self.num_frames
        label = label_ratio * labels + (1 - label_ratio) * labels[rand_index, :]

        return mixed_imgs, label


@BLENDINGS.register_module()
class FrameCutmix(BaseMiniBatchBlending):
    """Implementing FrameCutMix"""

    def __init__(self, num_classes, num_frames, alpha=.2):
        super().__init__(num_classes=num_classes)
        self.num_frames = num_frames
        self.beta = Beta(alpha, alpha)

    def do_blending(self, imgs, labels, **kwargs):
        """
        Blending images with FloatFrameCutMixup.

        """
        assert len(kwargs) == 0, f'unexpected kwargs for floatframecutmix {kwargs}'

        batch_size = imgs.size(0)
        rand_index = torch.randperm(batch_size)
        lam = self.beta.sample()

        sequence_length = round(self.num_frames * lam.item())

        mixup_ratio = torch.full((sequence_length,), lam)
        ones = torch.ones(self.num_frames - sequence_length, dtype=torch.float32)
        weights = torch.cat((mixup_ratio, ones)).view(1, self.num_frames, 1, 1)

        A = imgs
        B = A.clone()[rand_index, ...]
        A = A * weights
        B = B * (1 - weights)

        mixed_imgs = A + B

        label_ratio = (torch.sum(weights).item()) / self.num_frames
        label = label_ratio * labels + (1 - label_ratio) * labels[rand_index, :]

        return mixed_imgs, label
