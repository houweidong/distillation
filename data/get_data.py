import os
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, \
    RandomHorizontalFlip, Normalize, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
import torch.utils.data as data
from data.read_newdata import NewdataAttr
from data.transforms import ToMaskedTargetTensor, ToMaskedTargetTensorPaper, \
    get_inference_transform_person, get_inference_transform_person_lr, square_no_elastic


def _get_newdata(opt, mean, std, attrs):
    root = os.path.join(opt.root_path, 'new')
    # cropping_transform = get_inference_transform_person_lr
    if opt.logits_vac:
        cropping_transform = Compose([get_inference_transform_person_lr, square_no_elastic])
        train_img_transform = Compose(
            [
             # [RandomHorizontalFlip(), RandomRotation(10, expand=True),
             # Resize((opt.person_size, opt.person_size)),
             ToTensor(), Normalize(mean, std)])
        # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
        val_img_transform = Compose(
            [# Resize((opt.person_size, opt.person_size)),
             ToTensor(), Normalize(mean, std)])
    else:
        cropping_transform = get_inference_transform_person_lr
        train_img_transform = Compose(
             [square_no_elastic,
              RandomHorizontalFlip(), RandomRotation(10, expand=True),
              Resize((opt.person_size, opt.person_size)),
             ToTensor(), Normalize(mean, std)])
        # [CenterCrop(178), Resize((256, 256)), RandomCrop(224), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])
        val_img_transform = Compose(
            [square_no_elastic, Resize((opt.person_size, opt.person_size)),
             ToTensor(), Normalize(mean, std)])
    target_transform = ToMaskedTargetTensor(attrs, opt.label_smooth, opt.at, opt.at_loss)

    train_data = NewdataAttr(attrs, root, 'train', opt.mode, opt.state, cropping_transform, img_transform=train_img_transform,
                             target_transform=target_transform, logits_vac=opt.logits_vac)
    val_data = NewdataAttr(attrs, root, 'test', opt.mode, opt.state, cropping_transform,
                           img_transform=val_img_transform, target_transform=target_transform, logits_vac=opt.logits_vac)

    return train_data, val_data


_dataset_getters = {'New': _get_newdata}


def get_data(opt, available_attrs, mean, std):
    name = opt.dataset

    assert name in _dataset_getters
    train, val = _dataset_getters[name](opt, mean, std, available_attrs)

    train_loader = DataLoader(train, batch_size=opt.batch_size, num_workers=opt.n_threads, shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=opt.batch_size, num_workers=opt.n_threads,
                            pin_memory=True)

    return train_loader, val_loader
