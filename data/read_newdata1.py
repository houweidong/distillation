from torch.utils.data import Dataset
import os
from data.attributes import WiderAttributes as WdAt, Attribute, AttributeType as AtTp
from torchvision.datasets.folder import pil_loader
import json
from data.image_loader import opencv_loader
from data.read_newdata import transform_try, get_image_list


class NewdataAttr1(Dataset):
    def __init__(self, attributes, root, subset, mode, state, cropping_transform,
                 img_transform=None, target_transform=None, logits_vac=False):
        for attr in attributes:
            assert isinstance(attr, Attribute)
        self._attrs = attributes
        self._attrs_values = [attr.key.value for attr in self._attrs]
        # mode is in ["paper", "branch"]
        self.mode = mode
        self.state = state
        self.data = self._make_dataset(root, subset)

        self.cropping_transform = cropping_transform
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.img_loader = opencv_loader
        self.logits_vac = logits_vac

    def _make_dataset(self, root, subset):
        assert subset in ['train', 'test']

        data = []
        if subset == 'train':
            anno_path = os.path.join(root, 'labels_train.txt')
        else:
            anno_path = os.path.join(root, 'labels_val.txt')

        with open(anno_path) as f:
            lines = f.readlines()
            lines = lines[:1000] if self.state else lines
            for line in lines:
                line_list = line.split()
                if line_list:  # may have []
                    img_name = line_list[0]
                    img_path = os.path.join(root, 'pictures', img_name)
                    for i in range(1, len(line_list), 16):
                        label = line_list[i:i+12]
                        box = list(map(lambda x: float(x), line_list[i+12:i+16]))
                        # there have 9 pictures' boxes have problems, so need to filter them
                        if box[2] < box[0] or box[3] < box[1]:
                            print(img_name, box)
                            continue
                        sample = dict(img=img_path, bbox=box)
                        recognizability = dict()

                        label = [label[ii] for ii in range(12) if ii in self._attrs_values]
                        for attr, l in zip(self._attrs, label):
                            if int(l) < 0:
                                recognizability[attr.key] = 0
                                sample[attr.key] = -10
                            else:
                                recognizability[attr.key] = 1
                                sample[attr.key] = int(l)
                        sample['recognizability'] = recognizability
                        data.append(sample)
        return data

    def __getitem__(self, index):
        sample = self.data[index]
        img_path = sample['img']
        bbox = sample['bbox']
        # print(img_path)

        img = self.img_loader(img_path)
        # print(img)
        crop = self.cropping_transform((img, bbox))

        # Transform target
        target = sample.copy()  # Copy sample so that the original one won't be modified
        target.pop('img')
        target.pop('bbox')
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.logits_vac:
            image_list = get_image_list(crop)
            # Transform image crop
            if self.img_transform is not None:
                result = []
                for img in image_list:
                    result.append(self.img_transform(img))
                # image_list = self.img_transform(image_list)
            return result, target
        else:
            # Transform image crop
            if self.img_transform is not None:
                crop = self.img_transform(crop)
            return crop, target

    def __len__(self):
        return len(self.data)
