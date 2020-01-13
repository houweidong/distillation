from torch.utils.data import Dataset
import os
from data.attributes import WiderAttributes as WdAt, Attribute, AttributeType as AtTp
from torchvision.datasets.folder import pil_loader
import json
from data.image_loader import opencv_loader


class NewdataAttr(Dataset):
    def __init__(self, attributes, root, subset, mode, state, cropping_transform,
                 img_transform=None, target_transform=None):
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


    def _make_dataset(self, root, subset):
        assert subset in ['train', 'test']

        data = []

        # Assuming dataset directory is layout as
        # Wider Attribute
        #   -- Anno
        #     -- wider_attribute_trainval.json
        #     -- wider_attribute_test.json
        #   -- Image
        #     -- train
        #       -- 0--Parade
        #         -- ...
        #     -- val
        #     -- test.py
        # test set end with 3, 6, 9
        anno_dir = os.path.join(root, 'labels1')
        anno_list = list(filter(lambda x: x.endswith('txt'), os.listdir(anno_dir)))

        for anno_txt in anno_list:
            anno_path = os.path.join(anno_dir, anno_txt)
            with open(anno_path) as f:
                lines = f.readlines()
                lines = lines[:20] if self.state else lines
                if subset == 'train':
                    lines = [lines[i] for i in range(len(lines)) if not str(i).endswith(('3' '6', '9'))]
                else:
                    lines = [lines[i] for i in range(len(lines)) if str(i).endswith(('3' '6', '9'))]
                for line in lines:
                    # if line.startswith('img10.360buyimg.com_n1_jfs_t1_33182_1_6378_140673_5cbdf607E79137fee_6846c2e9c4f8d7f1.jpg'):
                    #     a = 5
                    line_list = line.split()
                    if line_list:  # may have []
                        img_name = line_list[0]
                        img_path = os.path.join(root, 'pictures', anno_txt.rstrip('.txt'), img_name)
                        for i in range(1, len(line_list), 16):
                            label = line_list[i:i+12]
                            box = list(map(lambda x: float(x), line_list[i+12:i+16]))
                            # there have 9 pictures' boxes have problems, so need to filter them
                            if box[2] < box[0] or box[3] < box[1]:
                                continue
                            sample = dict(img=img_path, bbox=box)
                            recognizability = dict()

                            label = [label[ii] for ii in range(12) if ii in self._attrs_values]
                            for attr, l in zip(self._attrs, label):
                                # for the dadunan
                                if attr.branch_num == 1 and int(l) == 2:
                                    attr = attr.key
                                    recognizability[attr] = 1
                                    sample[attr] = 1
                                else:
                                    attr = attr.key
                                    if int(l) < 0:
                                        recognizability[attr] = 0
                                        sample[attr] = -10
                                    else:
                                        recognizability[attr] = 1
                                        sample[attr] = int(l)
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

        # Transform image crop
        if self.img_transform is not None:
            crop = self.img_transform(crop)

        # Transform target
        target = sample.copy()  # Copy sample so that the original one won't be modified
        target.pop('img')
        target.pop('bbox')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return crop, target

    def __len__(self):
        return len(self.data)
