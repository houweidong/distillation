import os
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, \
    RandomHorizontalFlip, Normalize, RandomRotation, ColorJitter

from data.transforms import square_no_elastic, get_inference_transform_person_lr
from utils.opts import parse_opts
from data.image_loader import opencv_loader, cv_to_pil_image
import cv2
from models.generate_model import get_model as get_m
from utils.get_tasks import get_tasks
import matplotlib.pyplot as plt


def get_input(cuda=True, transform=None, box=None, path=None, camera=False, cap_img=None):
    pic_path = opt.img_path if not path else path
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    val_img_transform = Compose(
        [square_no_elastic,
         Resize((opt.person_size, opt.person_size)),
         ToTensor(), Normalize(mean, std)])
    img_ori = cap_img if camera else cv2.imread(pic_path)
    # img = opencv_loader(pic_path)
    img = cv_to_pil_image(img_ori)
    if transform:
        img = transform((img, box))
    img = val_img_transform(img)
    # print(img)
    img = img.unsqueeze(0)
    if cuda:
        img = img.cuda()
    return img_ori, img


def get_model(cuda=True):
    attr, _ = get_tasks(opt)
    attr, attr_name = get_tasks(opt)
    device = 'cuda' if cuda else 'cpu'
    model, _, _ = get_m(opt.conv, device=device, classifier=opt.classifier, attr=attr)

    # load the model, need to move the prefix "module."
    state_dict = torch.load(opt.model_path, map_location='cpu')
    # for k in list(state_dict.keys()):
    #     k_new = k[7:]
    #     state_dict[k_new] = state_dict[k]
    #     state_dict.pop(k)
    model.load_state_dict(state_dict, strict=True)

    if cuda:
        model = model.cuda()
    model.eval()
    return model


def camera(model, wait=10):
    # detect from camera
    cap = cv2.VideoCapture(-1)
    ret, _ = cap.read()

    while ret:
        ret, frame = cap.read()
        # some pre-process
        # frame = np.uint8(np.clip((0.9 * frame + 30), 0, 255))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        # frame = cv2.filter2D(frame, -1, kernel=kernel)
        img_ori, img = get_input(camera=True, cap_img=frame)
        tensor_p = model(img)
        display(img_ori, tensor_p, title='Camera', wait=wait)
        ret, frame = cap.read()


def vedio(model, path):

    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('.\\log\\output.avi', fourcc, fps, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # frame = cv2.flip(frame, 0)
            img_ori, img = get_input(camera=True, cap_img=frame)
            tensor_p = model(img)
            display(img_ori, tensor_p, title='Camera', wait=1/fps*1000)
            out.write(img_ori)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


def display(im, tensor_p, title='result', wait=0):
    probs = []
    for i, tensor in enumerate(tensor_p):
        probs.append(tensor_p[i].cpu().detach().numpy()[0])
    # probs = tensor_p.cpu().detach().numpy()[0]
    start = 20
    for i, attr in enumerate(opt.specified_attrs):
        if len(probs[i]) != 1:
            for j in range(len(probs[i])):
                caption = "{}:{:.2f}".format(attr+str(j), probs[i][j])
                im = cv2.putText(
                    im, caption, (0, start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                )
                start += 20
        else:
            caption = "{}:{:.2f}".format(attr,  probs[i][0])
            im = cv2.putText(
                im, caption, (0, start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
            start += 20
    cv2.imshow(title, im)
    cv2.waitKey(wait)


def test_dir(model, subset='val'):
    root = "/root/dataset/dpan/dataset/new_data"
    # path = "/root/dataset/new/pictures/jinshenyi/"
    # label = "/root/dataset/new/labels1/jinshenyi/"

    if subset == 'train':
        anno_path = os.path.join(root, 'labels_train.txt')
    else:
        anno_path = os.path.join(root, 'labels_val.txt')

    with open(anno_path) as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            if line_list:  # may have []
                img_name = line_list[0]
                img_path = os.path.join(root, 'pictures', img_name)
                for i in range(1, len(line_list), 16):
                    label = line_list[i:i + 12]
                    box = list(map(lambda x: float(x), line_list[i + 12:i + 16]))
                    # there have 9 pictures' boxes have problems, so need to filter them
                    if box[2] < box[0] or box[3] < box[1]:
                        print(img_name, box)
                        continue

                    img_ori, img = get_input(transform=get_inference_transform_person_lr, box=box, path=img_path)
                    output = model(img)
                    display(img_ori, output)


opt = parse_opts()
opt.pretrain = False
model = get_model()
if opt.test_mode == 'train_dir':
    test_dir(model)
elif opt.test_mode == 'pic':
    img_ori, img = get_input()
    output = model(img)
    display(img_ori, output)
elif opt.test_mode == 'camera':
    camera(model, wait=10)
elif opt.test_mode == 'vedio':
    vedio(model, opt.vedio_path)

