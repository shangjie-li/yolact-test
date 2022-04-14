import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from data import COCODetection
from data import cfg, set_cfg, set_dataset, MEANS, STD
from utils.augmentations import SSDAugmentation, BaseTransform

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Dataset Displayer')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--training', action='store_true',
                        help='Whether or not to use training set.')

    global args
    args = parser.parse_args(argv)

def create_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    
    color = (r, g, b)
    return color

def draw_mask(img, mask, color):
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0
    
    mask = mask[:, :, None]
    color_tensor = torch.Tensor(color).to(img_gpu.device.index).float() / 255.
    alpha = 0.45
    
    mask_color = mask.repeat(1, 1, 3) * color_tensor * alpha
    inv_alph_mask = mask * (- alpha) + 1
    img_gpu = img_gpu * inv_alph_mask + mask_color
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    return img_numpy

def draw_annotation(img, mask, box, classname, color, score=None):
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.4
    font_thickness, line_thickness = 1, 2
    
    x1, y1, x2, y2 = box[:]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    
    img = draw_mask(img, mask, color)
    
    u, v = int(x1), int(y1)
    text_str = '%s: %.2f' % (classname, score) if score else classname
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    
    if v - text_h - 4 < 0: v = text_h + 4
    cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img

if __name__ == '__main__':
    parse_args()
    if args.dataset:
        set_dataset(args.dataset)
    cfg.dataset.print()
    print()
    
    if args.training:
        dataset = COCODetection(image_path=cfg.dataset.train_images,
                                info_file=cfg.dataset.train_info,
                                transform=SSDAugmentation(),
                                dataset_name=cfg.dataset.name)
    else:
        dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                info_file=cfg.dataset.valid_info,
                                transform=BaseTransform(),
                                dataset_name=cfg.dataset.name)
    
    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i + 1, len(dataset)))
        img_id_i, file_name, img_raw, height, width = dataset.pull_image(i) # BGR
        
        if args.training:
            img_tensor, (gt, masks, _) = dataset[i]
            boxes, labels = gt[:, :4], gt[:, 4]
            _, h, w = img_tensor.shape
            img_array = img_tensor.cpu().numpy().astype(np.float32).transpose(1, 2, 0)
            img_array = (img_array * STD) + MEANS
            img_array = np.clip(img_array, a_min=0, a_max=255)
            img_array = img_array.astype(np.uint8)[:, :, ::-1] # to BGR
            img_array = img_array.copy()
            scale = [w, h, w, h]
        else:
            img_tensor, (gt, masks, _) = dataset[i]
            boxes, labels = gt[:, :4], gt[:, 4]
            img_array = img_raw.copy()
            scale = [width, height, width, height]
        
        for j in range(boxes.shape[0]):
            mask = torch.from_numpy(masks[j]).cuda().float()
            box = boxes[j] * scale
            label = int(labels[j])
            if label >= 0:
                classname = cfg.dataset.class_names[label]
            else:
                classname = 'crowd'
            color = create_random_color()
            img_array = draw_annotation(img_array, mask, box, classname, color)
        
        cv2.imshow('Raw Data', img_raw)
        cv2.imshow('Processed Data', img_array)
        
        print('index:', img_id_i)
        print('shape:', img_raw.shape)
        print('labels:\n', gt)
        print()
        
        # press 'Esc' to shut down, and every key else to continue
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            continue
