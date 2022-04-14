import matplotlib.pyplot as plt # For WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from data import COCODetection, get_label_map, COLORS
from data import cfg, set_cfg, set_dataset
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default=None, type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=20, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display', dest='display', default=False, action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--score_threshold', default=0.10, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')

    global args
    args = parser.parse_args(argv)
    

def create_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def draw_mask(img, mask, color):
    """
    Args:
        img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
        mask <class 'torch.Tensor'> torch.Size([frame_height, frame_width])
        color <class 'tuple'>
        
    Returns:
        img_numpy <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    """
    img_gpu = torch.from_numpy(img).cuda().float()
    img_gpu = img_gpu / 255.0
    mask = mask[:, :, None]
    
    # color_tensor <class 'torch.Tensor'> torch.Size([3])
    color_tensor = torch.Tensor(color).to(img_gpu.device.index).float() / 255.
    alpha = 0.45
    mask_color = mask.repeat(1, 1, 3) * color_tensor * alpha
    inv_alph_mask = mask * (- alpha) + 1
    img_gpu = img_gpu * inv_alph_mask + mask_color
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    return img_numpy


def draw_segmentation_result(img, mask, classname, score, box, color):
    """
    Args:
        img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
        mask <class 'torch.Tensor'> torch.Size([frame_height, frame_width])
        classname <class 'str'>
        score <class 'float'>
        box <class 'numpy.ndarray'> (4,)
        color <class 'tuple'>
        
    Returns:
        img <class 'numpy.ndarray'> (frame_height, frame_width, 3)
    """
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_thickness, line_thickness = 1, 2
    
    x1, y1, x2, y2 = box[:]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    img = draw_mask(img, mask, color)
    
    u, v = x1, y1
    text_str = '%s: %.2f' % (classname, score)
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    
    cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return img


def process_dets(dets_out, img_h, img_w):
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, img_w, img_h, visualize_lincomb=False, crop_masks=True, score_threshold=args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        ids, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    
    classes = []
    for i in range(ids.shape[0]):
        name = cfg.dataset.class_names[ids[i]]
        classes.append(name)
    classes = np.array(classes)
    
    remains = []
    num_item = 0
    for i in range(classes.shape[0]):
        if num_item < args.top_k and scores[i] > args.score_threshold:
            remains.append(i)
            num_item += 1
            
    masks, classes, scores, boxes = masks[remains], classes[remains], scores[remains], boxes[remains]
    return masks, classes, scores, boxes


def evalimage(net:Yolact, path:str, save_path:str=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    img_h, img_w, _ = frame.shape
    masks, classes, scores, boxes = process_dets(preds, img_h, img_w)
    
    if args.display:
        img_out = cv2.imread(path)
        for i in range(classes.shape[0]):
            mask = masks[i]
            classname = str(classes[i])
            score = float(scores[i])
            box = boxes[i]
            color = create_random_color()
            img_out = draw_segmentation_result(img_out, mask, classname, score, box, color)
        
        cv2.imshow('img_out', img_out)
        cv2.waitKey(0)
    
    basename = os.path.basename(path)
    if save_path is None:
        save_path = '.'.join(basename.split('.')[:-1]) + '.json'
    
    json_data = {}
    json_data['version'] = "3.16.2"
    json_data['flags'] = {}
    json_data['shapes'] = []
    
    for i in range(classes.shape[0]):
        obj = {}
        obj['label'] = classes[i]
        
        binary = masks[i].byte().cpu().numpy()
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            num_points = 0
            best_idx = 0
            for j in range(len(contours)):
                if contours[j].shape[0] > num_points:
                    num_points = contours[j].shape[0]
                    best_idx = j
            
            points = np.array(contours[best_idx]).squeeze()
            obj['points'] = points.tolist()
            obj['shape_type'] = 'polygon'
            obj['flags'] = {}
            obj['line_color'] = None
            obj['fill_color'] = None
            json_data['shapes'].append(obj)
        
    json_data['lineColor'] = [0, 255, 0, 128]
    json_data['fillColor'] = [255, 0, 0, 128]
    json_data['imageData'] = None
    json_data['imagePath'] = basename
    json_data['imageHeight'] = img_h
    json_data['imageWidth'] = img_w
    json.dump(json_data, open(save_path, 'w'), indent=1)
    print('Saving json file to %s.' % save_path)


def str2int(x):
    try:
        return int(x)
    except:
        y = ''
        for i in x:
            if i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                y += i
        return int(y)


def evalimages(net:Yolact, input_folder:str, output_folder:str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    files = os.listdir(input_folder)
    files.sort(key = lambda x: str2int(x.split('.')[0]))
    for image in files: 
        basename = os.path.basename(image)
        inp_path = os.path.join(input_folder, image)
        out_path = os.path.join(output_folder, '.'.join(basename.split('.')[:-1]) + '.json')

        evalimage(net, inp_path, out_path)
        print(inp_path + ' -> ' + out_path)
    print('Done.')


if __name__ == '__main__':
    parse_args()
    
    if args.trained_model is None:
        print('Error: args.trained_model must be set.')
        exit()
    else:
        assert Path(args.trained_model).exists(), 'Not Found: %s' % args.trained_model

    if args.config is not None:
        set_cfg(args.config)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.image is None and args.images is None:
            print('Error: args.image or args.images must be set.')
            exit()

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        net.detect.use_fast_nms = args.fast_nms
        net.detect.use_cross_class_nms = args.cross_class_nms
        cfg.mask_proto_debug = args.mask_proto_debug
    
        # TODO Currently we do not support Fast Mask Re-scroing in evalimage, evalimages
        if args.image is not None:
            if ':' in args.image:
                inp, out = args.image.split(':')
                evalimage(net, inp, out)
            else:
                evalimage(net, args.image)
        elif args.images is not None:
            if ':' in args.images:
                inp, out = args.images.split(':')
                evalimages(net, inp, out)
            else:
                print('Error: An input folder of images and output folder of labels must be specified.' \
                    + 'Should be in the format `input_folder:output_folder`.')
                exit()


