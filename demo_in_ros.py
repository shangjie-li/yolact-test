import os
import time
import threading
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess
from data import cfg, set_cfg, set_dataset


parser = argparse.ArgumentParser(
    description='Demo script for YOLACT')
parser.add_argument('--trained_model', default='weights/coco/yolact_resnet50_25_380000.pth', type=str,
    help='Weights of the model.')
parser.add_argument('--dataset', default=None, type=str,
    help='If specified, override the dataset specified in the config.')
parser.add_argument('--score_threshold', default=0.3, type=float,
    help='The score threshold for detection.')
parser.add_argument('--top_k', default=20, type=int,
    help='The number of objects for detection.')
parser.add_argument('--class_filter', default=None, type=str,
    help='A filter to keep desired classes, e.g., 0/2/5/7 (split by a slash).')
parser.add_argument('--sub_image', default='/kitti/camera_color_left/image_raw', type=str,
    help='The image topic to subscribe.')
parser.add_argument('--pub_image', default='/result', type=str,
    help='The image topic to publish.')
parser.add_argument('--frame_rate', default=10, type=int,
    help='Working frequency.')
parser.add_argument('--display', action='store_true',
    help='Whether to display and save all videos.')
parser.add_argument('--print', action='store_true',
    help='Whether to print and record infos.')
args = parser.parse_args()


image_lock = threading.Lock()


def get_stamp(header):
    return header.stamp.secs + 0.000000001 * header.stamp.nsecs


def publish_image(pub, data, frame_id='base_link'):
    assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
    header = Header(stamp=rospy.Time.now())
    header.frame_id = frame_id

    msg = Image()
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = 'rgb8'
    msg.data = np.array(data).tobytes()
    msg.header = header
    msg.step = msg.width * 1 * 3

    pub.publish(msg)


def display(img, v_writer, win_name='result'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    v_writer.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        v_writer.release()
        return False
    else:
        return True


def print_info(frame, stamp, delay, labels, scores, boxes, file_name='result.txt'):
    time_str = 'frame:%d  stamp:%.3f  delay:%.3f' % (frame, stamp, delay)
    print(time_str)
    with open(file_name, 'a') as fob:
        fob.write(time_str + '\n')
    for i in range(len(labels)):
        info_str = 'box:%d %d %d %d  score:%.2f  label:%s' % (
            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], labels[i]
        )
        print(info_str)
        with open(file_name, 'a') as fob:
            fob.write(info_str + '\n')
    print()
    with open(file_name, 'a') as fob:
        fob.write('\n')


def create_random_color():
    r = np.random.randint(0, 255)
    g = np.random.randint(0, 255)
    b = np.random.randint(0, 255)
    color = (r, g, b)

    return color


def draw_mask(img, mask, color):
    """
    Args:
        img: numpy.ndarray, (h, w, 3), BGR format
        mask: torch.Tensor, (h, w)
        color: tuple
    Returns:
        img_numpy: numpy.ndarray, (h, w, 3), BGR format
    """
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


def draw_segmentation_result(img, mask, label, score, box, color):
    """
    Args:
        img: numpy.ndarray, (h, w, 3), BGR format
        mask: torch.Tensor, (h, w)
        label: str
        score: float
        box: numpy.ndarray, (4,), xyxy format
        color: tuple
    Returns:
        img: numpy.ndarray, (h, w, 3), BGR format
    """
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_thickness, line_thickness = 1, 2

    x1, y1, x2, y2 = box.squeeze()
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)
    img = draw_mask(img, mask, color)

    u, v = x1, y1
    text_str = '%s: %.2f' % (label, score)
    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    cv2.rectangle(img, (u, v), (u + text_w, v - text_h - 4), color, -1)
    cv2.putText(img, text_str, (u, v - 3), font_face, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return img


class YolactDetector():
    def __init__(self, trained_model, dataset=None):
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        pth = SavePath.from_str(trained_model)
        config = pth.model_name + '_config'
        set_cfg(config)

        if dataset is not None:
            set_dataset(dataset)

        self.net = Yolact()
        self.net.load_weights(trained_model)
        self.net.eval()
        self.net = self.net.cuda()
        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False

    def run(self, img, score_threshold=0.3, top_k=20, class_filter=None):
        """
        Args:
            img: (h, w, 3), RGB format
            score_threshold: float, object confidence threshold
            top_k: int, number of objects for detection
            class_filter: list(int), filter by class, for instance [0, 2, 3]
        Returns:
            masks: torch.Tensor, (n, h, w)
            labels: list(str), names of objects
            scores: list(float)
            boxes: numpy.ndarray, (n, 4), xyxy format
        """
        frame = torch.from_numpy(img).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)

        with torch.no_grad():
            h, w, _ = frame.shape
            with timer.env('Postprocess'):
                save = cfg.rescore_bbox
                cfg.rescore_bbox = True
                t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True,
                                score_threshold=score_threshold)
                cfg.rescore_bbox = save

            with timer.env('Copy'):
                idx = t[1].argsort(0, descending=True)[:top_k]
                masks = t[3][idx]
                labels, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        if class_filter is not None:
            selected_indices = [i for i in range(labels.shape[0]) if labels[i] in class_filter]
            labels = labels[selected_indices]
            scores = scores[selected_indices]
            boxes = boxes[selected_indices]
            masks = masks[selected_indices]

        labels = [cfg.dataset.class_names[label] for label in labels]
        scores = [score for score in scores]
        return masks, labels, scores, boxes


def image_callback(image):
    global image_stamp, image_frame
    image_lock.acquire()
    image_stamp = get_stamp(image.header)
    image_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1) # BGR image
    image_lock.release()


def timer_callback(event):
    global image_stamp, image_frame
    image_lock.acquire()
    cur_stamp = image_stamp
    cur_frame = image_frame[:, :, ::-1].copy() # to RGB
    image_lock.release()
    
    global frame
    frame += 1
    start = time.time()
    masks, labels, scores, boxes = detector.run(
        cur_frame, score_threshold=score_threshold, top_k=top_k, class_filter=class_filter)

    cur_frame = cur_frame[:, :, ::-1].copy() # to BGR
    for i in np.argsort(scores):
        color = create_random_color()
        cur_frame = draw_segmentation_result(
            cur_frame, masks[i], str(labels[i]), float(scores[i]), boxes[i], color)
    
    if args.display:
        if not display(cur_frame, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")

    cur_frame = cur_frame[:, :, ::-1].copy() # to RGB
    publish_image(pub, cur_frame)
    delay = round(time.time() - start, 3)
    
    if args.print:
        print_info(frame, cur_stamp, delay, labels, scores, boxes, file_name)


if __name__ == '__main__':
    rospy.init_node("yolact", anonymous=True, disable_signals=True)
    frame = 0

    if args.print:
        file_name = 'result.txt'
        with open(file_name, 'w') as fob:
            fob.seek(0)
            fob.truncate()

    assert os.path.exists(args.trained_model), '%s Not Found' % args.trained_model
    detector = YolactDetector(trained_model=args.trained_model, dataset=args.dataset)
    score_threshold = args.score_threshold
    top_k = args.top_k
    class_filter = list(map(int, args.class_filter.split('/'))) if args.class_filter is not None else None

    image_stamp = None
    image_frame = None
    rospy.Subscriber(args.sub_image, Image, image_callback, queue_size=1, buff_size=52428800)
    while image_frame is None:
        time.sleep(0.1)
        print('Waiting for topic %s...' % args.sub_image)
    print('  Done.\n')

    if args.display:
        win_h, win_w = image_frame.shape[0], image_frame.shape[1]
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)

    pub = rospy.Publisher(args.pub_image, Image, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback)
    rospy.spin()
