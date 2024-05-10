###################################
#### CODE PROVIDED "AS IS"     ####
#### AUTHOR: Priler (Howdy Ho) ####
#### THIS IS FOR Yolov7        ####
###################################

import argparse
from pathlib import Path
import torch.multiprocessing as multiprocessing
import time
import cv2
import mss
import numpy
import sys
import math
import random

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

# import pyautogui
# import pydirectinput as pyautogui # due to default pyautogui.moveTo not works in game window
import win32api, win32con, win32gui
import keyboard

title = "FPS benchmark"
start_time = time.time()
display_time = 1 # displays the frame rate every 2 second
fps = 0
real_fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
monitor = {"top": 400, "left": 1200, "width": int(1312/2), "height": int(736/2)}
monitor = {"top": 200, "left": 1300, "width": 1312, "height": 736}

monitor = {"top": 100, "left": 100, "width": 1312, "height": 736}

_once = True
_resizeToFitMultiplyOf32 = False

SHOW_CV2_WINDOW = True # turning this off adds some FPS
HALF_CV2_WINDOW_SIZE = False # turn this on to half cv2 window size

# yolov7 params
weights = "./yolov7-csgoV4.pt" # model.pt path(s)
img_size = 640 # inference size (pixels)
conf_thres = 0.25 # object confidence threshold
iou_thres = 0.45 # IOU threshold for NMS
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False # class-agnostic NMS
augment = False # augmented inference
device = '' # cuda device, i.e. 0 or 0,1,2,3 or cpu
source = 0 # file/folder, 0 for webcam
trace = source
no_trace = False # don`t trace model
MIN_REQUIRED_CONF = 0.6 # minimum required confidence to show box (and process it further)

# DATASET COLLECTING PARAMS
DT_IMG_SAVE_PATH = "./data/collected/images/"
DT_LABEL_SAVE_PATH = "./data/collected/labels/"
DT_LABEL_SAVE_PATH = DT_IMG_SAVE_PATH
DT_LABEL_FORMAT = "{id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}"
DT_TEAM = "ct" # ct, t or mixed

DT_TEAM = input("Select team (ct, t, mixed): ")

# HERE YOU CAN MAKE CAPTURE_DELAY MORE, BUT FOR LESS CONFIDENCE
# OR YOU CAN RAISE CONFIDENCE AND LOWER CAPTURE_DELAY
# EITHER WAY, SUGGESTED PARAMS IS CAPTURE_DELAY=0.1 and CONFIDENCE=0.8
# BUT IT'S DEPENDS ON YOUR CURRENT MODEL AP
DT_CAPTURE_DELAY = 0.5 # how much seconds to wait until next detected frame will be saved in dataset
DT_REQUIRED_CONF = 0.5 # how much confidence required in order to save the box


def get_label_index(label):
    if label == "c":
        return 0
    elif label == "ch":
        return 1
    elif label == "t":
        return 2
    elif label == "th":
        return 3


def gen_dt_label_content(label, xmin, ymin, xmax, ymax, image_width, image_height):
    global DT_LABEL_FORMAT
    data = DT_LABEL_FORMAT

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    x_center_norm = abs(x_center) / image_width
    y_center_norm = abs(y_center) / image_height 

    width_norm = abs(xmax-xmin) / image_width
    height_norm = abs(ymax-ymin) / image_height

    data = data.replace("{id}", str(get_label_index(label)))
    data = data.replace("{x_center_norm}", "{:.4f}".format(x_center_norm))
    data = data.replace("{y_center_norm}", "{:.4f}".format(y_center_norm))
    data = data.replace("{width_norm}", "{:.4f}".format(width_norm))
    data = data.replace("{height_norm}", "{:.4f}".format(height_norm))

    return data


def save_dt_object(mss_img, label_content):
    global DT_IMG_SAVE_PATH, DT_LABEL_SAVE_PATH, monitor

    filename = "mss-{top}x{left}_{width}x{height}_{ts}".format(**monitor, ts = time_synchronized())
    img_path = f"{DT_IMG_SAVE_PATH}{filename}.png"
    label_path = f"{DT_LABEL_SAVE_PATH}{filename}.txt"

    # save image file
    mss.tools.to_png(mss_img.rgb, mss_img.size, output=img_path)

    # save label file
    with open(label_path, 'w') as f:
        f.write(label_content)

    return (img_path, label_path)



# print("===TEST===")
# print(gen_dt_label_content("ch", 330, 3, 503, 170, 640, 360))
# sys.exit()

# Initialize
set_logging()
device = select_device(device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(img_size, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, img_size)

if half:
    model.half()  # to FP16

# Second-stage classifier
classify = False

# cuDNN
cudnn.benchmark = True # set True to speed up constant image size inference

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


# ~ 13-15ms GRAB
# 

dt_last_capture = 0
# PROCESS
while True:
    mss_img = sct.grab(monitor)
    img = numpy.array(mss_img)
    # img = cv2.resize(img, (640, 352))

    #global device, model, half, stride, imgsz, classify,\
    #    names, colors, conf_thres, iou_thres, classes, agnostic_nms,\
    #    augment

    img0 = img # preserve for plotting & displaying
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # leave 3 channels
    img = numpy.moveaxis(img, -1, 0) # reshape for PyTorch (samples, channels, height, width)

    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # print(img.size())
    # pad image dimensions to be multiple of 32 (ex. 1280x720 to 1312x736)
    # note.
    # Keras format is (samples, height, width, channels)
    # PyTorch is (samples, channels, height, width)
    if _resizeToFitMultiplyOf32:
        padding1_mult = math.floor(img.shape[2] / 32) + 1
        padding2_mult = math.floor(img.shape[3] / 32) + 1
        pad1 = (32 * padding1_mult) - img.shape[2]
        pad2 = (32 * padding2_mult) - img.shape[3] 
        padding = torch.nn.ReplicationPad2d((0, pad2, pad1, 0, 0 ,0))

        img = padding(img)

    if _once:
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        _once = False

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t3 = time_synchronized()


    #t100 = time_synchronized()
    #t200 = time_synchronized()
    #print(f"{int(1E3 * (t200 - t100))}ms PER LOOP")
    #sys.exit()

    # Process detections
    dataset = []
    for i, det in enumerate(pred):  # detections per image
        s = ''

        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Enemies list
            e_list = []

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if conf < MIN_REQUIRED_CONF:
                    continue

                # Add bbox to image
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)


                if conf >= DT_REQUIRED_CONF:
                    label = names[int(cls)]
                    if DT_TEAM == "ct":
                        # only detect T
                        if label == "c":
                            label = "t"
                        elif label == "ch":
                            label = "th"
                    elif DT_TEAM == "t":
                        # only detect CT
                        if label == "t":
                            label = "c"
                        elif label == "th":
                            label = "ch"

                    # recalibrate labels (mixed is ONLY for wrong trained model)
                    #if label == "c":
                    #    label = "ch"
                    #elif label == "ch":
                    #    label = "c"
                    #elif label == "t":
                    #    label = "th"
                    #elif label == "th":
                    #    label = "t"

                    dataset.append([label, xyxy])

    if(dataset):
        if (time_synchronized() - dt_last_capture) > DT_CAPTURE_DELAY:
            dataset_content = ""
            for dt in dataset:
                if dataset_content != "":
                    dataset_content += "\n"
                dataset_content += gen_dt_label_content(dt[0], dt[1][0], dt[1][1], dt[1][2], dt[1][3], int(monitor['width']), int(monitor['height']))

            dt_save_result = save_dt_object(mss_img, dataset_content)
            print(f"Dataset item saved as {dt_save_result[0]}")
            # sys.exit()
            
            dt_last_capture = time_synchronized()



    # Print time (inference + NMS)
    # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    # torch.cuda.empty_cache()

    # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # Calculate FPS
    fps+=1
    TIME = time.time() - start_time
    if (TIME) >= display_time :
        print("FPS: ", fps / (TIME))
        real_fps = int(fps / (TIME))
        fps = 0
        start_time = time.time()

    if SHOW_CV2_WINDOW:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img0, str(real_fps), (7, 40), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # imS = cv2.resize(img0, (854, 480))

        # Display the picture
        if HALF_CV2_WINDOW_SIZE:
            img0 = cv2.resize(img0, (int(monitor['width']/2), int(monitor['height']/2)))

        cv2.imshow(title, cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        # cv2.imshow(title, img0)

        cv2.waitKey(1)