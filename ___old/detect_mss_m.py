import argparse
from pathlib import Path
import torch.multiprocessing as multiprocessing
import time
import cv2
import mss
import numpy
import sys
import math

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

title = "FPS benchmark"
start_time = time.time()
display_time = 1 # displays the frame rate every 2 second
fps = 0
real_fps = 0
sct = mss.mss()
# Set monitor size to capture
monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
monitor = {"top": 466, "left": 1239, "width": 1280, "height": 720}

# yolov7 params
weights = "./yolov7.pt" # model.pt path(s)
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

# p1
def GRABMSS_screen(q):
    while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # To get real color we do this:
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q.put_nowait(img)
        q.join()

# p2
def DETECT_screen(q):
    if not q.empty():
        img = q.get_nowait()
        q.task_done()

        # show
        q.put_nowait(img)
        q.join()
        print("TESTTTT")
        return

        # detect
        global device, model, half, stride, imgsz, classify,\
        names, colors, conf_thres, iou_thres, classes, agnostic_nms,\
        augment

        img0 = img # preserve for plotting & displaying
        img = numpy.moveaxis(img, -1, 0) # reshape for PyTorch (samples, channels, height, width)

        old_img_w = old_img_h = imgsz
        old_img_b = 1
        t0 = time.time()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        print(img.size())

        # pad image dimensions to be multiple of 32 (ex. 1280x720 to 1312x736)
        # note.
        # Keras format is (samples, height, width, channels)
        # PyTorch is (samples, channels, height, width)
        padding1_mult = math.floor(img.shape[2] / 32) + 1
        padding2_mult = math.floor(img.shape[3] / 32) + 1
        pad1 = (32 * padding1_mult) - img.shape[2]
        pad2 = (32 * padding2_mult) - img.shape[3] 
        padding = torch.nn.ReplicationPad2d((0, pad2, pad1, 0, 0 ,0))

        img = padding(img)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Process detections
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

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # torch.cuda.empty_cache()

        # show
        q.put_nowait(img0)
        q.join()

# p3
def SHOWMSS_screen(q):
    global fps, start_time, real_fps
    while True:
        if not q.empty():
            img = q.get_nowait()
            q.task_done()
            # To get real color we do this:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(real_fps), (7, 40), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

            imS = cv2.resize(img, (854, 480))

            # Display the picture
            cv2.imshow(title, cv2.cvtColor(imS, cv2.COLOR_BGR2RGB))
            # Calculate FPS
            fps+=1
            TIME = time.time() - start_time
            if (TIME) >= display_time :
                print("FPS: ", fps / (TIME))
                real_fps = int(fps / (TIME))
                fps = 0
                start_time = time.time()
            # Press "q" to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

if __name__=="__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.set_start_method('spawn', force=True)

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

    # Queue
    q = multiprocessing.JoinableQueue()
    # creating new processes
    p1 = multiprocessing.Process(target=GRABMSS_screen, args=(q, ))
    p2 = multiprocessing.Process(target=DETECT_screen, args=(q, ))
    p3 = multiprocessing.Process(target=SHOWMSS_screen, args=(q, ))

    # starting our processes
    
    
    p1.start()
    p2.start()
    p3.start()