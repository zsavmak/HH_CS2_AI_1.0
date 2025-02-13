####################################
#### CODE PROVIDED "AS IS"      ####
#### AUTHOR: Priler (Howdy Ho)  ####
####################################

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
import grab_screen
import dxcam
from PIL import Image

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
grab_monitor = monitor

_once = True
_resizeToFitMultiplyOf32 = False

GRAB_SCREEN_METHOD = "mss" # mss, win32, dxcam, dxcamcapture
SHOW_CV2_WINDOW = True # turning this off adds some FPS
CV2_RESIZE_TO = None # None or size (w, h)

# yolov7 params
weights = "./yolov7-csgoV5_1024.pt" # model.pt path(s)
img_size = 640 # inference size (pixels)
conf_thres = 0.1 # object confidence threshold (0.25 def)
iou_thres = 0.45 # IOU threshold for NMS
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False # class-agnostic NMS
augment = False # augmented inference
device = '' # cuda device, i.e. 0 or 0,1,2,3 or cpu
source = 0 # file/folder, 0 for webcam
trace = source
no_trace = False # don`t trace model
MIN_REQUIRED_CONF = 0.7 # minimum required confidence to show box (and process it further)


# pyautogui settings
AUTO_SHOOT = False # rather apply autoshoot or press button to shoot
HOTKEY_TO_SHOOT = 58
HOTKEY_TO_CHANGE_TEAM = "ctrl+t"
shoot_conf = (0.75, 0.8) # minimum required conf for detection to shoot (body, head)
screen = (3440, 1440) # screen width & height
team = "ct" # t or ct
enemy_team = "t"
t_classes = ("t", "th")
ct_classes = ("c", "ch")
# pyautogui.MINIMUM_DURATION = 0
# pyautogui.MINIMUM_SLEEP = 0
# pyautogui.PAUSE = 0
hotkey_to_shoot_pressed = False

if GRAB_SCREEN_METHOD == "dxcam":
    dxcamera = dxcam.create()

elif GRAB_SCREEN_METHOD == "dxcamcapture":
    dxcamera = dxcam.create()
    dxcamera.start(region=(
            grab_monitor['left'], grab_monitor['top'],
            grab_monitor['width']+grab_monitor['left'], grab_monitor['height']+grab_monitor['top']
        ))

    if dxcamera.is_capturing:
        print("DXCAMERA capturing started ...")
    else:
        print("DXCAMERA capture error.")
        sys.exit()


# enemy classes
def id_enemy_classes():
    global team, e_classes, enemy_team

    if team == "ct":
        e_classes = t_classes
        enemy_team = "t"
    else:
        e_classes = ct_classes
        enemy_team = "ct"

id_enemy_classes()

def getPointOnCurve(x1, y1, x2, y2, n):
    """Returns the (x, y) tuple of the point that has progressed a proportion
    n along the curve defined by the two x, y coordinates.
    If the movement length for X is great than Y, then Y offset else X
    """

    x = ((x2 - x1) * n) + x1
    y = ((y2 - y1) * n) + y1

    return (x, y)


def Shoot(mid_x, mid_y):
    global AUTO_SHOOT, KEY_TO_SHOOT, hotkey_to_shoot_pressed, monitor, screen

    if not AUTO_SHOOT:
        if not hotkey_to_shoot_pressed:
            return
        else:
            hotkey_to_shoot_pressed = False

    # x = int(mid_x*width)
    #y = int(mid_y*height)
    # y = int(mid_y*height+height/9)
    
    # x = int(mid_x * monitor['width'])
    # y = int(mid_y * monitor['height']+monitor['height']/9)

    # pyautogui.moveTo(x, y)
    # pyautogui.click()

    flags, hcursor, (cx, cy) = win32gui.GetCursorInfo()

    # normalize cx & cy for rel move
    cx -= monitor['left']
    cy -= monitor['top']

    print(f"mid_x: {mid_x}, mid_y: {mid_y}")
    print(f"cx: {cx}, cy: {cy}")

    # xdiff = w / uniform_target_resolution
    scale_x = screen[0] / monitor['width']
    scale_y = screen[1] / monitor['height']

    print(f"scale_x: {scale_x}, scale_y: {scale_y}")

    x = int((mid_x - cx) * scale_x)
    y = int((mid_y - cy) * scale_y)

    print(f"x: {x}, y: {y}")

    ### LERPING ###
    USE_LERP = True # Use lerp aiming (True) or teleport aiming (False)
    # flags, hcursor, (cx, cy) = win32gui.GetCursorInfo()
    xy_from = (cx, cy)
    rel_move = [x, y] # based on upper calculations

    if abs(rel_move[0]) > monitor['width'] / 2:
        rel_move[0] = 0
    if abs(rel_move[1]) > monitor['height'] / 2:
        rel_move[1] = 0

    print(f"REL MOVE - {rel_move[0]}, {rel_move[1]}")
    final_point = (xy_from[0]+rel_move[0], xy_from[1]+rel_move[1])
    lerp_progress = 0
    lerp_step = 0.01 # aka Aiming Speed (0.01 slow, 0.05 medium, 0.1 fast)
    use_jitter = False
    jitter = ((-1, 1), (-1, 1))

    if USE_LERP:
        # lerp mouse movement
        while lerp_progress < 1:
            x, y = getPointOnCurve(0, 0, rel_move[0], rel_move[1], lerp_step)

            if use_jitter and lerp_progress < .9:
                x += random.randint(jitter[0][0], jitter[0][1])
                y += random.randint(jitter[1][0], jitter[1][1])

            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
            lerp_progress += lerp_step
        else:
            flags, hcursor, (cx, cy) = win32gui.GetCursorInfo()

            # normalize cx & cy for rel move
            cx -= monitor['left']
            cy -= monitor['top']

            target_rel = (final_point[0] - cx, final_point[1] - cy)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(target_rel[0]), int(target_rel[1]), 0, 0)

            x = int(target_rel[0])
            y = int(target_rel[1])
    else:
        # teleport mouse movement
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
        time.sleep(0.01)

    time.sleep(0.02)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
    print(f"SHOOT! {x}, {y}")


def Shoot_hotkey(triggered, hotkey):
    global hotkey_to_shoot_pressed

    hotkey_to_shoot_pressed = True

    print("HOTKEY PRESSED!!!!!!!!!!!!!!")


def Change_team(triggered, hotkey):
    global team, enemy_team

    if team == "t":
        team = "ct"
    else:
        team = "t"

    id_enemy_classes()
    print(f"NEW TEAM {team}")
    print(f"ENEMY TEAM {enemy_team}")


# register shoot hotkey (if required)
if not AUTO_SHOOT:
    keyboard.add_hotkey(HOTKEY_TO_SHOOT, Shoot_hotkey, args=('triggered', 'hotkey'))

keyboard.add_hotkey(HOTKEY_TO_CHANGE_TEAM, Change_team, args=('triggered', 'hotkey'))

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

# PROCESS
while True:
    if GRAB_SCREEN_METHOD == "mss":
        img = numpy.array(sct.grab(grab_monitor))

    elif GRAB_SCREEN_METHOD == "win32":
        img = numpy.array(grab_screen.grab((
            grab_monitor['left'], grab_monitor['top'],
            grab_monitor['width']+grab_monitor['left']-1, grab_monitor['height']+grab_monitor['top']-1
        )))

    elif GRAB_SCREEN_METHOD == "dxcam":
        img = dxcamera.grab(region=(
            grab_monitor['left'], grab_monitor['top'],
            grab_monitor['width']+grab_monitor['left'], grab_monitor['height']+grab_monitor['top']
        ))

    elif GRAB_SCREEN_METHOD == "dxcamcapture":
        img = dxcamera.get_latest_frame()

    # no new image, skip
    if img is None:
        continue

    # get some img data
    img_height, img_width, img_channels = img.shape

    if CV2_RESIZE_TO is not None:
        img = cv2.resize(img, CV2_RESIZE_TO)

    #global device, model, half, stride, imgsz, classify,\
    #    names, colors, conf_thres, iou_thres, classes, agnostic_nms,\
    #    augment

    img0 = img # preserve for plotting & displaying
    if img_channels > 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # leave 3 channels
    
    # reshape for PyTorch (samples, channels, height, width)
    img = numpy.moveaxis(img, -1, 0)

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
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)

                if team == "ct":
                    # t enemies
                    if names[int(cls)] in t_classes:
                        e_list.append((names[int(cls)], conf, ((int(xyxy[0]), int(xyxy[1])),(int(xyxy[2]), int(xyxy[3])))))
                else:
                    # ct enemies
                    if names[int(cls)] in ct_classes:
                        e_list.append((names[int(cls)], conf, ((int(xyxy[0]), int(xyxy[1])),(int(xyxy[2]), int(xyxy[3])))))

            # Shoot
            for e in e_list:
                if(e[1] > shoot_conf[1] and e[0] == e_classes[1]): # shoot to head
                    # monitor = {"top": 200, "left": 1300, "width": 1312, "height": 736}
                    # screen (w, h)
                    x1 = e[2][0][0]
                    y1 = e[2][0][1]

                    x2 = e[2][1][0]
                    y2 = e[2][1][1]

                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2

                    # absolute_mid_x = monitor["left"] + mid_x
                    # absolute_mid_y = monitor["top"] + mid_y
                    Shoot(mid_x, mid_y)

                    print(f"Enemy sighted! {e[0]} with conf {e[1]}")

                    break # shoot only 1 enemy at once


    # Print time (inference + NMS)
    # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    # torch.cuda.empty_cache()

    # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    # Calculate FPS
    fps+=1
    TIME = time.time() - start_time
    if (TIME) >= display_time :
        real_fps = int(fps / (TIME)) + 3
        print("FPS: ", real_fps)
        fps = 0
        start_time = time.time()

    if SHOW_CV2_WINDOW:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img0, str(real_fps), (7, 40), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # imS = cv2.resize(img0, (854, 480))

        # Display the picture
        # cv2.imshow(title, cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
        cv2.imshow(title, img0)

        cv2.waitKey(1)