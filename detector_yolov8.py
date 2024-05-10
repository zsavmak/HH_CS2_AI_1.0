from ultralytics import YOLO
import numpy
import cv2
import torch
import random

from configurator import config


class Detector:
    # yolov7 params
    weights = config["yolov8"]["weights"]  # model.pt path(s)
    names = []

    def __init__(self, names):

        # set names
        self.names = names

        # generate random colors
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


        # load model
        self.model = model = YOLO(config["yolov8"]["weights"])

        # set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def detect(self, img: numpy.ndarray):
        """
        Make an inference (prediction) of a given image.
        Image size must be multiple of 32 (ex. 640).
        And the channels count must be 3 (RGB).
        :param img: Input image, must be numpy.ndarray with shape of (samples, channels, height, width).
        :return: Detected bounding boxes in a dict format, where's the key is a class.
        """

        # get some img data
        img_height, img_width, img_channels = img.shape

        if img_channels > 3:
            return False
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # leave 3 channels

        # reshape for PyTorch (samples, channels, height, width)
        # img = numpy.moveaxis(img, -1, 0)

        result = {}
        presults = self.model.predict(source = img)

        for presult in presults:
            for idx, cls in enumerate(presult.boxes.cls):
                result.setdefault(self.names[int(cls)], [])

                result[self.names[int(cls)]].append({
                    "cls": int(cls),
                    "conf": presult.boxes.conf[idx].item(),
                    "xyxy": presult.boxes[idx].xyxy.cpu().numpy()[0].tolist()
                })

        return result

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=3):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def paint_boxes(self, img: numpy.ndarray, bbox_list: dict, min_conf: float) -> numpy.ndarray:
        """
        Paint predicted bounding boxes to a given image.
        :param img: Input image.
        :param bbox_list: Detected bounding boxes (expected output from detect method).
        :return:
        """
        for i, (c, ds) in enumerate(bbox_list.items()):
            for d in ds:
                if float(d['conf']) > min_conf:
                    self.plot_one_box(d["xyxy"], img, label=f"{c} {d['conf']:.2f}", color=self.colors[int(d["cls"])], line_thickness=2)

        return img
