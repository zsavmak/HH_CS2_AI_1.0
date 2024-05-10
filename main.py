import logging
import multiprocessing

from configurator import config

from detector_yolov8 import Detector
from uutils.grabbers.mss import Grabber
import cv2

from uutils.win32 import WinHelper
from uutils.fps import FPS
from uutils.cv2 import round_to_multiple
from uutils.time import sleep
import keyboard

# logging
logging.basicConfig(level=logging.INFO & logging.DEBUG)

# read config
if not config:
    logging.error("Errors while parsing config file. Exiting.")
    exit(1)

# set selected detector as main
if multiprocessing.current_process().name == "MainProcess":
    logging.debug(f'Detector is set to {config["main"]["detector"]}')
config["detector"] = config[config["main"]["detector"]]


# config
AUTO_SHOOT = False  # rather apply auto shoot or press button to shoot
SHOOT_HOTKEY = 58  # 58 = CAPS-LOCK
CHANGE_TEAM_HOTKEY = "ctrl+t"
shoot_conf = (0.75, 0.8)  # minimum required conf for detection to shoot (body, head)

team = "ct"  # initial team
t_classes = ("t", "th")
ct_classes = ("c", "ch")

# vars
enemy_team = None
e_classes = None


# defs
def id_enemy_classes():
    global team, e_classes, enemy_team

    if team == "ct":
        e_classes = t_classes
        enemy_team = "t"
    else:
        e_classes = ct_classes
        enemy_team = "ct"


def change_team_hotkey_callback(triggered, hotkey):
    global team, enemy_team

    if team == "t":
        team = "ct"
    else:
        team = "t"

    id_enemy_classes()
    print(f"NEW TEAM {team}")
    print(f"ENEMY TEAM {enemy_team}")


# some preparations
id_enemy_classes()
keyboard.add_hotkey(CHANGE_TEAM_HOTKEY, change_team_hotkey_callback, args=('triggered', 'hotkey'))


def grab_process(q):
    logging.info("GRAB process started")
    grabber = Grabber()
    game_window_rect = list(WinHelper.GetWindowRect(config["grabber"]["window_title"], (8, 30, 16, 39)))  # cut the borders

    # assure that width & height of capture area is multiple of 32
    if not config["detector"]["resize_image_to_fit_multiply_of_32"] and (
            int(game_window_rect[2]) % 32 != 0 or int(game_window_rect[3]) % 32 != 0):
        print("Width and/or Height of capture area must be multiply of 32")
        print("Width is", int(game_window_rect[2]), ", closest multiple of 32 is",
              round_to_multiple(int(game_window_rect[2]), 32))
        print("Height is", int(game_window_rect[3]), ", closest multiple of 32 is",
              round_to_multiple(int(game_window_rect[3]), 32))

        game_window_rect[2] = round_to_multiple(int(game_window_rect[2]), 32)
        game_window_rect[3] = round_to_multiple(int(game_window_rect[3]), 32)
        print("Width & Height was updated accordingly")

    while True:
        img = grabber.get_image({"left": int(game_window_rect[0]), "top": int(game_window_rect[1]), "width": int(game_window_rect[2]), "height": int(game_window_rect[3])})

        if img is None:
            continue

        q.put_nowait(img)
        q.join()


def cv2_process(q):
    logging.info("CV2 process started")
    detector = Detector(['c', 'ch', 't', 'th'])

    fps = FPS()
    fps_font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        if not q.empty():
            img = q.get_nowait()
            q.task_done()

            # Preprocess (predict, paint boxes, etc)
            bbox = detector.detect(img)
            #print(bbox)


            #for bbox in bboxes:
            #    print(bbox.boxes.xyxy)
            #    print(bbox.boxes.cls)
            #exit()
            img = detector.paint_boxes(img, bbox, 0.5)

            # CV window stuff (fps, resize, etc)
            if config["cv2"]["show_window"]:
                if config["cv2"]["show_fps"]:
                    img = cv2.putText(img, f"{fps():.2f}", (20, 120), fps_font,
                                      1.7, (0, 255, 0), 7, cv2.LINE_AA)

                if config["cv2"]["resize_window"]:
                    img = cv2.resize(img, (config["cv2"]["window_width"], config["cv2"]["window_height"]))

                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(config["cv2"]["title"], img)
                cv2.waitKey(1)


if __name__ == "__main__":
    logging.info("Starting processes.")

    q = multiprocessing.JoinableQueue()
    p1 = multiprocessing.Process(target=grab_process, args=(q,))
    p2 = multiprocessing.Process(target=cv2_process, args=(q,))

    p1.start()
    p2.start()
