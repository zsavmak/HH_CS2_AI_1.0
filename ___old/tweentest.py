import win32api, win32con, win32gui
import keyboard
from time import sleep
import random


def getPointOnCurve(x1, y1, x2, y2, n):
    """Returns the (x, y) tuple of the point that has progressed a proportion
    n along the curve defined by the two x, y coordinates.
    If the movement length for X is great than Y, then Y offset else X
    """

    x = ((x2 - x1) * n) + x1
    y = ((y2 - y1) * n) + y1

    return (x, y)


flags, hcursor, (cx, cy) = win32gui.GetCursorInfo()
xy_from = (cx, cy)
rel_move = (100, 0)
final_point = (xy_from[0]+rel_move[0], xy_from[1]+rel_move[1])
lerp_progress = 0
lerp_step = 0.05
use_jitter = True
jitter = ((-1, 1), (-1, 1))

print(f"From {cx}, {cy}")
print(f"To {cx+rel_move[0]}, {cx+rel_move[1]}")

while lerp_progress < 1:
	x, y = getPointOnCurve(0, 0, rel_move[0], rel_move[1], lerp_step)

	if use_jitter and lerp_progress < .9:
		x += random.randint(jitter[0][0], jitter[0][1])
		y += random.randint(jitter[1][0], jitter[1][1])

	win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
	lerp_progress += lerp_step
else:
	flags, hcursor, (cx, cy) = win32gui.GetCursorInfo()
	target_rel = (final_point[0] - cx, final_point[1] - cy)
	win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(target_rel[0]), int(target_rel[1]), 0, 0)


flags, hcursor, (cx, cy) = win32gui.GetCursorInfo()
print(f"Final point {cx}, {cy}")