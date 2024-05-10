from mouse import Mouse
import numpy
from utils.torch_utils import time_synchronized
import time, sys
import pydirectinput as pyautogui
import win32gui, win32api, win32con, ctypes


mouse = Mouse()
monitor = {"top": 188, "left": 1278, "width": 1280, "height": 720}
screen = (3440, 1440)
sx = 2.2
sy = 5.2

sqrt3 = numpy.sqrt(3)
sqrt5 = numpy.sqrt(5)

mid_x = 777 + monitor['left']
mid_y = 482 + monitor['top']

rx = 95 * sx
ry = 100 * sy

rx = 290 * sx
ry = -(310 * sy)

print(ry)

win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(rx), int(ry), 0, 0)
sys.exit()

# x = int(mid_x*monitor['width'])
#y = int(mid_y*height)
# y = int(mid_y*monitor['height']+monitor['height']/9)

x = mid_x / s
y = mid_y / s


pyautogui.moveTo(int(x), int(y))
sys.exit()



x = (tx - cx) * s
y = (ty - cy) * s

print(f"{x}")

d = 4
mouse.move_mouse_rel( (int(x), int(y)) )
sys.exit()


def wind_mouse(start_x, start_y, dest_x, dest_y, G_0=9, W_0=3, M_0=15, D_0=12, move_mouse=lambda x,y: None):
    '''
    WindMouse algorithm. Calls the move_mouse kwarg with each new step.
    Released under the terms of the GPLv3 license.
    G_0 - magnitude of the gravitational fornce
    W_0 - magnitude of the wind force fluctuations
    M_0 - maximum step size (velocity clip threshold)
    D_0 - distance where wind behavior changes from random to damped
    '''
    current_x,current_y = start_x,start_y
    v_x = v_y = W_x = W_y = 0
    while (dist:=numpy.hypot(dest_x-start_x,dest_y-start_y)) >= 1:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/sqrt3 + (2*numpy.random.random()-1)*W_mag/sqrt5
            W_y = W_y/sqrt3 + (2*numpy.random.random()-1)*W_mag/sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = numpy.random.random()*3 + 3
            else:
                M_0 /= sqrt5
        v_x += W_x + G_0*(dest_x-start_x)/dist
        v_y += W_y + G_0*(dest_y-start_y)/dist
        v_mag = numpy.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0/2 + numpy.random.random()*M_0/2
            v_x = (v_x/v_mag) * v_clip
            v_y = (v_y/v_mag) * v_clip
        start_x += v_x
        start_y += v_y
        move_x = int(numpy.round(start_x))
        move_y = int(numpy.round(start_y))
        if current_x != move_x or current_y != move_y:
            #This should wait for the mouse polling interval
            move_mouse(current_x:=move_x,current_y:=move_y)
    return current_x,current_y


def main(target_x, target_y, hold_mouse = False):
	# get current cursor pos & log
	(cx, cy) = mouse.get_position()
	print(f"Current cursor pos: {cx},{cy}")

	# log target
	print(f"Target cursor pos: {target_x},{target_y}")

	# points
	windmouse_points = []
	t1 = time_synchronized()
	wind_mouse(cx,cy,target_x,target_y,
	    G_0=6,
	    W_0=6,
	    M_0=1, # step size
	    D_0=12,
	    move_mouse=lambda x,y: windmouse_points.append([x,y]))
	t2 = time_synchronized()
	print(f"{int(1E3 * (t2 - t1))}ms WIND MOUSE")
	# print(windmouse_points)

	if hold_mouse:
		mouse.hold_mouse()

	t1 = time_synchronized()
	for p in windmouse_points:

		x = (p[0] - cx)
		y = p[1] - cy

		print(f"Move to {x}, {y}")

		# x = int(x_move*monitor['width'])
		#y = int(mid_y*height)
		# y = int(y_move*monitor['height']+monitor['height']/9)

		mouse.move_mouse_rel((x, y))

	if hold_mouse:
		time.sleep(0.05)
		mouse.release_mouse()

	t2 = time_synchronized()
	print(f"{int(1E3 * (t2 - t1))}ms MOUSE MOVE")


if __name__ == "__main__":
	(cx, cy) = mouse.get_position()

	x = cx + 50
	y = cy

	main(x, y)