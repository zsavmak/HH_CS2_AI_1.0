import os, sys, pygame, random, math, logging
from pygame.locals import *
import numpy as np

sqrt3 = np.sqrt(3)
sqrt5 = np.sqrt(5)

width = 1280
height = 720
screen_color = (255, 255, 255)
line_color_from = (46,113,206)
line_color_to = (107,201,29)
clock = pygame.time.Clock()
pygame.font.init()
my_font = pygame.font.SysFont('PT Sans', 35)
my_font2 = pygame.font.SysFont('PT Sans', 75)

point_from = (1280/2, 720/2)
# point_to = (1180, 340)

point_to = (280, 640)


def restart_program():
    """Restarts the current program, with file objects and descriptors
       cleanup
    """

    try:
        p = psutil.Process(os.getpid())
        for handler in p.get_open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        logging.error(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)


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
    while (dist:=np.hypot(dest_x-start_x,dest_y-start_y)) >= 1:
        W_mag = min(W_0, dist)
        if dist >= D_0:
            W_x = W_x/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
            W_y = W_y/sqrt3 + (2*np.random.random()-1)*W_mag/sqrt5
        else:
            W_x /= sqrt3
            W_y /= sqrt3
            if M_0 < 3:
                M_0 = np.random.random()*3 + 3
            else:
                M_0 /= sqrt5
        v_x += W_x + G_0*(dest_x-start_x)/dist
        v_y += W_y + G_0*(dest_y-start_y)/dist
        v_mag = np.hypot(v_x, v_y)
        if v_mag > M_0:
            v_clip = M_0/2 + np.random.random()*M_0/2
            v_x = (v_x/v_mag) * v_clip
            v_y = (v_y/v_mag) * v_clip
        start_x += v_x
        start_y += v_y
        move_x = int(np.round(start_x))
        move_y = int(np.round(start_y))
        if current_x != move_x or current_y != move_y:
            #This should wait for the mouse polling interval
            move_mouse(current_x:=move_x,current_y:=move_y)
    return current_x,current_y


def hanging_line(point1, point2):
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b

    return (x, y)


def getPointOnCurve(x1, y1, x2, y2, n):
    """Returns the (x, y) tuple of the point that has progressed a proportion
    n along the curve defined by the two x, y coordinates.
    If the movement length for X is great than Y, then Y offset else X
    """

    x = ((x2 - x1) * n) + x1
    y = ((y2 - y1) * n) + y1

    return (x, y)


# PARAMETERS
lerp_progress = 0
linear_step = 0.1
lerp_step = 0.01 # aka Aiming Speed (0.01 slow, 0.05 medium, 0.1 fast)
jitter = ((-1, 1), (-1, 1))
jitter = ((-5, 5), (-5, 5))
jitter = ((-10, 10), (-10, 10))
jitter = ((-15, 15), (-15, 15))

jitter = ((-30, 30), (-5, 5))

windmouse_points = []

# CONFIG
MOVEMENT_TYPE = "windmouse" # linear, jitter, curve or windmouse


def gen_windmouse_points():
    global windmouse_points

    wind_mouse(point_from[0],point_from[1],point_to[0],point_to[1],
        G_0=6,
        W_0=4,
        M_0=15,
        D_0=12,
        move_mouse=lambda x,y: windmouse_points.append([x,y]))


def main():
    global lerp_progress, windmouse_points

    screen = pygame.display.set_mode((width,height))
    screen.fill(screen_color)

    # vars
    end_point = list(point_from)
    line_color = list(line_color_from)
    prev_point = list(point_from)

    _curve_points = hanging_line(point_from, point_to)
    curve_points = (_curve_points[0].tolist(), _curve_points[1].tolist())
    y = point_to[1]
    ysw = False

    # windmouse
    gen_windmouse_points()

    # some info
    print(f"MOVEMENT TYPE IS {MOVEMENT_TYPE}")

    # pre render labels
    point1_text = my_font.render('Point from', True, line_color_from)
    point2_text = my_font.render('Point to', True, line_color_to)

    if MOVEMENT_TYPE == "jitter":
        mtype_text = my_font2.render(f"{MOVEMENT_TYPE.upper()} ({jitter})", True, (0, 0, 0))
    else:
        mtype_text = my_font2.render(MOVEMENT_TYPE.upper(), True, (0, 0, 0))
        

    while True:
        dt = clock.tick(30) / 1000.0

        screen.blit(point1_text, (point_from[0] - 50, point_from[1] - 50))
        screen.blit(point2_text, (point_to[0] - 50, point_to[1] - 50))

        screen.blit(mtype_text, (100, 100))

        # animate movement
        if MOVEMENT_TYPE == "linear":
            # lerp
            if end_point[0] < point_to[0]:
                end_point[0] += point_to[0] * linear_step * dt

            if end_point[1] < point_to[1]:
                end_point[1] += point_to[1] * linear_step * dt

            # save previous point (for color gradient)
            if end_point[0] - prev_point[0] >= 1:
                prev_point[0] = end_point[0] - 1

        elif MOVEMENT_TYPE == "jitter":
            if end_point[0] < point_to[0] or end_point[1] < point_to[1]:
                x, y = getPointOnCurve(point_from[0], point_from[1], point_to[0], point_to[1], lerp_progress)

                if lerp_progress < .9:
                    x += random.randint(jitter[0][0], jitter[0][1])
                    y += random.randint(jitter[1][0], jitter[1][1])

                # save previous point (for color gradient)
                prev_point[0] = end_point[0]
                prev_point[1] = end_point[1]

                # update
                end_point[0] = x
                end_point[1] = y
                lerp_progress += lerp_step

        elif MOVEMENT_TYPE == "curve":
            if end_point[0] < point_to[0]:
                x = curve_points[0].pop(0)
                #y = curve_points[1].pop(0)

                if y < 365 and not ysw:
                    y += 0.5
                else:
                    ysw = True
                    y -= 0.5

                print(f"{x}, {y}")

                if math.isnan(x):
                    x = point_to[0]
                if math.isnan(y):
                    y = point_to[1]

                # save previous point (for color gradient)
                prev_point[0] = end_point[0]
                prev_point[1] = end_point[1]

                end_point[0] = x
                end_point[1] = y

        elif MOVEMENT_TYPE == "windmouse":
            if end_point[0] < point_to[0] or end_point[1] < point_to[1]:
                x, y = windmouse_points.pop(0)

                print(f"{x}, {y}")

                # save previous point (for color gradient)
                prev_point[0] = end_point[0]
                prev_point[1] = end_point[1]

                end_point[0] = x
                end_point[1] = y

            else:
                main()
                screen.fill(screen_color)
                screen.fill(screen_color)
                screen.fill(screen_color)
                screen.fill(screen_color)
                screen.fill(screen_color)
                pygame.display.flip()
                break
                #restart_program()
                #sys.exit()

                end_point = list(point_from)
                line_color = list(np.random.choice(range(256), size=3))
                prev_point = list(point_from)

                gen_windmouse_points()

                screen.fill(screen_color)
                pygame.display.flip()
                continue

        # gradient color
        if line_color[0] < line_color_to[0]:
            line_color[0] += line_color_to[0] * lerp_step * dt
        if line_color[1] < line_color_to[1]:
            line_color[1] += line_color_to[1] * lerp_step * dt
        if line_color[2] < line_color_to[2]:
            line_color[2] += line_color_to[2] * lerp_step * dt

        # paint
        pygame.draw.line(screen, line_color, prev_point, end_point, 5)

        # flip
        pygame.display.flip()
        for events in pygame.event.get():
            if events.type == QUIT:
                sys.exit(0)

main()