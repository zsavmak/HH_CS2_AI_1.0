from mouse import Mouse
import pygame, sys
import mouse_move
import keyboard
import numpy

mouse = Mouse()
dot_color = (0 , 0 , 255)
SCREEN_COLOR = (255, 255, 255)
is_painting = False

screen = pygame.display.set_mode( (1280, 720) )
screen.fill(SCREEN_COLOR)
clock = pygame.time.Clock()
prev_point = pygame.mouse.get_pos()

bg_image = pygame.image.load('iOvVLU1QBI.png')
screen.blit(bg_image, [0, 0]) 


def random_color():
	return list(numpy.random.choice(range(256), size=3))


def aim(triggered, hotkey):
	global dot_color, prev_point
	prev_point = pygame.mouse.get_pos()
	dot_color = random_color()
	mouse_move.main(1706, 709, True)


def paint_on(triggered, hotkey):
	global dot_color, prev_point, is_painting

	if is_painting:
		return

	is_painting = True
	prev_point = pygame.mouse.get_pos()
	dot_color = random_color()
	mouse.hold_mouse()


def paint_off(triggered, hotkey):
	global is_painting

	is_painting = False
	mouse.release_mouse()


# hotkeys
keyboard.add_hotkey(58, aim, args=('triggered', 'hotkey'))
keyboard.add_hotkey("ctrl", paint_on, args=('triggered', 'hotkey'))
keyboard.add_hotkey("shift", paint_off, args=('triggered', 'hotkey'))


def drawCircle( screen, x, y ):
  pygame.draw.circle( screen, dot_color, ( x, y ), 2 )


while True:
	clock.tick(240)

	( x, y ) = pygame.mouse.get_pos()
	click_state = pygame.mouse.get_pressed()

	if click_state[0] == True:
		# left mouse button is down
		# drawCircle( screen, x, y )
		pygame.draw.line(screen, dot_color, prev_point, (x, y), 5)
		prev_point = [x, y]

	pygame.display.flip()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.quit()