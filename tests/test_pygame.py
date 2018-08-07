import pygame
import time
import random
#pygame.font.get_fonts() # Run it to get a list of all system fonts
display_w = 800
display_h = 600

BLUE = (128, 128, 255)
DARK_BLUE = (1, 50, 130)
RED = (255, 192, 192)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()
gameDisplay = pygame.display.set_mode((display_w,display_h))
pygame.display.set_caption('Neural Introspection')
clock = pygame.time.Clock()

im = pygame.image.load('image.jpg')
#im = pygame.transform.scale(im, (100, 160)) # SCALE IMAGE

def _im(x,y):
    gameDisplay.blit(im,(x,y))

def text_objects(text, font):
    textSurface = font.render(text, True, BLACK)
    return textSurface, textSurface.get_rect()

def screen_mssg_variable(variable):
    font = pygame.font.SysFont('arial', 20)
    text = font.render('Random= ' +str(variable), True, BLACK)
    gameDisplay.blit(text,(0,0))


def screen_mssg(text):
    #largetext = pygame.font.Font('TimesNewRoman.ttf',115)
    largetext = pygame.font.SysFont('arial', 115)
    # Get the text and the rectangular around it
    TextSurf, TextRect = text_objects(text, largetext)
    TextRect.center = ((display_w/2),(display_h/2))
    gameDisplay.blit(TextSurf, TextRect)
    pygame.display.update()
    time.sleep(4)

    #game_loop( ) # You put your gaming function below (you just need to define it from all the stuff below)

x = (0)#(display_w * 0.45)
y = (0)#(display_h * 0.8)

running = True
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        print(event)

    # Fill background color
    gameDisplay.fill(DARK_BLUE)
    # Draw image
    _im(x,y)
    screen_mssg('Go!')
    screen_mssg_variable(random.randint(1,5)) # It is run on a loop so it will change all the time
    pygame.display.update() # You need ONE UPDATE for all blits you make (no need to update for every blit)
    clock.tick(60)

pygame.quit()
quit()