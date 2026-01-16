import pygame
import numpy as np
import cv2
import csv

WINDOW_SIZE = 280
GRID_SIZE = 28 
BRUSH_SIZE = 1 

def save_image(surface):
    # 1. Get pixel data
    pixels = pygame.surfarray.array3d(surface)
    
    pixels = np.rot90(pixels, -1)
    pixels = np.fliplr(pixels)
    
    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    
    small_image = cv2.resize(pixels, (28, 28), interpolation=cv2.INTER_AREA)
    
    
    flat_data = small_image.flatten().astype(int) 
    
    with open("my_drawing.csv", "w", newline='') as f:
        writer = csv.writer(f)
        
        header = ["label"] + ["pixel" + str(i) for i in range(784)]
        writer.writerow(header)
        
        row = ['?'] + list(flat_data)
        writer.writerow(row)
        
    print("Drawing Saved")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Draw a Digit (Press S to Save, C to Clear)")
    
    canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    canvas.fill((0, 0, 0))
    
    drawing = False
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_image(canvas)
                elif event.key == pygame.K_c:
                    canvas.fill((0, 0, 0))

        if drawing:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(canvas, (255, 255, 255), mouse_pos, 15)

        screen.blit(canvas, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()