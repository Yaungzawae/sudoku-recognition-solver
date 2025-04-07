import sys
import pygame
import sys
import numpy as np
from PIL import Image
from processing import process_image
from sudoku_solver import solve_and_return_grid
img_path = sys.argv[1]

# Pygame setup
pygame.init()

# Constants for the grid and window
GRID_SIZE = 9
CELL_SIZE = 50
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
FONT = pygame.font.Font(None, 36)

# Create a Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sudoku Solver")

# Initialize with zeros
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# Function to draw the grid
def draw_grid():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pygame.draw.rect(screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
            if grid[row][col] != 0:  # This now checks properly
                text = FONT.render(str(grid[row][col]), True, BLACK)
                screen.blit(text, (col * CELL_SIZE + 15, row * CELL_SIZE + 10))

# Handle user input
def handle_input(event):
    global selected_cell
    if event.type == pygame.MOUSEBUTTONDOWN:
        x, y = event.pos
        row = y // CELL_SIZE
        col = x // CELL_SIZE
        selected_cell = (row, col)

    if event.type == pygame.KEYDOWN and selected_cell:
        row, col = selected_cell
        if event.key == pygame.K_0:
            grid[row][col] = 0
        elif event.key == pygame.K_1:
            grid[row][col] = 1
        elif event.key == pygame.K_2:
            grid[row][col] = 2
        elif event.key == pygame.K_3:
            grid[row][col] = 3
        elif event.key == pygame.K_4:
            grid[row][col] = 4
        elif event.key == pygame.K_5:
            grid[row][col] = 5
        elif event.key == pygame.K_6:
            grid[row][col] = 6
        elif event.key == pygame.K_7:
            grid[row][col] = 7
        elif event.key == pygame.K_8:
            grid[row][col] = 8
        elif event.key == pygame.K_9:
            grid[row][col] = 9

# Function to solve the Sudoku grid
def solve_sudoku():
    global grid
    grid, _ = solve_and_return_grid(grid)
    print(grid)

# Function to process the image and extract the Sudoku grid
def process_sudoku_image(img):
    return process_image(img)  # This would return a 9x9 grid from the image

# Initialize selected_cell for tracking the cell being edited
selected_cell = None

# Main game loop
img = Image.open(img_path)
img.show()
grid = process_sudoku_image(img_path)
while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            img.close()
            sys.exit()

        # Handle user input for editing cells
        handle_input(event)

    # Fill the screen with white background
    screen.fill(WHITE)

    # Draw the grid and values
    draw_grid()

    # Solve button click simulation (for testing purposes)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_RETURN]:  # Press 'Enter' to solve
        solve_sudoku()

    # Update the display
    pygame.display.flip()
