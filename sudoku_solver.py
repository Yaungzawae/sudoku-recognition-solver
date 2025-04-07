#is valid is going to check if the number is valid for that place,
import numpy as np

def is_valid(grid:list[list[int]],row: int, col: int, number: int):
    if number in grid[row]:#check if the number is in the same row
        return False
    for i in range(9):#check for the number in different row,same column
        if grid[i][col] == number:
            return False
    start = (row//3) * 3
    end = (col//3) * 3
    for i in range(start,start+3):#check in the same grid
        for j in range(end,end+3):
            if grid[i][j] == number:
                return False
    return True


def solve_sudoku(grid: list[list[int]],row: int, col: int):
    if row == 9:
        return True

    elif col == 9:
        return solve_sudoku(grid,row+1,0)

    elif grid[row][col] >= 1:
        return solve_sudoku(grid,row,col+1)
    else:
        for trial_nb in range(1,10):
            if is_valid(grid,row,col,trial_nb):
                grid[row][col] = trial_nb
                if solve_sudoku(grid,row,col+1):
                    return True
                grid[row][col] = 0
        return False


def solve_and_return_grid(grid_input):
    grid_array = np.array(grid_input).astype(int)
    print(grid_array)
    original = grid_array.copy()
    if solve_sudoku(grid_array, 0, 0):
        return grid_array, "OK"
    else:
        return original, "ERR"
