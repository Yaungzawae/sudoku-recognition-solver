#is valid is going to check if the number is valid for that place,
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
                grid[row][col] = -1
        return False
    
grid=[[9,-1,6,-1,7,-1,-1,-1,-1],
      [-1,-1,-1,2,-1,-1,-1,9,-1],
      [8,5,1,-1,-1,9,7,-1,-1],
      [5,6,-1,-1,2,-1,9,-1,3],
      [-1,-1,-1,-1,1,-1,6,8,-1],
      [-1,-1,7,6,-1,-1,2,-1,4],
      [-1,1,9,-1,-1,4,-1,3,8],
      [7,-1,4,5,-1,8,1,6,-1],
      [-1,8,5,-1,3,-1,4,7,-1]
]

solve_sudoku(grid,0,0)
print(grid)