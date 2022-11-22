import cv2
import sys
import random
import numpy as np
from time import time
from sudoku import main as gen_sudoku
from sudoku import display_sudoku, gen_data
from solveSudoku import sudoku_solver, check_data

given = np.zeros((9,9))
remained_grid = np.ones((9,9))
# random.seed(3)
indices = []

def gen_index(num):
    if(len(indices) > 0):
        num = num % len(indices)

    i = 0
    while(i < num):
        x = random.randint(0, 8)
        y = random.randint(0, 8)

        if((x, y) not in indices):
            indices.append((x, y))
            i += 1
    
    return indices
    

def left_down(res):
    cells_removed = 0
    sudoku = res
    rc_bound = 0
    temp = 3
    bound = random.randint(22, 23)
    while(temp > 0):
        temp -= 1
        for i in range(9):
            for j in range(9):
                if(given[i][j] == 1):
                    break
                
                last_val = sudoku[i][j]
                sudoku[i][j] = 0
                gen_data(sudoku)
                total_sols = sudoku_solver("src/count_sudoku.lp", "count")
                given[i][j] = 1
                remained_grid[i][j] = 0

                row_bound = remained_grid.sum(axis=1)[i]
                col_bound = remained_grid.sum(axis=0)[j]
                # print("\nTotal sols are : {}\n".format(total_sols))
                if((total_sols > 1) or (remained_grid.sum()) < bound or (row_bound < rc_bound) or (col_bound < rc_bound)):
                    sudoku[i][j] = last_val
                    remained_grid[i][j] = 1

                cells_removed += 1
            # print(given.sum())
            
    
    display_sudoku(sudoku.tolist())
    # print("\n\nThe total number cells removed are : {}\n\n".format(cells_removed))



def least_clues(res):
    cells_removed = 25
    sudoku = res
    while(True):
        all_indices = gen_index(cells_removed)
        # print(indices)
        for index in all_indices:
            x, y = index
            sudoku[x][y] = 0
        
        # check for the number of model counts
        gen_data(sudoku)
        # display_sudoku(sudoku.tolist())

        total_sols = sudoku_solver("src/count_sudoku.lp", "count")
        # print("\n\nThe total number of solutions to the above sudoku are : {} and the cells removed are : {}\n\n".format(total_sols, cells_removed))

        cells_removed += 1

        if(total_sols > 1):
            break
    
    display_sudoku(sudoku.tolist())


def random_sudoku():
    grid = np.random.randint(0, 9, (9, 9))
    gen_data(grid)
    # print("Solving your Sudoku!!")
    check_data("src/invalidSudoku.lp")
    res = sudoku_solver("src/sudoku.lp", "solve")
    display_sudoku(res)
    print("\n\nGenerated Sudoku is \n")
    return res

if __name__ == '__main__':
    option = sys.argv[1]
    if(option == 'naive'):
        print("Solving using Naive approach!")
        least_clues(random_sudoku())
    else:
        print("Solving using Left->Right->Top->Down approach!")
        left_down(random_sudoku())
    