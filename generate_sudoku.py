import cv2
import sys
import random
from time import time
from sudoku import main as gen_sudoku
from sudoku import display_sudoku, gen_data
from solveSudoku import sudoku_solver

random.seed(3)
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
    

def least_clues(image_path):
    cells_removed = 25
    sudoku = gen_sudoku(image_path)
    while(True):
        all_indices = gen_index(cells_removed)
        # print(indices)
        for index in all_indices:
            x, y = index
            sudoku[x][y] = 0
        
        # check for the number of model counts
        gen_data(sudoku)
        display_sudoku(sudoku.tolist())

        total_sols = sudoku_solver("src/count_sudoku.lp", "count")
        print("\n\nThe total number of solutions to the above sudoku are : {} and the cells removed are : {}\n\n".format(total_sols, cells_removed))

        cells_removed += 1

        if(total_sols > 1):
            break


if __name__ == '__main__':
   image_path = sys.argv[1]
   least_clues(image_path)