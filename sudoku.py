import cv2
import sys
from time import time
import matplotlib.pyplot as plt
from extractSudoku import extract_sudoku
from extractNumber import extract_number
from solveSudoku import sudoku_solver


def output(a):
    sys.stdout.write(str(a))

def display_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            cell = sudoku[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')

            if j != 8:
                output('  ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("--------+----------+---------\n")

def show_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def gen_data(grid):
    f1 = open('data.lp', 'w')
    for i in range(len(grid[0])):
        for j in range(len(grid[1])):
            if(grid[i][j] == 0):
                continue
            else:
                line = 'given(' + str(i) + ',' + str(j) + ',' + str(grid[i][j]) + ').\n'
                f1.write(line)

    f1.close


def main(image_path):
    image = extract_sudoku(image_path)
    # show_image(image)
    grid = extract_number(image)
    print('Sudoku:')
    # print(len(grid))
    # print(grid)

    '''
    validate sudoku # yet to implement this part
    gen_data
    solveSudoky
    '''
    gen_data(grid)
    print("\nBefore Solving the Sudoku!")
    display_sudoku(grid.tolist())

    res = sudoku_solver()
    try:
        display_sudoku(res.tolist())
        print("\n Your Sudoku is Solved!")
    except:
        print("\nThere are no possible solutions to the given Sudoku\n")

    # solution = sudoku_solver(grid)
    # print('Solution:')
#    print(solution)  
    # display_sudoku(solution.tolist())
        
def convert_sec_to_hms(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%08d" % (hour, minutes, seconds) 

if __name__ == '__main__':
   image_path = sys.argv[1]
   main(image_path)