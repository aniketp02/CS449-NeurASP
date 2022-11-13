import clingo
import numpy as np

ctr=0
res = np.zeros((9, 9), dtype='int32')

def solve(m):

    global ctr
    ctr += 1
    print("Answer number: ", ctr, " is printed below")
    
    atoms=m.symbols(atoms=True)
    # print("Atoms are: ", atoms)
    for i, atom in enumerate(atoms):
        # print("Processing atom with name: ", atom.name)
        if(atom.name == 'filled'):
            x = atom.arguments[0].number
            y = atom.arguments[1].number
            res[x][y] = atom.arguments[2].number
    
    print(res)
    cont= input("Enter y to generate more answers -> ")
    if cont != 'y':
        return False


def validate(ans):
    print("\nValidating the detected Sudoku!\n")

    f1 = open('data.lp', 'a')
    atoms=ans.symbols(atoms=True)
    # print("Atoms are: ", atoms)
    for i, atom in enumerate(atoms):
        # print("Processing atom with name: ", atom.name)
        if(atom.name == 'invalid'):
            x = atom.arguments[0].number
            y = atom.arguments[1].number
            val = atom.arguments[2].number
            line = 'invalid(' + str(x) + ',' + str(y) + ',' + str(val) + ').\n'
            f1.write(line)
    f1.close()
    return False

def check_data(file):
    ctl1 = clingo.Control("0")
    ctl1.load("data.lp")
    ctl1.load(file)
    ctl1.configuration.solve.models = 0
    ctl1.ground([("base", [])])

    with ctl1.solve(on_model=lambda m: validate(m), async_=True) as handle1:
        while not handle1.wait(0):
            handle1.get()


def sudoku_solver(solver):
    ctl = clingo.Control("0")
    ctl.load("data.lp")
    ctl.load(solver)
    ctl.configuration.solve.models = 0
    ctl.ground([("base", [])])

    with ctl.solve(on_model=lambda m: solve(m), async_=True) as handle:
        while not handle.wait(0):
            handle.get()
    
    print("\nReturning the data of the solved sudoku!\n")
    return res


# data = sudoku_solver()
# print("is this working")
# print(data)