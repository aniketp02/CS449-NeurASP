import clingo
import numpy as np

ctr=0
res = np.zeros((9, 9), dtype='int32')

def myfn(ans):

    global ctr
    ctr += 1
    print("Answer number: ", ctr, " is printed below")
    
    atoms=ans.symbols(atoms=True)
    # print("Atoms are: ", atoms)
    for i, atom in enumerate(atoms):
        # print("Processing atom with name: ", atom.name)
        x = atom.arguments[0].number
        y = atom.arguments[1].number
        res[x][y] = atom.arguments[2].number
    
    # print(res)
    cont=int(input("Do you want one more answer? (0-stop, 1-yes) -> "))
    if cont == 0:
        return False


def sudoku_solver():
    ctl = clingo.Control("0")
    ctl.load("data.lp")
    ctl.load("sudoku_clingo.lp")
    ctl.configuration.solve.models = 0
    ctl.ground([("base", [])])

    with ctl.solve(on_model=lambda m: myfn(m), async_=True) as handle:
        while not handle.wait(0):
            handle.get()
    
    print("\nReturning the data of the solved sudoku!\n")
    return res


# data = sudoku_solver()
# print("is this working")
# print(data)