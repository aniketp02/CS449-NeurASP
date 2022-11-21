# CS449-NeurASP

Course project for CS449

- To demonstrate that reasoning can help to identify perception mistakes by neural networks that violate semantic constraints, which in turn can make perception more robust.

## Team
- Aniket Pokle (20D070011)
- Vedang Gupta (200100166)

## Getting Started

- Setting up the environment
```
conda create -n myenv
pip3 install -r requirements.txt
conda activate myenv
```

- To train the digit recognition model

```
cd digit_detector
python3 model.py -train
```
The model is saved at assets/model.pth

- Solve Sudoku and it's Varients

```
python3 sudoku.py <image_path>
```

## Generating Unique Sudoku
Solves the sudoku in the image and and recursively removes digits from the solved sudoku till a non-unique sudoku is obtained
```
python3 generate_sudoku.py <image_path>
```

## Available varients of Sudoku

- **Anti-Knight Sudoku** :  No number repeats at a knight move
- **Sudoku-X** : No number repeats at the diagonals

## Reference
- https://www.ijcai.org/proceedings/2020/0243.pdf
