import torch
import os
import cv2
import numpy as np
import torchvision.transforms as transforms

import digit_detector.cnn_model as cnn_model


model = cnn_model.Net()

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))
        # The nos. are global mean and global std for MNIST dataset
	])

if os.path.exists('digit_detector/assets/model.pth'):
    model.load_state_dict(torch.load('digit_detector/assets/model.pth'))
    model.eval()
else:
    raise Exception('Model not trained')


def identify_number(image):
    prediction = 0
    if not model:
        return prediction

    # pred_img = cv2.imread(image, 0)
    # pred_img = image / 255

    pred_img = cv2.resize(image, (28, 28))
    pred_img_tr = transform(pred_img)
    pred_img_tr = torch.reshape(pred_img_tr, shape=(1, 28, 28))

    # Predict the digit value using the model
    prediction_arr = model(pred_img_tr).cpu().detach().numpy()
    prediction = np.argmax(prediction_arr)

    return prediction


def extract_number(sudoku):
    sudoku = cv2.resize(sudoku, (450,450))

    # split sudoku
    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
#            image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
            image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
#            filename = "images/sudoku/file_%d_%d.jpg"%(i, j)
#            cv2.imwrite(filename, image)
            # print(image.sum())
            if image.sum() > 80000:
                grid[i][j] = identify_number(image)
            else:
                grid[i][j] = 0
    return grid.astype(int)