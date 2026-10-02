# MNIST Digit Recognition: Perceptron vs ANN vs CNN

Handwritten digit classification on the **MNIST** dataset, comparing three models built with TensorFlow/Keras:

1. **Perceptron-style model**: single softmax layer (linear baseline)
2. **ANN**: fully connected network (128 -> 64 -> 10)
3. **CNN**: two Conv2D + MaxPooling blocks, Dense(128), Dropout(0.5)

## Dataset
[MNIST in CSV format](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv): 60,000 training and 10,000 test images (28x28 grayscale).
Each row contains a `label` and 784 pixel values.

> The CSV files are too large for GitHub (over 100 MB), so they are **not included**. Download `mnist_train.csv` and `mnist_test.csv` from the link above and place them next to the notebook.

## Workflow
1. Load and inspect data
2. Normalise pixels (0-1), reshape, one-hot encode labels
3. Train the three models (5 epochs, batch size 32)
4. Compare learning curves, side-by-side predictions, CNN confusion matrix and final accuracy

## Results
| Model | Test accuracy |
|---|---|
| Perceptron (softmax) | 90.80% |
| ANN | 97.74% |
| CNN | 99.14% |



## Key takeaways
- Adding hidden layers lifts accuracy substantially over the linear baseline.
- CNNs exploit the spatial structure of images and give the best accuracy.
- Possible improvements: more epochs with early stopping, data augmentation, a separate validation split.

## Files
| File | Description |
|---|---|
| `mnist_perceptron_ann_cnn.ipynb` | Runnable notebook with explanations |
| `requirements.txt` | Python dependencies |

## Getting started
```bash
git clone https://github.com/RimeeAwasthi01/mnist-perceptron-ann-cnn.git
cd mnist-perceptron-ann-cnn
pip install -r requirements.txt
jupyter notebook mnist_perceptron_ann_cnn.ipynb
```

## Tech stack
Python, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras

## Author
Rimee Awasthi
