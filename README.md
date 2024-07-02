# k-Nearest Neighbor (kNN) Classifier with CIFAR-10

This repository contains the implementation of a k-Nearest Neighbor (kNN) classifier to classify images from the CIFAR-10 dataset. The steps include mounting Google Drive, downloading the dataset, training the classifier, evaluating it using cross-validation, and visualizing the results.

## Task Description

The goal of this task is to implement a k-Nearest Neighbor (kNN) classifier to classify images from the CIFAR-10 dataset. The classifier will be trained on a subset of the CIFAR-10 training data, and its performance will be evaluated on a subset of the CIFAR-10 test data. We will also use cross-validation to find the best value of the hyperparameter \(k\), which represents the number of nearest neighbors to consider for classification.

## Inline Questions

### Inline Question 1: Observations on Distance Matrix

**What causes the distinctly bright rows or columns in the distance matrix?**

Bright rows or columns in the distance matrix indicate that the distances between certain test examples and all training examples are significantly higher (or lower) than average. This can happen due to large differences in pixel intensities between those test examples and the training examples, often caused by different backgrounds or objects in the images.

### Inline Question 2: Accuracy Comparison

**What can be concluded from the accuracy results with different \(k\) values?**

The accuracy results show that with \(k=1\), the model may overfit the training data, leading to lower accuracy on the test data. As \(k\) increases, the model generalizes better, but if \(k\) is too high, the model may become too smooth and underfit the data. Cross-validation helps find the optimal \(k\) value that balances bias and variance.

### Inline Question 3: k-NN Characteristics

**Which of the following statements about k-NN are true in a classification setting, and for all \(k\)?**

1. The decision boundary of the k-NN classifier is linear. (False)
2. The training error of a 1-NN will always be lower than that of a 5-NN. (True)
3. The test error of a 1-NN will always be lower than that of a 5-NN. (False)
4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set. (True)

## Files Structure

- `main.py`: The main script that runs the entire workflow step by step.
- `cs231n/classifiers/k_nearest_neighbor.py`: The kNN classifier implementation.
- `cs231n/data_utils.py`: Utility functions to load the CIFAR-10 dataset.
- `setup_drive_and_data.py`: Script to mount Google Drive and download the CIFAR-10 dataset.

## Setup and Running the Code

### Step 1: Mount Google Drive and Download Dataset

First, you need to mount your Google Drive and download the CIFAR-10 dataset.

1. Open `setup_drive_and_data.py` and run it.

**setup_drive_and_data.py**

```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Set the folder name
FOLDERNAME = 'cs231n/Assignments/assignment1/'
assert FOLDERNAME is not None, "[!] Enter the foldername"

# Append folder to sys.path
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# Change directory and run the dataset script
%cd /content/drive/My\ Drive/$FOLDERNAME/CV7062610/
!bash get_datasets.sh
%cd /content
```

### Step 2: Implement the kNN Classifier

Open `cs231n/classifiers/k_nearest_neighbor.py` and implement the kNN classifier. The file should contain:

**cs231n/classifiers/k_nearest_neighbor.py**

```python
import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For kNN this is just memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
        - y: A numpy array of shape (num_train,) containing the training labels
        """
        self.X_train = X
        self.y_train = y

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.

        Returns:
        - y_pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y_pred[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            y_pred[i] = np.bincount(closest_y).argmax()
        return y_pred
```

### Step 3: Load CIFAR-10 Data

Ensure the `cs231n/data_utils.py` file contains the following utility functions to load the CIFAR-10 dataset:

**cs231n/data_utils.py**

```python
import pickle
import numpy as np
import os

def load_CIFAR_batch(filename):
    """ Load single batch of CIFAR-10 """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ Load all of CIFAR-10 """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
```

### Step 4: Train and Evaluate the Classifier

Run `main.py` to execute the entire process step by step.

**main.py**

```python
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
from cs231n.data_utils import load_CIFAR10

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Set the folder name
FOLDERNAME = 'cs231n/Assignments/assignment1/'
assert FOLDERNAME is not None, "[!] Enter the foldername"

# Append folder to sys.path
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# Change directory and run the dataset script
%cd /content/drive/My\ Drive/$FOLDERNAME/CV7062610/
!bash get_datasets.sh
%cd /content

# Load the CIFAR-10 dataset
cifar10_dir = '/content/drive/My Drive/cs231n/Assignments/assignment1/CV7062610/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape

(X_test, (X_test.shape[0], -1))

# Create a kNN classifier instance and train it
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Compute the distance matrix
dists = classifier.compute_distances_two_loops(X_test)

# Predict labels for the test data using k=1 and k=5
y_test_pred_1 = classifier.predict_labels(dists, k=1)
y_test_pred_5 = classifier.predict_labels(dists, k=5)

# Compute and print the accuracy
accuracy_1 = np.mean(y_test_pred_1 == y_test)
accuracy_5 = np.mean(y_test_pred_5 == y_test)
print(f'Accuracy with k=1: {accuracy_1}')
print(f'Accuracy with k=5: {accuracy_5}')

# Cross-validation to find the best k
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for current_k in k_choices:
    k_to_accuracies[current_k] = []
    
    for validation_fold in range(num_folds):
        classifier = KNearestNeighbor()
        
        X_train_cv = np.concatenate(X_train_folds[:validation_fold] + X_train_folds[validation_fold + 1:])
        y_train_cv = np.concatenate(y_train_folds[:validation_fold] + y_train_folds[validation_fold + 1:])
        X_val_cv = X_train_folds[validation_fold]
        y_val_cv = y_train_folds[validation_fold]

        classifier.train(X_train_cv, y_train_cv)
        y_pred_fold = classifier.predict_labels(classifier.compute_distances_two_loops(X_val_cv), k=current_k)
        accuracy = np.mean(y_pred_fold == y_val_cv)
        k_to_accuracies[current_k].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print(f'k = {k}, accuracy = {accuracy}')

# Plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# Plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on cross-validation, choose the best k and test it on the test data
best_k = k_choices[np.argmax(accuracies_mean)]

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict_labels(classifier.compute_distances_two_loops(X_test), k=best_k)

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print(f'Got {num_correct} / {num_test} correct => accuracy: {accuracy}')
```

### What to Expect When Running the Code

1. **Mount Google Drive and Download Dataset**: The script will mount your Google Drive and download the CIFAR-10 dataset if not already present.
2. **Training and Evaluating the Classifier**: The script will load the CIFAR-10 data, train the kNN classifier, and compute the accuracy for different values of \(k\).
3. **Cross-Validation**: The script will perform cross-validation to find the best \(k\) value and plot the accuracy results.
4. **Final Evaluation**: The script will evaluate the classifier on the test data using the best \(k\) value found during cross-validation and print the final accuracy.

Certainly! Here is the continuation and completion of the README file:

---

By following these steps and explanations, you should have a complete understanding of the kNN classifier implementation and how to run the provided code.

### Running the Code Step-by-Step

1. **Mount Google Drive and Download Dataset**
    - Open `setup_drive_and_data.py` and run it in Google Colab.
    - This script will mount your Google Drive and download the CIFAR-10 dataset to the specified directory.

2. **Implement the kNN Classifier**
    - Ensure `cs231n/classifiers/k_nearest_neighbor.py` contains the correct implementation of the kNN classifier.
    - This file should define the `KNearestNeighbor` class with methods to train the classifier, compute distances, and predict labels.

3. **Load CIFAR-10 Data**
    - Ensure `cs231n/data_utils.py` contains utility functions to load the CIFAR-10 dataset.
    - These functions will be used to load and preprocess the CIFAR-10 data in the main script.

4. **Train and Evaluate the Classifier**
    - Run `main.py` to execute the entire process step by step.
    - This script will:
        1. Mount Google Drive and download the CIFAR-10 dataset.
        2. Load the CIFAR-10 data.
        3. Train the kNN classifier.
        4. Compute distances and predict labels for the test data using \(k=1\) and \(k=5\).
        5. Evaluate and print the accuracy for \(k=1\) and \(k=5\).
        6. Perform cross-validation to find the best \(k\) value.
        7. Plot the accuracy results of cross-validation.
        8. Evaluate the classifier on the test data using the best \(k\) value and print the final accuracy.

### What to Expect in Each Part

1. **Mounting Google Drive and Downloading Dataset**
    - The script will output a message indicating that Google Drive has been mounted.
    - It will change directory to the specified folder and run `get_datasets.sh` to download the CIFAR-10 dataset.

2. **Loading CIFAR-10 Data**
    - The script will load the CIFAR-10 data using `load_CIFAR10` function.
    - It will print the shapes of the training and test data arrays to confirm successful loading.

3. **Training and Evaluating the Classifier**
    - The script will train the kNN classifier on the training data.
    - It will compute distances between test examples and training examples using a nested loop.
    - It will predict labels for the test data using \(k=1\) and \(k=5\).
    - It will print the accuracy of the classifier for \(k=1\) and \(k=5\).

4. **Cross-Validation to Find the Best \(k\)**
    - The script will perform cross-validation with 5 folds and a range of \(k\) values.
    - It will print the accuracy for each \(k\) value and each fold.
    - It will plot the accuracy results with error bars corresponding to standard deviation.
    - The plot will help visualize the performance of the classifier for different \(k\) values.

5. **Final Evaluation**
    - The script will select the best \(k\) value based on cross-validation results.
    - It will train the classifier using the entire training data and evaluate it on the test data using the best \(k\).
    - It will print the final accuracy of the classifier on the test data.

### Conclusion

By following the steps outlined in this README, you will be able to:
- Mount Google Drive and download the CIFAR-10 dataset.
- Implement and understand the kNN classifier.
- Load and preprocess CIFAR-10 data.
- Train and evaluate the kNN classifier.
- Perform cross-validation to find the optimal \(k\) value.
- Visualize and interpret the results of the classifier.

This project will give you a solid understanding of the kNN algorithm and its application in image classification using the CIFAR-10 dataset.

### References

- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [k-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Cross-Validation in Machine Learning](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### Author

Roni Michaeli
