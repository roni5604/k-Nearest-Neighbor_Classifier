from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
from k_nearest_neighbor import KNearestNeighbor
from data_utils import load_CIFAR10

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Set the folder name
FOLDERNAME = 'cs231n/Assignments/assignment1/'
assert FOLDERNAME is not None, "[!] Enter the foldername"

# Append folder to sys.path
import sys

sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# Change directory and run the dataset script
% cd / content / drive / My\ Drive /$FOLDERNAME / CV7062610 /
!bash
get_datasets.sh
% cd / content

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
X_test = np.reshape(X_test, (X_test.shape[0], -1))

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
