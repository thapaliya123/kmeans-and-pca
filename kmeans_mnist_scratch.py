"""
1.Implement k-means clustering on mnist dataset from scratch (use appripriate library for 
data structure (not for algorithms))
"""
# Import
import sys
import sklearn
import matplotlib
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.datasets import mnist

print('Python: {}'.format(sys.version))
print('Sklearn: {}'.format(sklearn.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('NumPy: {}'.format(np.__version__))

def load_mnist():
    """
    Loads mnist dataset using keras library and visualizes 9 training images
    """
    # Load mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('Training Data: {}'.format(x_train.shape))
    print('Training Labels: {}'.format(y_train.shape))
    print('Testing Data: {}'.format(x_test.shape))
    print('Testing Labels: {}'.format(y_test.shape))

    return x_train, y_train, x_test, y_test
    
def visualize_mnist(X, Y):
    """
    Visualizes mnist datasets

    Arguments:
    X: ndarray containing mnist data, eg. X_train
    Y: 1D array containing labels for X, eg. Y_train
    """
    # Visualize train_mnist
    # create figure with 3x3 subplots using matplotlib.pyplot
    fig, axs = plt.subplots(3, 3, figsize = (12, 12))
    plt.gray()

    # loop through subplots and add mnist images
    for i, ax in enumerate(axs.flat):
        ax.matshow(X[i])
        ax.axis('off')
        ax.set_title('Number {}'.format(Y[i]))
        
    # display the figure
    plt.show()

def preprocess_image(X):
    """
    Reshape input ndarray(mnist data) to 2D array and normlaizes all the data to be in range
    [0-255]
    """
    # Preprocessing the image
    # Convert each images to one-dimensional
    X = X.reshape(X.shape[0], -1)

    # Normalize the data, each data in range[0-1]
    X = X.astype(float)/255.

    print(f"X shape: {X.shape}") # each row represents different digits
    
    return X
# Kmeans implementation
def initialize_centroids(X, k):
    """
    Randomly initializes centroids from the given datasets

    Arguments:
    X: datasets to be clustered
    k: number of clusters

    Returns: 
    init_centroids: initialzed centroids
    """
    # Init centroid index
    init_centroids_index = random.sample(range(0, len(X)), k)

    # Use init_centroids as indices and get the points of these indices
    init_centroids = [X[i] for i in init_centroids_index]
    init_centroids = np.array(init_centroids)
    
    return init_centroids

def calc_distance(x1, x2):
    """
    Calculate Euclidean distances between two numpy as using vectorization.
    
    Arguments:
    x1: point 1, numpy array
    x2: point 2, numpy array
    
    Returns:
    Euclidean distance between x1 and x2.
    """
    return np.sum((x1-x2)**2, 1)**0.5

def find_closest_centroids(X, centroids):
    """
    Returns index of the closest centroids for given points
    
    Arguments:
    X: datasets, 1D or muti-dimensional numpy array
    centroids: cluster centroid, 1D or multi-dimensional numpy array
    
    Returns:
    closest_centroids: 1D array indicating closest centroid index for each data point
    """
    points_centroids_distances = [calc_distance(X, centroid) for centroid in centroids]
    points_centroids_distances = np.column_stack(points_centroids_distances)
    closest_centroids = np.argmin(points_centroids_distances, 1)
    return closest_centroids

def calc_centroids(X, clusters, number_of_cluster):
    """
    Calculates new centroids for different clusters by averaging datapoints assigned to different clusters.
    
    Arguments:
    X: datapoints
    clusters: Cluster label based on centroid index for each datapoints
    number_of_centroids: number of cluster i.e. k
    """
    new_centroids = [np.mean(X[clusters == cluster_index], 0) for cluster_index in range(number_of_cluster)]
    return new_centroids

def main():
    K = 10

    # Load mnist datasets
    x_train, y_train, x_test, y_test = load_mnist()
    X = x_train
    Y = y_train

    # Preprocess mnist
    X = preprocess_image(X)
    # Initialize centroids
    init_centroids = initialize_centroids(X, K)
    centroids = init_centroids

    for i in range(100):
        prev_centroids = centroids
        closest_centroids = find_closest_centroids(X, centroids)
        centroids = calc_centroids(X, closest_centroids, K)
        centroids = np.array(centroids)
        
        # Exit from loop if centroids are same for two iteration
        print(f"Iteration:{i}")
        if ((prev_centroids == centroids).all()):
            print(f"Centroids are same for two consecutive iteration so exiting.....")
            break
    print(f"final centroids shape: {centroids.shape}")
    print(f"1st centroid:\n {centroids[0]}")

if __name__ == "__main__":
    main()