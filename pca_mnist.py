"""
Q. Implement PCA and apply it for dimensinionaly reduction of mnist dataset to 3 Dimension 
(and visualize the result)
"""
# Import
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from kmeans_mnist_scratch import load_mnist, preprocess_image

# Make covariance matrix
def generate_covariance_matrix(X):
    """
    generates covariance matrix for 2D numpy array passed as an argument without using library

    Arguments:
    X: 2D numpy array, shape: nxd

    Returns:
    cov_mat: covariance matrix for X, shape: dxd
    """

    # Covariance matrix without numpy
    mean_vec = np.mean(X, axis=0)
    cov_mat = (X-mean_vec).T.dot((X-mean_vec))/(X.shape[0]-1)

    return cov_mat

# Compute eigenvectors and eigenvalues 
def compute_eigenvectors_eigenvalues(square_matrix):
    """
    Computes eigenvectors and eigenvalues for square matrix passed as an arguments
    """

    # Compute Eigenvectors and Eigenvalues
    eig_vals, eig_vecs = np.linalg.eig(square_matrix) 
    return eig_vals, eig_vecs

# Sorting Eigenvectors and Eigen values in descending order
def sort_eigenvector_eigenvalue_desc(eig_vecs, eig_vals):
    """
    Return list of  (eigenvalue, eigenvector) tuples sorting eigenvalue desc
    """

    #Make a list of (eigenvalue, eigenvector) tuples 
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))] 
    
    #Sort the (eigenvalue, eigenvector) tuples from high to low 
    eig_pairs.sort(key=lambda x:x[0], reverse=True) 

    return eig_pairs

# Construct project matrix
def construct_project_using_three_eigenvector(eig_pairs):
    """
    Returns Projection matrix "proj_matrix_w" using top 3 Eigen vectors from eig_pairs
    """
    # Construct Projection matrix W using top 3 Eigen vectors from eig_pairs
    # numpy.hstack: Stack arrays in sequence horizontally (column wise)
    proj_matrix_w = np.hstack((eig_pairs[0][1].reshape(784, 1), eig_pairs[1][1].reshape(784, 1), eig_pairs[2][1].reshape(784, 1)))
    
    return proj_matrix_w

# Merge all the modules that represents PCA steps
def pca_steps(X, Y):
    """
    Reduced dimensions of given input image X using pca into 3 Dimensions and returns pandas 
    dataframe containing 3 principal components and labels Y as a columns

    Arguments:
    X: input image of shape: (m, x, y), generally mnist train datasets
    Y: labels of X

    Returns:
    principal_df: pandas dataframe containing 3 principal components and labels for X
                    Columns of dataframe is:
                    1. principal_component_1
                    2. principal_component_2
                    3. principal_component_3
                    4. labels
    """
    #1.Standardization
    X = preprocess_image(X)

    #2.Generate Covariance matrix
    cov_mat = generate_covariance_matrix(X)

    #3.Compute Eigenvectors and Eigenvalues
    eig_vals, eig_vecs = compute_eigenvectors_eigenvalues(cov_mat)
    print(f"Eigen values shape: {eig_vals.shape}")
    print(f"Eigen vectors shape: {eig_vecs.shape}")

    #3.Sort Eigenvalues in Descending order
    eig_pairs = sort_eigenvector_eigenvalue_desc(eig_vecs, eig_vals)

    #4.Construct project matrix taking top 3 Eigenvectors for reduction into 3D
    proj_matrix_w = construct_project_using_three_eigenvector(eig_pairs)
    print('Projection Matrix W Shape:\n', proj_matrix_w.shape)

    #5. Projection Onto the New Feature Space

    # project onto the New Feature Space
    # projection to span of top 3 Eigenvectors
    projected_matrix = X.dot(proj_matrix_w)
    # create dataframe 
    principal_df = pd.DataFrame(data=projected_matrix, columns = ['principal_component_1', 'principal_component_2', 'principal_component_3'])
    # add target class to the dataframe
    principal_df["labels"] = Y

    return principal_df

# Visualize 3D projection
def scatter_plot_3d(x, y, z, labels):
    """
    Plots scatter plot for (x, y, z) in 3 dimension with labels as hue
    """
    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

    # Initialize figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    sc = ax.scatter(x, y, z, c=labels, marker='o', cmap=cmap)

    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

    plt.show()

def main():
    # Load mnist data
    x_train, y_train, x_test, y_test = load_mnist()

    # Dimension reduction using pca
    principal_df = pca_steps(x_train, y_train)

    # Visualize 3D projection
    principal_component_1 = np.array(principal_df['principal_component_1'])
    principal_component_2 = np.array(principal_df['principal_component_2'])
    principal_component_3 = np.array(principal_df['principal_component_3'])
    scatter_plot_3d(principal_component_1, principal_component_2, principal_component_3, y_train)

if __name__ == "__main__":
    main()