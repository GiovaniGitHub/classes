import numpy as np


class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


class PCA2:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        unitary_matrix, singular_values, eigen_vectors = np.linalg.svd(X, full_matrices=False)
        max_abs_cols = np.argmax(np.abs(unitary_matrix), axis=0)
        signs = np.sign(unitary_matrix[max_abs_cols, range(unitary_matrix.shape[1])])
        unitary_matrix *= signs
        eigen_vectors *= signs[:, np.newaxis]
        
        self.components = eigen_vectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
    
# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_wine()
    X, y = data.data, data.target

    lda = PCA(2)
    lda.fit(X)
    X_projected = lda.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1, x2 = X_projected[:, 0]*-1, X_projected[:, 1]*-1
    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Components 1")
    plt.ylabel("Principal Components 2")
    plt.colorbar()
    plt.show()