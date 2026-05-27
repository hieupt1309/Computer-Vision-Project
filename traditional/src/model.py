import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

class TraditionalPipeline:
    """
    An Object-Oriented Machine Learning pipeline that:
    1. Scales features using StandardScaler to prevent magnitude bias.
    2. Performs dimensionality reduction using PCA.
    3. Classifies features using K-Nearest Neighbors (KNN) with Euclidean distance.
    """
    def __init__(self, n_components=0.95, n_neighbors=5):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean")
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the pipeline on training features and labels.
        """
        # 1. Scaling to prevent magnitude bias
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)
        
        # 3. KNN classification
        self.knn.fit(X_pca, y)
        self.classes_ = self.knn.classes_
        return self

    def predict(self, X):
        """
        Predict labels for the input features.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.knn.predict(X_pca)

    def predict_proba(self, X):
        """
        Predict class probabilities for the input features.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.knn.predict_proba(X_pca)

    def get_neighbors(self, X, n_neighbors=5):
        """
        Returns distances and indices of neighbors in the training set.
        Useful for retrieving Top-K retrieval matches.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.knn.kneighbors(X_pca, n_neighbors=n_neighbors)

    def save(self, path):
        """
        Serialize and save the pipeline to a file.
        """
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load the pipeline from a file.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Integration of manual test block for independent verification
    print("========================================")
    print("Testing TraditionalPipeline...")
    print("========================================")
    
    # Generate mock training and testing data
    np.random.seed(42)
    X_train = np.random.rand(20, 100)
    y_train = np.array([0, 1] * 10)
    
    # Initialize and fit model
    pipeline = TraditionalPipeline(n_components=5, n_neighbors=3)
    pipeline.fit(X_train, y_train)
    print(f"Fit completed successfully.")
    print(f"Number of PCA components selected: {pipeline.pca.n_components_}")
    
    # Run predictions
    X_test = np.random.rand(5, 100)
    preds = pipeline.predict(X_test)
    print(f"Test predictions: {preds}")
    
    # Retrieve neighbors
    dists, indices = pipeline.get_neighbors(X_test, n_neighbors=3)
    print(f"Nearest neighbors distances: {dists[0]}")
    print(f"Nearest neighbors indices: {indices[0]}")
    
    # Test serialization
    test_path = "models_test_pipeline.pkl"
    pipeline.save(test_path)
    print(f"Model saved successfully to {test_path}")
    
    loaded_pipeline = TraditionalPipeline.load(test_path)
    loaded_preds = loaded_pipeline.predict(X_test)
    print(f"Loaded model predictions: {loaded_preds}")
    
    # Clean up test file
    if os.path.exists(test_path):
        os.remove(test_path)
        print("Cleaned up temporary test model file.")
        
    assert np.array_equal(preds, loaded_preds), "Loaded model predictions do not match!"
    print("TraditionalPipeline test PASSED!")
    print("========================================")
