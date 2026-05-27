import os
import numpy as np
from src import config
from src.utils import FacePreprocessor, FeatureExtractor, load_dataset, plot_2d_space
from src.model import TraditionalPipeline

def main():
    print("========================================")
    print("Starting Traditional CV Pipeline Training")
    print("========================================")
    
    # 1. Initialize preprocessing and feature extraction components
    preprocessor = FacePreprocessor()
    extractor = FeatureExtractor()
    
    # 2. Load and preprocess training dataset (Gallery)
    if not os.path.exists(config.GALLERY_DIR):
        raise FileNotFoundError(f"Gallery directory not found at: {config.GALLERY_DIR}")
        
    X_train, y_train, label_names, image_paths = load_dataset(
        config.GALLERY_DIR,
        preprocessor,
        extractor,
        max_identities=config.MAX_IDENTITIES,
        max_images_per_identity=config.MAX_IMAGES_PER_IDENTITY
    )
    
    if len(X_train) == 0:
        raise ValueError("No training samples loaded. Please check the gallery directory.")
        
    print(f"\nLoaded {len(X_train)} training images across {len(label_names)} identities.")
    print(f"Feature vector dimensionality: {X_train.shape[1]}")
    
    # 3. Train Pipeline (StandardScaler -> PCA -> KNN)
    print("\nTraining the TraditionalPipeline (StandardScaler -> PCA -> KNN)...")
    
    n_samples = len(X_train)
    pca_comp = config.PCA_COMPONENTS
    
    # Safety adjustment for small datasets (like the testing mock dataset)
    if isinstance(pca_comp, int) and pca_comp >= n_samples:
        pca_comp = max(1, n_samples - 1)
        print(f"  [WARNING] Adjusting PCA components to {pca_comp} due to small sample size.")
        
    pipeline = TraditionalPipeline(
        n_components=pca_comp,
        n_neighbors=min(config.KNN_NEIGHBORS, n_samples)
    )
    pipeline.fit(X_train, y_train)
    
    # Store training metadata in the pipeline object for inference visualization
    pipeline.label_names = label_names
    pipeline.gallery_image_paths = image_paths
    pipeline.gallery_labels = y_train
    
    # Calculate training accuracy
    train_preds = pipeline.predict(X_train)
    train_acc = np.mean(train_preds == y_train)
    print(f"Training identification accuracy: {train_acc * 100:.2f}%")
    print(f"Explained variance ratio: {np.sum(pipeline.pca.explained_variance_ratio_) * 100:.2f}%")
    print(f"PCA components selected: {pipeline.pca.n_components_}")
    
    # 4. Save trained pipeline
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    pipeline.save(config.MODEL_PATH)
    print(f"Pipeline model successfully saved to: {config.MODEL_PATH}")
    
    # 5. Generate 2D projection visualization of the gallery features
    print("\nGenerating t-SNE 2D projection of gallery feature space...")
    tsne_save_path = os.path.join(config.RESULT_DIR, "tsne_gallery.png")
    
    # Scale features as done in the pipeline for representation of scaled classification space
    X_train_scaled = pipeline.scaler.transform(X_train)
    plot_2d_space(X_train_scaled, y_train, label_names, tsne_save_path)
    print(f"t-SNE visualization saved to: {tsne_save_path}")
    
    print("\n========================================")
    print("Training Pipeline execution completed!")
    print("========================================")

if __name__ == "__main__":
    main()
