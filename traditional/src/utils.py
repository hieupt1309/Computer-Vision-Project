import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Import configuration
from . import config

try:
    from skimage.feature import hog, local_binary_pattern
except ImportError:
    pass

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    try:
        from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops
    except ImportError:
        pass


class FacePreprocessor:
    """
    Performs face preprocessing:
    Color normalization (Gray/HSV) -> Face detection and cropping -> Histogram Equalization -> Resizing.
    """
    def __init__(self, target_size=None, cascade_path=None):
        self.target_size = target_size if target_size is not None else config.IMAGE_SIZE
        if cascade_path is None:
            # OpenCV default Haar Cascade filepath
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Could not load Haar Cascade classifier from {cascade_path}")

    def preprocess_image(self, bgr_image):
        """
        Processes a raw BGR image:
        1. Color space normalization to Grayscale and HSV.
        2. Face detection and cropping using Haar Cascades.
        3. Histogram equalization (on Grayscale face).
        4. Resizing to a uniform dimension.
        
        Returns:
            (gray_face, hsv_face, bounding_box) if face is found,
            None otherwise.
        """
        if bgr_image is None:
            return None
            
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # Fallback to full image frame if face detection fails
            h_img, w_img = gray.shape[:2]
            x, y, w, h = 0, 0, w_img, h_img
        else:
            # Select the largest face by bounding box area
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
        
        # Crop from both color spaces
        gray_face = gray[y:y+h, x:x+w]
        hsv_face = hsv[y:y+h, x:x+w]
        
        # Histogram equalization on grayscale face for illumination robustness
        gray_face_eq = cv2.equalizeHist(gray_face)
        
        # Resize to uniform dimension
        gray_face_resized = cv2.resize(
            gray_face_eq,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_CUBIC
        )
        hsv_face_resized = cv2.resize(
            hsv_face,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_CUBIC
        )
        
        return gray_face_resized, hsv_face_resized, (x, y, w, h)


class FeatureExtractor:
    """
    Extracts hand-crafted features from preprocessed images:
    - Histogram of Oriented Gradients (HOG)
    - Local Binary Patterns (LBP)
    - Color Histograms (H & S channels from HSV space)
    - Gabor Filters (mean & variance)
    - Gray-Level Co-occurrence Matrix (GLCM) properties
    """
    def __init__(self):
        # Initialize Gabor filter bank kernels
        self.gabor_kernels = []
        for theta in config.GABOR_PARAMS["orientations"]:
            for lambd in config.GABOR_PARAMS["lambdas"]:
                kernel = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=config.GABOR_PARAMS["sigma"],
                    theta=theta,
                    lambd=lambd,
                    gamma=config.GABOR_PARAMS["gamma"],
                    psi=0,
                    ktype=cv2.CV_32F
                )
                self.gabor_kernels.append(kernel)

    def extract_hog(self, gray_face):
        """
        Extract HOG features and return both features and HOG visual representation.
        """
        features, hog_image = hog(
            gray_face,
            orientations=config.HOG_PARAMS["orientations"],
            pixels_per_cell=config.HOG_PARAMS["pixels_per_cell"],
            cells_per_block=config.HOG_PARAMS["cells_per_block"],
            visualize=True
        )
        return features, hog_image

    def extract_lbp(self, gray_face):
        """
        Extract LBP histogram feature and return both histogram and LBP visualization.
        """
        P = config.LBP_PARAMS["P"]
        R = config.LBP_PARAMS["R"]
        method = config.LBP_PARAMS["method"]
        lbp_image = local_binary_pattern(gray_face, P=P, R=R, method=method)
        
        # Compute normalized histogram (uniform method has P + 2 bins)
        n_bins = int(lbp_image.max() + 1)
        hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float32")
        hist /= (hist.sum() + 1e-6)
        return hist, lbp_image

    def extract_color_hist(self, hsv_face):
        """
        Extract 2D joint Color Histogram utilizing H and S channels from HSV face.
        """
        h_bins = config.COLOR_PARAMS["h_bins"]
        s_bins = config.COLOR_PARAMS["s_bins"]
        # Joint histogram of H and S channels (ignore Value/brightness for illumination robustness)
        hist = cv2.calcHist([hsv_face], [0, 1], None, [h_bins, s_bins], [0, 180, 0, 256])
        # Flatten and normalize to sum to 1.0
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def extract_gabor(self, gray_face):
        """
        Extract mean and variance response features from the Gabor filter bank.
        """
        feats = []
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(gray_face, cv2.CV_32F, kernel)
            feats.append(filtered.mean())
            feats.append(filtered.var())
        return np.array(feats, dtype=np.float32)

    def extract_glcm(self, gray_face):
        """
        Extract texture features from GLCM.
        """
        # Quantize image to 64 levels for speed and memory efficiency
        quantized = (gray_face // 4).astype(np.uint8)
        glcm = graycomatrix(
            quantized,
            distances=config.GLCM_PARAMS["distances"],
            angles=config.GLCM_PARAMS["angles"],
            levels=64,
            symmetric=True,
            normed=True
        )
        feats = []
        for prop in config.GLCM_PARAMS["properties"]:
            feats.extend(graycoprops(glcm, prop).flatten())
        return np.array(feats, dtype=np.float32)

    def extract_and_concatenate(self, gray_face, hsv_face):
        """
        Extracts HOG, LBP, Color Hist, Gabor, and GLCM features, and concatenates them.
        """
        hog_feat, _ = self.extract_hog(gray_face)
        lbp_feat, _ = self.extract_lbp(gray_face)
        color_feat = self.extract_color_hist(hsv_face)
        gabor_feat = self.extract_gabor(gray_face)
        glcm_feat = self.extract_glcm(gray_face)
        
        return np.concatenate([
            hog_feat,
            lbp_feat,
            color_feat,
            gabor_feat,
            glcm_feat
        ])


def plot_features(original_bgr, gray_face, hog_img, lbp_img, save_path):
    """
    Saves a figure displaying original, preprocessed face, HOG, and LBP views side-by-side.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Convert BGR to RGB for plotting
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(gray_face, cmap="gray")
    axes[1].set_title("Preprocessed Face", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(hog_img, cmap="gray")
    axes[2].set_title("HOG Representation", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    
    axes[3].imshow(lbp_img, cmap="gray")
    axes[3].set_title("LBP Representation", fontsize=12, fontweight="bold")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_2d_space(features, labels, label_names, save_path):
    """
    Saves t-SNE 2D projection of the feature space.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # Restrict perplexity based on sample count
    perplexity = min(30, len(features) - 1)
    if perplexity < 2:
        perplexity = 1
        
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto")
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        idx = (labels == label)
        name = label_names[label] if label < len(label_names) else str(label)
        plt.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            color=color,
            label=name,
            edgecolors='k',
            alpha=0.8,
            s=70
        )
        
    plt.title("2D Feature Space Projection (t-SNE)", fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Identities")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_inference_results(query_bgr, query_name, matches, save_path):
    """
    Saves top-K retrieved matches visualization with Euclidean distances and boundary border highlights:
    - Green border: Correct identity match.
    - Red border: Wrong identity match.
    - Blue border: Query image.
    matches format: list of tuples (match_bgr_img, match_identity_name, distance_float)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    K = len(matches)
    
    # 1 query plot + K match plots
    fig, axes = plt.subplots(1, K + 1, figsize=(3 * (K + 1), 4))
    if K + 1 == 1:
        axes = [axes]
        
    # Plot Query
    query_rgb = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB)
    axes[0].imshow(query_rgb)
    axes[0].set_title(f"Query:\n{query_name}", fontsize=10, fontweight="bold")
    axes[0].axis("on")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_color("blue")
        spine.set_linewidth(4)
        
    # Plot Matches
    for i, (match_bgr, name, dist) in enumerate(matches):
        ax = axes[i + 1]
        match_rgb = cv2.cvtColor(match_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(match_rgb)
        
        is_correct = (name == query_name)
        border_color = "green" if is_correct else "red"
        
        ax.set_title(f"Rank {i+1}: {name}\nDist: {dist:.2f}", color=border_color, fontsize=9, fontweight="bold")
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(4)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_dataset(root_dir, preprocessor, extractor, max_identities=None, max_images_per_identity=None):
    """
    Loads, preprocesses and extracts features for images under root_dir / identity_name / *.jpg.
    Returns:
        features: numpy array (N, D)
        labels: numpy array (N,)
        label_names: list of unique identity strings
        image_paths: list of source image paths
    """
    features = []
    labels = []
    image_paths = []
    all_label_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if max_identities is not None:
        label_names = all_label_names[:max_identities]
    else:
        label_names = all_label_names
        
    name_to_label = {name: i for i, name in enumerate(label_names)}
    
    print(f"Loading dataset from: {root_dir}")
    for name in label_names:
        person_dir = os.path.join(root_dir, name)
        label = name_to_label[name]
        
        files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if max_images_per_identity is not None:
            files = files[:max_images_per_identity]
            
        print(f"  Identity '{name}': processing {len(files)} files...")
        
        for filename in files:
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            res = preprocessor.preprocess_image(img)
            if res is None:
                print(f"    [WARNING] Face detection failed for {img_path}, skipping.")
                continue
                
            gray_face, hsv_face, _ = res
            feat = extractor.extract_and_concatenate(gray_face, hsv_face)
            
            features.append(feat)
            labels.append(label)
            image_paths.append(img_path)
            
    return np.array(features), np.array(labels), label_names, image_paths


if __name__ == "__main__":
    # Integration of manual test block for independent verification
    print("========================================")
    print("Testing Utils and Feature Extraction...")
    print("========================================")
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Create mock face images (128x128)
    mock_gray = (np.random.rand(128, 128) * 255).astype(np.uint8)
    mock_hsv = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    
    print("Testing Feature Extraction components individually:")
    
    # 1. HOG
    hog_feat, hog_img = extractor.extract_hog(mock_gray)
    print(f"  HOG feature shape: {hog_feat.shape}, image shape: {hog_img.shape}")
    
    # 2. LBP
    lbp_feat, lbp_img = extractor.extract_lbp(mock_gray)
    print(f"  LBP feature shape: {lbp_feat.shape}, image shape: {lbp_img.shape}")
    
    # 3. Color Hist
    color_feat = extractor.extract_color_hist(mock_hsv)
    print(f"  Color Histogram feature shape: {color_feat.shape}")
    
    # 4. Gabor
    gabor_feat = extractor.extract_gabor(mock_gray)
    print(f"  Gabor feature shape: {gabor_feat.shape}")
    
    # 5. GLCM
    glcm_feat = extractor.extract_glcm(mock_gray)
    print(f"  GLCM feature shape: {glcm_feat.shape}")
    
    # 6. Concatenated
    concat_feat = extractor.extract_and_concatenate(mock_gray, mock_hsv)
    print(f"  Concatenated feature shape: {concat_feat.shape}")
    
    # Test visualizations with mock data
    print("\nTesting visualization functions with mock data...")
    mock_original = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_original[50:178, 50:178] = 255  # face bounding region
    
    # Plot features
    plot_features(mock_original, mock_gray, hog_img, lbp_img, "test_features.png")
    print("  plot_features test output saved to test_features.png")
    
    # Plot 2D space (t-SNE)
    mock_features = np.random.rand(10, concat_feat.shape[0])
    mock_labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    plot_2d_space(mock_features, mock_labels, ["ID_A", "ID_B"], "test_tsne.png")
    print("  plot_2d_space test output saved to test_tsne.png")
    
    # Plot inference results
    matches = [
        (mock_original[50:178, 50:178], "ID_A", 1.2),
        (mock_original[50:178, 50:178], "ID_B", 3.4)
    ]
    plot_inference_results(mock_original, "ID_A", matches, "test_inference.png")
    print("  plot_inference_results test output saved to test_inference.png")
    
    # Clean up test output files
    for f in ["test_features.png", "test_tsne.png", "test_inference.png"]:
        if os.path.exists(f):
            os.remove(f)
    print("Cleaned up temporary test visualization files.")
    
    print("Utils and Feature Extraction test PASSED!")
    print("========================================")
