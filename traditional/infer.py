import os
import cv2
import numpy as np
from src import config
from src.utils import FacePreprocessor, FeatureExtractor, plot_features, plot_inference_results
from src.model import TraditionalPipeline

def main():
    print("========================================")
    print("Starting Traditional CV Pipeline Inference")
    print("========================================")
    
    # 1. Load the trained pipeline
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Trained pipeline model not found at {config.MODEL_PATH}. Please run train.py first.")
        
    print(f"Loading pipeline model from: {config.MODEL_PATH}")
    pipeline = TraditionalPipeline.load(config.MODEL_PATH)
    label_names = pipeline.label_names
    gallery_image_paths = pipeline.gallery_image_paths
    gallery_labels = pipeline.gallery_labels
    
    # 2. Initialize preprocessor and extractor
    preprocessor = FacePreprocessor()
    extractor = FeatureExtractor()
    
    # 3. Traverse Probe directory and gather queries
    if not os.path.exists(config.PROBE_DIR):
        raise FileNotFoundError(f"Probe directory not found at: {config.PROBE_DIR}")
        
    probe_identities = sorted([d for d in os.listdir(config.PROBE_DIR) if os.path.isdir(os.path.join(config.PROBE_DIR, d))])
    # Only query identities the model was trained on
    probe_identities = [d for d in probe_identities if d in label_names]
    
    correct_predictions = 0
    total_predictions = 0
    
    # We will save some sample visualizations
    saved_feature_vis = False
    
    print("\nRunning identification (1-to-N matching) on Probe set...")
    
    for identity_name in probe_identities:
        person_dir = os.path.join(config.PROBE_DIR, identity_name)
        probe_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if config.MAX_IMAGES_PER_IDENTITY is not None:
            probe_files = probe_files[:config.MAX_IMAGES_PER_IDENTITY]
        
        # Check if identity is known in training
        if identity_name in label_names:
            true_label = label_names.index(identity_name)
        else:
            true_label = -1  # Unknown identity
            
        print(f"\nProcessing probe identity '{identity_name}' ({len(probe_files)} files):")
        
        for idx, filename in enumerate(probe_files):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Preprocess image
            res = preprocessor.preprocess_image(img)
            if res is None:
                print(f"  [WARNING] Face detection failed for {filename}, skipping.")
                continue
                
            gray_face, hsv_face, bbox = res
            
            # Extract features
            feat = extractor.extract_and_concatenate(gray_face, hsv_face)
            
            # Predict
            pred_label = pipeline.predict([feat])[0]
            pred_name = label_names[pred_label]
            
            is_correct = (pred_name == identity_name)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "CORRECT" if is_correct else f"WRONG (predicted: {pred_name})"
            print(f"  Probe {filename} -> Predicted: {pred_name} [{status}]")
            
            # Save visual features (original, HOG, LBP views) for the first probe image processed
            if not saved_feature_vis:
                # Get HOG & LBP representations
                _, hog_img = extractor.extract_hog(gray_face)
                _, lbp_img = extractor.extract_lbp(gray_face)
                feat_vis_path = os.path.join(config.RESULT_DIR, "features_sample.png")
                plot_features(img, gray_face, hog_img, lbp_img, feat_vis_path)
                print(f"  [VISUALIZATION] Saved feature visualization to: {feat_vis_path}")
                saved_feature_vis = True
                
            # Save Top-K matches visualization for the first image of each identity
            if idx == 0:
                # Retrieve Top-K nearest neighbors
                K = min(5, len(gallery_image_paths))
                dists, indices = pipeline.get_neighbors([feat], n_neighbors=K)
                
                matches = []
                for rank in range(K):
                    neighbor_idx = indices[0][rank]
                    dist = dists[0][rank]
                    neighbor_path = gallery_image_paths[neighbor_idx]
                    neighbor_label = gallery_labels[neighbor_idx]
                    neighbor_name = label_names[neighbor_label]
                    
                    neighbor_img = cv2.imread(neighbor_path)
                    matches.append((neighbor_img, neighbor_name, dist))
                    
                match_vis_path = os.path.join(config.RESULT_DIR, f"retrieval_{identity_name}.png")
                plot_inference_results(img, identity_name, matches, match_vis_path)
                print(f"  [VISUALIZATION] Saved Top-{K} retrieval results to: {match_vis_path}")
                
    # 4. Summary & Evaluation Metrics
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print("\n========================================")
        print("Inference Evaluation Summary:")
        print("========================================")
        print(f"Total processed queries: {total_predictions}")
        print(f"Correctly identified: {correct_predictions}")
        print(f"Top-1 Identification Accuracy: {accuracy * 100:.2f}%")
    else:
        print("\nNo queries successfully processed.")
        
    print("\n========================================")
    print("Inference Pipeline execution completed!")
    print("========================================")

if __name__ == "__main__":
    main()
