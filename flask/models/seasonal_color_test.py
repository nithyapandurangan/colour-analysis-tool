import cv2
import face_recognition
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
import os
import colorsys

class SeasonalColorTester:
    def __init__(self, model_path='seasonal_color_model_enhanced.joblib'):
        """Initialize the tester with the trained model"""
        self.model = joblib.load(model_path)
        # Get feature names from the model pipeline
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = self.model.feature_names_in_
        else:
            self.feature_names = self.model.steps[0][1].feature_names_in_
            
    def extract_color_features(self, region, prefix):
        """Extract comprehensive color features for a given region"""
        features = {}
        
        if region.size == 0 or region is None:
            return features
            
        # Original RGB values
        rgb_means = np.mean(region, axis=(0,1))
        rgb_vars = np.var(region, axis=(0,1))
        
        features[f'{prefix}_r'] = rgb_means[0]
        features[f'{prefix}_g'] = rgb_means[1]
        features[f'{prefix}_b'] = rgb_means[2]
        features[f'{prefix}_r_variance'] = rgb_vars[0]
        features[f'{prefix}_g_variance'] = rgb_vars[1]
        features[f'{prefix}_b_variance'] = rgb_vars[2]
        
        # HSV features
        hsv_region = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        hsv_means = np.mean(hsv_region, axis=(0,1))
        hsv_vars = np.var(hsv_region, axis=(0,1))
        
        features[f'{prefix}_h'] = hsv_means[0]
        features[f'{prefix}_s'] = hsv_means[1]
        features[f'{prefix}_v'] = hsv_means[2]
        features[f'{prefix}_h_variance'] = hsv_vars[0]
        features[f'{prefix}_s_variance'] = hsv_vars[1]
        features[f'{prefix}_v_variance'] = hsv_vars[2]
        
        # LAB features
        lab_region = cv2.cvtColor(region, cv2.COLOR_RGB2LAB)
        lab_means = np.mean(lab_region, axis=(0,1))
        lab_vars = np.var(lab_region, axis=(0,1))
        
        features[f'{prefix}_l'] = lab_means[0]
        features[f'{prefix}_a'] = lab_means[1]
        features[f'{prefix}_b_lab'] = lab_means[2]
        features[f'{prefix}_l_variance'] = lab_vars[0]
        features[f'{prefix}_a_variance'] = lab_vars[1]
        features[f'{prefix}_b_lab_variance'] = lab_vars[2]
        
        return features
        
    def extract_facial_features(self, rgb_image, face_landmarks):
        """Extract features from specific facial regions using landmarks"""
        features = {}
        height, width = rgb_image.shape[:2]
        
        # Function to get region of interest
        def get_roi(points, padding=5):
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            min_x, max_x = max(0, min(x_coords) - padding), min(width, max(x_coords) + padding)
            min_y, max_y = max(0, min(y_coords) - padding), min(height, max(y_coords) + padding)
            return rgb_image[min_y:max_y, min_x:max_x]
            
        # Extract features for each facial region
        facial_regions = {
            'left_eye': face_landmarks['left_eye'],
            'right_eye': face_landmarks['right_eye'],
            'nose_bridge': face_landmarks['nose_bridge'],
            'nose_tip': face_landmarks['nose_tip'],
            'top_lip': face_landmarks['top_lip'],
            'bottom_lip': face_landmarks['bottom_lip']
        }
        
        # Process each facial region
        for region_name, points in facial_regions.items():
            roi = get_roi(points)
            if roi.size > 0:
                region_features = self.extract_color_features(roi, region_name)
                features.update(region_features)
        
        # Extract skin features from specific regions
        skin_regions = {
            'skin_forehead': rgb_image[:height//4],
            'skin_left_cheek': rgb_image[height//4:height//2, :width//3],
            'skin_right_cheek': rgb_image[height//4:height//2, 2*width//3:],
            'skin_chin': rgb_image[3*height//4:]
        }
        
        for region_name, roi in skin_regions.items():
            if roi.size > 0:
                region_features = self.extract_color_features(roi, region_name)
                features.update(region_features)
                
        return features
        
    def extract_features(self, image_path):
        """Extract all features needed for the model"""
        # Read and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face and landmarks
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            raise ValueError("No face detected in the image")
            
        face_landmarks = face_recognition.face_landmarks(rgb_image)
        if not face_landmarks:
            raise ValueError("Could not detect facial landmarks")
            
        # Get face region
        top, right, bottom, left = face_locations[0]
        face_image = rgb_image[top:bottom, left:right]
        
        features = {}
        
        # Extract hair region features
        height = face_image.shape[0]
        hair_regions = {
            'hair_region0': face_image[:height//3],
            'hair_region1': face_image[height//3:2*height//3],
            'hair_region2': face_image[2*height//3:]
        }
        
        for region_name, region in hair_regions.items():
            region_features = self.extract_color_features(region, region_name)
            features.update(region_features)
        
        # Extract facial features
        facial_features = self.extract_facial_features(face_image, face_landmarks[0])
        features.update(facial_features)
        
        # Calculate global image features
        gray_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        features['contrast'] = np.std(gray_image)
        features['brightness'] = np.mean(gray_image)
        
        # Create DataFrame with all features
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            # Fill missing features with 0 or median values
            for feature in missing_features:
                df[feature] = 0
                
        # Reorder columns to match model's expected features
        df = df[self.feature_names]
        
        return df
    
    def predict_season(self, image_path):
        """Predict the season for a given image"""
        try:
            # Extract features
            features = self.extract_features(image_path)
            
            # Make prediction
            prediction = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            # Get confidence scores
            season_probs = {
                season: prob 
                for season, prob in zip(self.model.classes_, probabilities[0])
            }
            
            return {
                'predicted_season': prediction[0],
                'confidence_scores': season_probs
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }

def main():
    # Initialize tester
    tester = SeasonalColorTester()
    
    # Test directory structure
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Process all images in test directory
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nProcessing {image_file}...")
            
            result = tester.predict_season(image_path)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Predicted Season: {result['predicted_season']}")
                print("\nConfidence Scores:")
                for season, prob in result['confidence_scores'].items():
                    print(f"{season}: {prob:.2%}")

if __name__ == "__main__":
    main()