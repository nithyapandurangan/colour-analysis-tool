import pandas as pd
import numpy as np
import cv2
import face_recognition_models
import face_recognition
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils import resample
import joblib
import os
import re
from skimage.color import rgb2lab, rgb2hsv
from skimage.exposure import equalize_hist
from skimage.io import imread, imshow

# Define boost_hair_features outside of class for picklability
def boost_hair_features(X, multiplier=1.0):
    """Boost hair features by multiplier before feature selection"""
    X = X.copy()
    if isinstance(X, pd.DataFrame):
        hair_cols = [col for col in X.columns if 'hair_' in col]
        X[hair_cols] = X[hair_cols] * multiplier  
    return X

# Define boost eye features
def boost_eye_features(X, multiplier=1.0):
    """Boost eye features by multiplier before feature selection"""
    X = X.copy()
    if isinstance(X, pd.DataFrame):
        eye_cols = [col for col in X.columns if 'eye_' in col]
        X[eye_cols] = X[eye_cols] * multiplier
    return X

# Combined feature booster
def boost_features(X, hair_multiplier=1.5, eye_multiplier=1.2):
    """Boost both hair and eye features"""
    X = boost_hair_features(X, hair_multiplier)
    X = boost_eye_features(X, eye_multiplier)
    return X

class EnhancedSeasonalColorDatabase:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.scaler = StandardScaler()
        # Initialize face and hair detection models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def extract_enhanced_features(self, image_path):
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            if not face_locations:
                return None 

            # Convert to different color spaces
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
            image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

            features = {}

            # Get face landmarks
            face_landmarks = face_recognition.face_landmarks(image, face_locations)
            if not face_landmarks:
                return None

            landmarks = face_landmarks[0]
            top, right, bottom, left = face_locations[0]
            face = image_cv[top:bottom, left:right]
            face_lab = image_lab[top:bottom, left:right]
            face_hsv = image_hsv[top:bottom, left:right]

            # Extract eye features
            eye_features = self._extract_eye_features(
                image_cv, image_lab, image_hsv, landmarks)
            features.update(eye_features)

            # Extract skin features
            skin_features = self._extract_skin_features(
                face, face_lab, face_hsv, landmarks)
            features.update(skin_features)

            # Extract hair features
            hair_features = self._extract_hair_features(
                image_cv, image_lab, image_hsv, top, left, right)
            features.update(hair_features)

            # Extract contrast features
            contrast_features = self._extract_contrast_features(
                face, face_lab, face_hsv)
            features.update(contrast_features)

            return features

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def _extract_eye_features(self, image_cv, image_lab, image_hsv, landmarks):

        features = {}

        for eye_name, eye_points in [('left_eye', landmarks['left_eye']), 
                                   ('right_eye', landmarks['right_eye'])]:
            eye_region = self._get_eye_region(image_cv, eye_points)
            eye_region_lab = self._get_eye_region(image_lab, eye_points)
            eye_region_hsv = self._get_eye_region(image_hsv, eye_points)

            # BGR values
            bgr_mean = cv2.mean(eye_region)[:3]
            features.update({
                f'{eye_name}_b': bgr_mean[0],
                f'{eye_name}_g': bgr_mean[1],
                f'{eye_name}_r': bgr_mean[2],
            })

            # LAB values
            lab_mean = cv2.mean(eye_region_lab)[:3]
            features.update({
                f'{eye_name}_l': lab_mean[0],
                f'{eye_name}_a': lab_mean[1],
                f'{eye_name}_b_lab': lab_mean[2],
            })

            # HSV values
            hsv_mean = cv2.mean(eye_region_hsv)[:3]
            features.update({
                f'{eye_name}_h': hsv_mean[0],
                f'{eye_name}_s': hsv_mean[1],
                f'{eye_name}_v': hsv_mean[2],
            })

        # Add eye contrast feature (important for winters)
        left_eye_region = self._get_eye_region(image_cv, landmarks['left_eye'])
        right_eye_region = self._get_eye_region(image_cv, landmarks['right_eye'])
        features['eye_contrast'] = np.std(cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)) + \
                                np.std(cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2GRAY))

        # Calculate eye brightness and clarity - important for seasons
        features['eye_brightness'] = (features['left_eye_l'] + features['right_eye_l']) / 2
        features['eye_clarity'] = (features['left_eye_s'] + features['right_eye_s']) / 2

        return features

    def _extract_skin_features(self, face, face_lab, face_hsv, landmarks):

        features = {}

        # Define multiple sampling regions for more robust skin color detection
        regions = {
            'forehead': (0.2, 0.3, 0.8, 0.4),
            'left_cheek': (0.5, 0.2, 0.7, 0.35),
            'right_cheek': (0.5, 0.65, 0.7, 0.8),
            'chin': (0.8, 0.4, 0.9, 0.6)
        }

        for region_name, (y1, x1, y2, x2) in regions.items():
            h, w = face.shape[:2]
            region = face[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            region_lab = face_lab[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            region_hsv = face_hsv[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]

            if region.size > 0:
                # BGR values
                bgr_mean = cv2.mean(region)[:3]
                features.update({
                    f'skin_{region_name}_b': bgr_mean[0],
                    f'skin_{region_name}_g': bgr_mean[1],
                    f'skin_{region_name}_r': bgr_mean[2],
                })

                # LAB values
                lab_mean = cv2.mean(region_lab)[:3]
                features.update({
                    f'skin_{region_name}_l': lab_mean[0],
                    f'skin_{region_name}_a': lab_mean[1],
                    f'skin_{region_name}_b_lab': lab_mean[2],
                })

                # HSV values
                hsv_mean = cv2.mean(region_hsv)[:3]
                features.update({
                    f'skin_{region_name}_h': hsv_mean[0],
                    f'skin_{region_name}_s': hsv_mean[1],
                    f'skin_{region_name}_v': hsv_mean[2],
                })

        seasonal_features = self._extract_seasonal_signatures(face_lab, face_hsv)
        features.update({
            **seasonal_features,

            # Boosted importance features
            'skin_boosted_warmth': seasonal_features['global_warmth'] * 1.5,
            'skin_boosted_clarity': seasonal_features['color_clarity'] * 1.3
        })
        return features


    def _extract_hair_features(self, image_cv, image_lab, image_hsv, top, left, right):

        """Extracts comprehensive hair features from an image"""
        # Read and convert image
        img = image_cv
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detect face to help isolate hair region
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"No face detected for hair extraction")
            return None  # No face detected

        (x, y, w, h) = faces[0]

        # Define approximate hair region (above and around face)
        hair_region = img[max(0, y-int(h*0.5)):max(0, y-int(h*0.1)), 
                          max(0, x-int(w*0.3)):min(img.shape[1], x+w+int(w*0.3))]

        if hair_region.size == 0:
            return None

        # Calculate various color features in different color spaces
        features = {}

        # RGB features
        rgb_mean = np.mean(hair_region, axis=(0,1))
        rgb_std = np.std(hair_region, axis=(0,1))
        features.update({
            'hair_r_mean': rgb_mean[0],
            'hair_g_mean': rgb_mean[1],
            'hair_b_mean': rgb_mean[2],
            'hair_r_std': rgb_std[0],
            'hair_g_std': rgb_std[1],
            'hair_b_std': rgb_std[2]
        })

        # LAB features (perceptual color space)
        lab_region = cv2.cvtColor(hair_region, cv2.COLOR_RGB2LAB)
        lab_mean = np.mean(lab_region, axis=(0,1))
        lab_std = np.std(lab_region, axis=(0,1))
        features.update({
            'hair_l_mean': lab_mean[0],
            'hair_a_mean': lab_mean[1],
            'hair_b_mean': lab_mean[2],
            'hair_l_std': lab_std[0],
            'hair_a_std': lab_std[1],
            'hair_b_std': lab_std[2]
        })

        # HSV features
        hsv_region = cv2.cvtColor(hair_region, cv2.COLOR_RGB2HSV)
        hsv_mean = np.mean(hsv_region, axis=(0,1))
        hsv_std = np.std(hsv_region, axis=(0,1))
        features.update({
            'hair_h_mean': hsv_mean[0],
            'hair_s_mean': hsv_mean[1],
            'hair_v_mean': hsv_mean[2],
            'hair_h_std': hsv_std[0],
            'hair_s_std': hsv_std[1],
            'hair_v_std': hsv_std[2]
        })

        # Texture features using GLCM
        gray_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_hair, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
        features.update({
            'hair_contrast': graycoprops(glcm, 'contrast')[0, 0],
            'hair_dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
            'hair_homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'hair_ASM': graycoprops(glcm, 'ASM')[0, 0],
            'hair_energy': graycoprops(glcm, 'energy')[0, 0],
            'hair_correlation': graycoprops(glcm, 'correlation')[0, 0]
        })

        # Additional hair characteristics
        brightness = lab_mean[0]
        warmth = lab_mean[1]  # Positive = warm (red), Negative = cool (green)
        saturation = hsv_mean[1]

        features.update({
            'hair_brightness': brightness,
            'hair_warmth': warmth,
            'hair_saturation': saturation,
            'hair_is_dark': float(brightness < 50),
            'hair_is_warm': float(warmth > 0),
            'hair_is_saturated': float(saturation > 50)
        })

        # Hair coverage (percentage of frame occupied by hair)
        hair_mask = self._create_hair_mask(img, faces[0])
        hair_coverage = np.sum(hair_mask) / (hair_mask.shape[0] * hair_mask.shape[1])
        features['hair_coverage'] = hair_coverage

        return features

    def _create_hair_mask(self, img, face_rect):
        """Improved hair segmentation using adaptive thresholding and morphology"""
        x, y, w, h = face_rect
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Expand the search area above the face
        upper_head = gray[max(0, y-int(h*1.2)):max(y-int(h*0.2), y), 
                        max(0, x-int(w*0.4)):min(img.shape[1], x+w+int(w*0.4))]

        if upper_head.size == 0:
            return np.zeros(gray.shape[:2], dtype=np.uint8)

        # Adaptive thresholding works better than global for varying lighting
        mask = cv2.adaptiveThreshold(upper_head, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Create full image mask
        full_mask = np.zeros(gray.shape[:2], dtype=np.uint8)
        roi_y1 = max(0, y-int(h*1.2))
        roi_y2 = max(y-int(h*0.2), y)
        full_mask[roi_y1:roi_y2, max(0, x-int(w*0.4)):min(img.shape[1], x+w+int(w*0.4))] = mask

        return full_mask

    def _extract_contrast_features(self, face, face_lab, face_hsv):

        features = {}

        # Calculate overall contrast
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        features['contrast'] = np.std(gray)

        # Calculate color variance
        for i, channel in enumerate(['b', 'g', 'r']):
            features[f'{channel}_variance'] = np.std(face[:,:,i])

        # Calculate LAB space features
        for i, channel in enumerate(['l', 'a', 'b']):
            features[f'{channel}_variance'] = np.std(face_lab[:,:,i])

        # Calculate HSV space features
        for i, channel in enumerate(['h', 's', 'v']):
            features[f'{channel}_variance'] = np.std(face_hsv[:,:,i])

        return features

    def _get_eye_region(self, image, eye_points):

        eye_points = np.array(eye_points)
        x, y, w, h = cv2.boundingRect(eye_points)
        return image[y:y+h, x:x+w]

    def _extract_seasonal_signatures(self, face_lab, face_hsv):
        # New composite features that directly capture seasonal characteristics
        return {
            # Warm/Cool Dominance (positive = warm, negative = cool)
            'global_warmth': np.mean(face_lab[:,:,1]) - np.mean(face_lab[:,:,2]),

            # Muted vs Clear (low values = muted/soft, high = clear/bright)
            'color_clarity': np.mean(face_hsv[:,:,1]) * np.std(face_hsv[:,:,2]),

            # Light vs Deep (lightness combined with saturation)
            'depth_score': np.mean(face_lab[:,:,0]) * (1 - np.mean(face_hsv[:,:,1]))
        }

    def _get_dominant_colors(self, region, n_colors=3):
        """Helper for extracting dominant colors"""
        pixels = region.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return centers

class ImprovedSeasonalColorModel:
    def __init__(self, csv_path='season_db.csv'):
        self.df = pd.read_csv(csv_path)
        self.model = None
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)

    def preprocess_data(self):
        """Balance the dataset by upsampling minority classes"""
        classes = self.df['season'].value_counts()
        majority_class = classes.index[0]
        n_samples = classes[majority_class]

        # Upsample minority classes
        balanced_dfs = []
        for season in classes.index:
            season_df = self.df[self.df['season'] == season]
            if len(season_df) < n_samples:
                upsampled_df = resample(season_df,
                                      replace=True,
                                      n_samples=n_samples,
                                      random_state=42)
                balanced_dfs.append(upsampled_df)
            else:
                balanced_dfs.append(season_df)

        self.df = pd.concat(balanced_dfs)

    def analyze_feature_groups(self, model, feature_names):
        """Analyze the importance of different feature groups"""
        try:
            # Get feature importances
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importance = model.named_steps['classifier'].feature_importances_
            else:
                print("Classifier doesn't have feature_importances_ attribute")
                return

            # Ensure we have the same number of features as importance scores
            if len(importance) != len(feature_names):
                print(f"Mismatch between features ({len(feature_names)}) and importance scores ({len(importance)})")
                return

            # Create feature groups - only include indices that exist
            groups = {
                'Skin': [i for i,name in enumerate(feature_names) if 'skin_' in name and i < len(importance)],
                'Hair': [i for i,name in enumerate(feature_names) if 'hair_' in name and i < len(importance)],
                'Eyes': [i for i,name in enumerate(feature_names) if 'eye_' in name and i < len(importance)],
                'Global': [i for i,name in enumerate(feature_names) 
                          if name in ['contrast', 'global_warmth', 'color_clarity', 'depth_score'] 
                          and i < len(importance)]
            }

            print("\nFeature Group Importance:")
            for group, indices in groups.items():
                if indices:  # Only calculate if there are valid indices
                    group_importance = sum(importance[indices])
                    print(f"{group}: {group_importance:.2%}")
                else:
                    print(f"{group}: No features found")

            # Visualize only if we have valid groups
            if any(groups.values()):
                plt.figure(figsize=(10,5))
                plt.bar(groups.keys(), 
                       [sum(importance[indices]) if indices else 0 for indices in groups.values()])
                plt.title("Feature Group Contribution")
                plt.ylabel("Importance Percentage")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error in analyze_feature_groups: {str(e)}")

    def train_model(self, test_size=0.2, random_state=42):
        """Train the seasonal color classification model"""
        self.preprocess_data()

        # Include all feature types
        all_features = [col for col in self.df.columns 
                        if col not in ['season', 'image_file']]

        print(f"Total features: {len(all_features)}")
        print("Features to use:", all_features)

        if not all_features:
            raise ValueError("No valid features found in the dataset")

        X = self.df[all_features]
        y = self.df['season']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        # Create feature boosters - using standalone functions for picklability
        hair_booster_1_5 = FunctionTransformer(func=lambda X: boost_hair_features(X, 1.5))
        hair_booster_2_0 = FunctionTransformer(func=lambda X: boost_hair_features(X, 2.0))
        eye_booster_1_5 = FunctionTransformer(func=lambda X: boost_eye_features(X, 1.5))
        combined_booster = FunctionTransformer(func=lambda X: boost_features(X, 1.5, 1.2))

        # Modified pipeline
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('feature_booster', combined_booster),  # Use combined booster by default
            ('selector', SelectFromModel(
                ExtraTreesClassifier(n_estimators=50, max_features=0.8, random_state=random_state), 
                threshold='median')),  
            ('classifier', RandomForestClassifier(
                class_weight='balanced_subsample',
                n_estimators=100,
                n_jobs=-1,
                max_features=0.6,  
                min_samples_leaf=3,
                random_state=42 
            ))
        ])

        # Parameter grid - using static boosters instead of lambdas
        param_dist = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 15, 20],
            'classifier__min_samples_split': [5],
            'classifier__min_samples_leaf': [1, 3],
            'classifier__max_features': ['sqrt', 0.3, 0.6],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__class_weight': ['balanced', 'balanced_subsample', None],
            'feature_booster': [hair_booster_1_5, hair_booster_2_0, eye_booster_1_5, combined_booster]
        }

        # Use RandomizedSearchCV for efficiency with larger parameter space
        grid_search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist, 
            n_iter=20, 
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            scoring='balanced_accuracy',
            n_jobs=-1, 
            random_state=random_state,
            verbose=2,
            error_score='raise'
            )

        # Train model
        print("Starting model training...")
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print("Model training complete.")

        # Get selected features from the model
        selector = self.model.named_steps['selector']
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices]
        print(f"\nSelected {len(selected_features)} features out of {len(X.columns)}")
        print("Selected features:", selected_features.tolist())

        # Get feature importances from the best model
        feature_importances = self.model.named_steps['classifier'].feature_importances_
        selected_feature_importances = [(selected_features[i], feature_importances[i]) 
                                       for i in range(len(selected_features))]
        sorted_importances = sorted(selected_feature_importances, key=lambda x: x[1], reverse=True)

        print("\nTop 20 feature importances:")
        for feature, importance in sorted_importances[:20]:
            print(f"{feature}: {importance:.4f}")

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        indices = np.argsort(feature_importances)[::-1][:20]  # Show top 20 features
        plt.bar(range(len(indices)), feature_importances[indices], align="center")
        plt.xticks(range(len(indices)), [selected_features[i] for i in indices], rotation=90)
        plt.title("Feature Importances in Random Forest")
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

        # Analyze feature groups
        self.analyze_feature_groups(self.model, selected_features)

        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nBest parameters:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=np.unique(y),
                   yticklabels=np.unique(y))
        plt.title('Confusion Matrix')
        plt.ylabel('True Season')
        plt.xlabel('Predicted Season')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')

        # Save model
        print("Saving model...")
        try:
            joblib.dump(self.model, 'seasonal_color_model.joblib')
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

        return self.model

    def predict(self, features_dict):
        """Predict season based on extracted features"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Convert feature dictionary to DataFrame
        features_df = pd.DataFrame([features_dict])

        # Match columns with training data
        missing_cols = set(self.model[0].feature_names_in_) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = 0  # Fill with default values for missing features

        # Keep only columns used in training
        features_df = features_df[self.model[0].feature_names_in_]

        # Make prediction
        season = self.model.predict(features_df)[0]
        probabilities = self.model.predict_proba(features_df)[0]
        season_probabilities = {s: p for s, p in zip(self.model.classes_, probabilities)}

        return season, season_probabilities

def main():
    # Create enhanced database
    print("Starting seasonal color analysis...")
    db_creator = EnhancedSeasonalColorDatabase(r"/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/Dataset/Seasons")

    # Process images and create features
    data = []
    print(f"Looking for directory: {db_creator.image_directory}")
    for image_file in os.listdir(db_creator.image_directory):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            season_match = re.match(r'([a-zA-Z_]+)_\d+', image_file)
            if season_match:
                season = season_match.group(1)
                image_path = os.path.join(db_creator.image_directory, image_file)
                features = db_creator.extract_enhanced_features(image_path)

                if features:
                    features['image_file'] = image_file
                    features['season'] = season
                    data.append(features)
                    print(f"Processed {image_file}")

    # Create and save database
    df = pd.DataFrame(data)
    df.to_csv('season_db.csv', index=False)
    print("Database created and saved successfully!")

    # Train the model
    print("\nTraining the model...")
    model = ImprovedSeasonalColorModel('season_db.csv')
    print(f"Looking for CSV at: {os.path.abspath('season_db.csv')}")
    model.train_model()
    print("\nModel training completed!")

if __name__ == "__main__":
    main()





