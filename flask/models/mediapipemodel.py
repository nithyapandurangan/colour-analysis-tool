import pandas as pd
import numpy as np
import cv2
import face_recognition_models
import face_recognition
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.cluster import KMeans

class EnhancedSeasonalColorDatabase:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.scaler = StandardScaler()
        # Initialize hair segmenter
        model_path = "hair_segmenter.tflite"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )
        self.hair_segmenter = vision.ImageSegmenter.create_from_options(options)
        
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
            
            # Extract hair features using MediaPipe segmentation
            hair_features = self._extract_hair_features_mediapipe(
                image_cv, image_lab, image_hsv)
            features.update(hair_features)
            
            # Extract contrast features
            contrast_features = self._extract_contrast_features(
                face, face_lab, face_hsv)
            features.update(contrast_features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
            
    def _extract_hair_features_mediapipe(self, image_cv, image_lab, image_hsv):
        features = {}
        
        # Segment hair using MediaPipe
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        segmentation_result = self.hair_segmenter.segment(mp_image)
        hair_mask = segmentation_result.category_mask.numpy_view() == 1
        
        # Extract hair pixels from all color spaces
        hair_pixels_bgr = image_cv[hair_mask]
        hair_pixels_lab = image_lab[hair_mask]
        hair_pixels_hsv = image_hsv[hair_mask]
        
        if len(hair_pixels_bgr) == 0:
        # If no hair detected, fill with zeros for expected features
            features.update({
                'hair_region0_b': 0,
                'hair_region0_g': 0,
                'hair_region0_r': 0,
                'hair_region0_l': 0,
                'hair_region0_a': 0,
                'hair_region0_b_lab': 0,
                'hair_region0_h': 0,
                'hair_region0_s': 0,
                'hair_region0_v': 0,
                'hair_coverage': 0  # Important feature that was missing
            })
            return features
        
        # Calculate mean values (only region0 features)
        bgr_mean = np.mean(hair_pixels_bgr, axis=0)
        lab_mean = np.mean(hair_pixels_lab, axis=0)
        hsv_mean = np.mean(hair_pixels_hsv, axis=0)

            # Store features (only the ones used in training)
        features.update({
            'hair_region0_b': bgr_mean[0],
            'hair_region0_g': bgr_mean[1],
            'hair_region0_r': bgr_mean[2],
            'hair_region0_l': lab_mean[0],
            'hair_region0_a': lab_mean[1],
            'hair_region0_b_lab': lab_mean[2],
            'hair_region0_h': hsv_mean[0],
            'hair_region0_s': hsv_mean[1],
            'hair_region0_v': hsv_mean[2],
            'hair_coverage': np.mean(hair_mask)  # Important feature that was missing
        })
        
        return features
        
        # Get dominant color via K-means (BGR space)
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(hair_pixels_bgr)
        dominant_bgr = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        
        # Calculate mean values
        bgr_mean = np.mean(hair_pixels_bgr, axis=0)
        lab_mean = np.mean(hair_pixels_lab, axis=0)
        hsv_mean = np.mean(hair_pixels_hsv, axis=0)
        
        # Store features (maintaining original naming convention)
        features.update({
            'hair_region0_b': bgr_mean[0],
            'hair_region0_g': bgr_mean[1],
            'hair_region0_r': bgr_mean[2],
            'hair_region0_l': lab_mean[0],
            'hair_region0_a': lab_mean[1],
            'hair_region0_b_lab': lab_mean[2],
            'hair_region0_h': hsv_mean[0],
            'hair_region0_s': hsv_mean[1],
            'hair_region0_v': hsv_mean[2],
            'hair_coverage': np.mean(hair_mask)  # New feature
        })
        
        return features
    
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
        
        return features
    
    def _extract_skin_features(self, face, face_lab, face_hsv, landmarks):
        features = {}
        
        # Define multiple sampling regions
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
        
        return features
    
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
        
        # Enhanced features for seasonal analysis
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        features['autumn_warmth_index'] = (
            0.6 * np.mean(lab[:,:,1]) + 0.4 * np.mean(hsv[:,:,0])
        )
        
        features['autumn_softness'] = (
            np.mean(hsv[:,:,1]) * (1 - features['contrast']/255)
        )
        
        features['winter_clearness'] = (
            features['contrast'] + 0.5 * np.std(lab[:,:,2])
        )

        features['autumn_golden_ratio'] = np.mean(lab[:,:,1]) / (np.mean(lab[:,:,2]) + 1e-6)
        features['autumn_mutedness'] = np.mean(hsv[:,:,1]) * features['contrast']
        
        features['winter_contrast_depth'] = np.percentile(lab[:,:,0], 75) - np.percentile(lab[:,:,0], 25)
        features['winter_coolness'] = np.mean(lab[:,:,2]) - np.mean(lab[:,:,1])
        
        return features
    
    def _get_eye_region(self, image, eye_points):
        eye_points = np.array(eye_points)
        x, y, w, h = cv2.boundingRect(eye_points)
        return image[y:y+h, x:x+w]


class ImprovedSeasonalColorModel:
    def __init__(self, csv_path='season_db.csv'):
        self.df = pd.read_csv(csv_path)
        self.model = None
        self.scaler = RobustScaler() 
        self.pca = PCA(n_components=0.95)

    def preprocess_data(self):
        # Upsample minority classes
        classes = self.df['season'].value_counts()
        majority_class = classes.index[0]
        n_samples = classes[majority_class]
        
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

        # Enhanced feature engineering
        self.df['warm_cool_balance'] = (
            self.df['skin_forehead_a'] - 0.5 * self.df['skin_forehead_b_lab']
        )

        self.df['autumn_dark_warmth'] = (
            0.8 * self.df['skin_forehead_a'] + 
            0.6 * self.df['skin_forehead_l'] + 
            0.4 * self.df['contrast']
        )
        
        self.df['dark_season_indicator'] = (
            -0.7 * self.df['skin_forehead_l'] +
            -0.7 * self.df['hair_region0_l'] +
            0.3 * self.df['contrast']
        )
        
        self.df['winter_coolness_index'] = (
            -0.6 * self.df['skin_forehead_a'] +
            -0.6 * self.df['skin_left_cheek_a'] +
            0.4 * self.df['winter_contrast_depth']
        )
        
        self.df['soft_season_indicator'] = (
            -0.7 * self.df['contrast'] +
            0.5 * self.df['autumn_mutedness'] +
            -0.4 * self.df['v_variance']
        )

        self.df['winter_features'] = (
            0.5 * self.df['hair_region0_l'] + 
            0.3 * self.df['contrast'] + 
            0.2 * self.df['winter_coolness']
        )

    def augment_problem_seasons(self, n_samples=100):
        problem_seasons = ['dark_autumn', 'true_autumn', 'soft_autumn', 'true_winter', 'dark_winter']
        augmented = []
        
        for season in problem_seasons:
            season_df = self.df[self.df['season'] == season]
            
            for _ in range(n_samples):
                sample = season_df.sample(1).copy()
                
                if season == 'dark_autumn':
                    sample['skin_forehead_a'] *= np.random.uniform(1.05, 1.15)
                    sample['contrast'] *= np.random.uniform(0.8, 0.95)
                    sample['autumn_dark_warmth'] *= np.random.uniform(0.95, 1.05)
                    sample['autumn_golden_ratio'] *= np.random.uniform(0.95, 1.05)
                
                elif season == 'true_autumn':
                    sample['autumn_golden_ratio'] *= np.random.uniform(0.98, 1.08)
                
                elif season == 'soft_autumn':
                    sample['contrast'] *= np.random.uniform(0.85, 0.95)
                    sample['autumn_mutedness'] *= np.random.uniform(1.05, 1.15)
                
                elif season == 'true_winter':
                    sample['winter_coolness'] *= np.random.uniform(0.95, 1.05)
                    sample['winter_contrast_depth'] *= np.random.uniform(1.05, 1.15)
                
                elif season == 'dark_winter':
                    sample['winter_coolness'] *= np.random.uniform(1.05, 1.15)
                    sample['contrast'] *= np.random.uniform(0.8, 0.9)
                
                augmented.append(sample)
        
        self.df = pd.concat([self.df] + augmented, ignore_index=True)

    def analyze_feature_importance(self):
        if not hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            print("Classifier doesn't support feature importance analysis")
            return
        
        feature_cols = [col for col in self.df.columns if col not in ['image_file', 'season']]
        X = self.df[feature_cols]
        
        if 'pca' in self.model.named_steps:
            pca = self.model.named_steps['pca']
            rf = self.model.named_steps['classifier']
            
            importance = np.zeros(len(X.columns))
            for i, component_importance in enumerate(rf.feature_importances_):
                importance += np.abs(pca.components_[i]) * component_importance
            importance /= importance.sum()
        else:
            importance = self.model.named_steps['classifier'].feature_importances_

        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print("\nCOMPLETE FEATURE IMPORTANCE (All Features):")
        pd.set_option('display.max_rows', None)
        print(feature_importance)
        pd.reset_option('display.max_rows')

        feature_types = {
            'Eye': [f for f in X.columns if 'eye_' in f],
            'Skin': [f for f in X.columns if 'skin_' in f], 
            'Hair': [f for f in X.columns if 'hair_' in f],
            'Other': [f for f in X.columns if f not in ['contrast'] and 
                     not any(x in f for x in ['eye_', 'skin_', 'hair_'])]
        }

        type_importance = {
            typ: feature_importance.loc[feature_importance['Feature'].isin(features), 'Importance'].sum()
            for typ, features in feature_types.items()
        }

        plt.figure(figsize=(10,5))
        plt.bar(type_importance.keys(), type_importance.values())
        plt.title("Total Feature Importance by Category")
        plt.ylabel("Sum of Importance Scores")
        plt.show()

        print("\nFEATURE TYPE BREAKDOWN:")
        for typ, imp in type_importance.items():
            print(f"{typ}: {imp:.1%} of total importance")

    def train_model(self, test_size=0.2, random_state=42):
        self.preprocess_data()
        self.augment_problem_seasons()
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['image_file', 'season']]
        X = self.df[feature_cols]
        y = self.df['season']

        class_weights = {
            'dark_autumn': 3.0,
            'true_autumn': 2.8,  
            'bright_spring': 2.5,  
            'dark_winter': 2.2,  
            'soft_autumn': 2.0,  
            'bright_winter': 1.0,
            'light_spring': 2.0,
            'light_summer': 1.0,
            'soft_summer': 1.0,
            'true_spring': 1.0,
            'true_summer': 1.0,
            'true_winter': 1.0
        }
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('pca', self.pca),
            ('classifier', RandomForestClassifier(
                class_weight=class_weights,
                n_estimators=300,  
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features='sqrt',
                criterion='entropy',
                random_state=random_state
            ))
        ])

        param_dist = {
            'classifier__n_estimators': [200, 300, 400],
            'classifier__max_depth': [None, 20, 30],
            'classifier__min_samples_split': [2, 5, 5],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__criterion': ['gini', 'entropy'],
        }

        grid_search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist, 
            n_iter=100, 
            cv=5, 
            scoring='recall_macro',
            n_jobs=-1, 
            random_state=random_state
        )
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.analyze_feature_importance()

        y_pred = self.model.predict(X_test)
        print("\nBest parameters:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
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
        
        joblib.dump(self.model, 'seasonalcolourmodel.joblib')
        
        return self.model


def main():
    print("Creating enhanced database...")
    db_creator = EnhancedSeasonalColorDatabase("/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/Dataset/Seasons")
   
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

    df = pd.DataFrame(data)
    df.to_csv('seasondb.csv', index=False)
    print("Database created and saved successfully!")

    print("\nTraining the model...")
    model = ImprovedSeasonalColorModel('seasondb.csv')
    print(f"Looking for CSV at: {os.path.abspath('seasondb.csv')}")
    model.train_model()
    print("\nModel training completed!")

if __name__ == "__main__":
    main()
