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

class EnhancedSeasonalColorDatabase:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.scaler = StandardScaler()
        
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
        
        return features
    
    def _extract_hair_features(self, image_cv, image_lab, image_hsv, top, left, right):
        
        features = {}
        
        # Sample multiple regions of hair
        hair_regions = [
            image_cv[max(0, top-30):top, left:right],  # Top
            image_cv[max(0, top-30):top, left:left+30],  # Left side
            image_cv[max(0, top-30):top, right-30:right]  # Right side
        ]
        
        hair_regions_lab = [
            image_lab[max(0, top-30):top, left:right],
            image_lab[max(0, top-30):top, left:left+30],
            image_lab[max(0, top-30):top, right-30:right]
        ]
        
        hair_regions_hsv = [
            image_hsv[max(0, top-30):top, left:right],
            image_hsv[max(0, top-30):top, left:left+30],
            image_hsv[max(0, top-30):top, right-30:right]
        ]
        
        for i, (region, region_lab, region_hsv) in enumerate(zip(
            hair_regions, hair_regions_lab, hair_regions_hsv)):
            if region.size > 0:
                # BGR values
                bgr_mean = cv2.mean(region)[:3]
                features.update({
                    f'hair_region{i}_b': bgr_mean[0],
                    f'hair_region{i}_g': bgr_mean[1],
                    f'hair_region{i}_r': bgr_mean[2],
                })
                
                # LAB values
                lab_mean = cv2.mean(region_lab)[:3]
                features.update({
                    f'hair_region{i}_l': lab_mean[0],
                    f'hair_region{i}_a': lab_mean[1],
                    f'hair_region{i}_b_lab': lab_mean[2],
                })
                
                # HSV values
                hsv_mean = cv2.mean(region_hsv)[:3]
                features.update({
                    f'hair_region{i}_h': hsv_mean[0],
                    f'hair_region{i}_s': hsv_mean[1],
                    f'hair_region{i}_v': hsv_mean[2],
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
        
        return features
    
    def _get_eye_region(self, image, eye_points):
        
        eye_points = np.array(eye_points)
        x, y, w, h = cv2.boundingRect(eye_points)
        return image[y:y+h, x:x+w]


class ImprovedSeasonalColorModel:
    def __init__(self, csv_path='season_db.csv'):
        self.df = pd.read_csv(csv_path)
        self.model = None
        self.scaler = RobustScaler() # Changed to RobustScaler
        self.pca = PCA(n_components=0.95)

    def preprocess_data(self):
        # ... (Existing upsampling remains unchanged) ...
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

    def train_model(self, test_size=0.2, random_state=42):
        # ... (Data preparation remains largely unchanged) ...
        # Preprocess data
        self.preprocess_data()
        
        # Prepare features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['image_file', 'season']]
        X = self.df[feature_cols]
        y = self.df['season']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)

        #Improved Pipeline with additional preprocessing steps
        pipeline = Pipeline([
            ('scaler', self.scaler),
            ('pca', self.pca),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=random_state))
        ])

        # Expanded parameter grid for more thorough search
        param_dist = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30, 50],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__class_weight': ['balanced', 'balanced_subsample', None]
        }

        # Use RandomizedSearchCV for efficiency with larger parameter space
        grid_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1, random_state=random_state)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_

        # Get feature names before PCA
        feature_names = X_train.columns
    
        # Get feature importances from the best model
        feature_importances = self.model.named_steps['classifier'].feature_importances_
    
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        indices = np.argsort(feature_importances)[::-1]
        plt.bar(range(len(indices)), feature_importances[indices], align="center")
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.title("Feature Importances in Random Forest")
        plt.tight_layout()
        plt.show()

        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Print results
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
        joblib.dump(self.model, 'seasonal_color_model.joblib')
        
        return self.model


def main():
    # ... (Database creation remains unchanged) ...
    # Create enhanced database
    print("Creating enhanced database...")
    # Assuming you have the EnhancedSeasonalColorDatabase class defined
    db_creator = EnhancedSeasonalColorDatabase("/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/Dataset/Seasons")
   
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

