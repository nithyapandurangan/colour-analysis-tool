import pandas as pd
import numpy as np
import cv2
import face_recognition
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys
import joblib
import os
import re

class EnhancedSeasonalColorDatabase:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.scaler = StandardScaler()
        
    def extract_enhanced_features(self, image_path):
        """
        Extract enhanced color features from an image
        """
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
        """Extract detailed eye color features"""
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
        """Extract detailed skin color features"""
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
        """Extract detailed hair color features"""
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
        """Extract contrast and relationship features"""
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
        """Helper function to get eye region from landmarks"""
        eye_points = np.array(eye_points)
        x, y, w, h = cv2.boundingRect(eye_points)
        return image[y:y+h, x:x+w]

class EnhancedSeasonalColorModel:
    def __init__(self, csv_path='season_db.csv'):
        self.df = pd.read_csv(csv_path)
        self.models = {}
        self.feature_importance = None
        self.best_features = None
        
    def preprocess_data(self):
        """Enhanced preprocessing with feature selection and robust scaling"""
        # Separate majority and minority classes with SMOTE-like approach
        classes = self.df['season'].value_counts()
        majority_class = classes.index[0]
        n_samples = int(classes[majority_class] * 1.5)  # Increase samples
        
        balanced_dfs = []
        for season in classes.index:
            season_df = self.df[self.df['season'] == season]
            if len(season_df) < n_samples:
                upsampled_df = resample(
                    season_df,
                    replace=True,
                    n_samples=n_samples,
                    random_state=42
                )
                balanced_dfs.append(upsampled_df)
            else:
                balanced_dfs.append(season_df)
        
        self.df = pd.concat(balanced_dfs)
        
        # Remove highly correlated features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['image_file', 'season']]
        correlation_matrix = self.df[feature_cols].corr().abs()
        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        to_drop = [column for column in upper.columns 
                   if any(upper[column] > 0.95)]
        self.df = self.df.drop(to_drop, axis=1)
        
        return self.df
    
    def select_features(self, X, y):
        """Select most important features using multiple methods"""
        # Random Forest for feature importance
        rf_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=42),
            max_features=40
        )
        rf_selector.fit(X, y)
        
        # Recursive Feature Elimination
        rfe_selector = RFE(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            n_features_to_select=40
        )
        rfe_selector.fit(X, y)
        
        # Combine selected features
        selected_features_rf = X.columns[rf_selector.get_support()].tolist()
        selected_features_rfe = X.columns[rfe_selector.support_].tolist()
        self.best_features = list(set(selected_features_rf + selected_features_rfe))
        
        return X[self.best_features]
    
    def create_ensemble(self):
        """Create a voting ensemble of multiple models"""
        models = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight='balanced'
            )),
            ('gb', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8
            )),
            ('ada', AdaBoostClassifier(
                estimator=None,
                n_estimators=200,
                learning_rate=0.05
            ))
        ]
        
        return VotingClassifier(
            estimators=models,
            voting='soft',
            weights=[2, 1, 2, 1]
        )
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train an improved ensemble model with advanced parameter tuning"""
        # Preprocess data
        self.preprocess_data()
        
        # Prepare features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['image_file', 'season']]
        X = self.df[feature_cols]
        y = self.df['season']
        
        # Feature selection
        X_selected = self.select_features(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Create pipeline with robust scaler
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', self.create_ensemble())
        ])
        
        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=5, scoring='accuracy'
        )
        print("\nCross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())
        print("CV score std:", cv_scores.std())
        
        # Fit the model
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        
        # Print results
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y)
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Season')
        plt.xlabel('Predicted Season')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Feature importance visualization
        self.plot_feature_importance(X_selected.columns, pipeline)
        
        # Save model
        joblib.dump(pipeline, 'seasonal_color_model_enhanced.joblib')
        
        return pipeline
    
    def plot_feature_importance(self, feature_names, pipeline):
        """Plot feature importance for the ensemble model"""
        plt.figure(figsize=(15, 8))
        
        # Get feature importance from the first model in ensemble
        if hasattr(pipeline.named_steps['classifier'], 'estimators_'):
            importance = pipeline.named_steps['classifier'].estimators_[0].feature_importances_
        else:
            importance = pipeline.named_steps['classifier'].feature_importances_
            
        indices = np.argsort(importance)[::-1]
        
        plt.title("Feature Importance")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(
            range(len(importance)), 
            [feature_names[i] for i in indices], 
            rotation=45, 
            ha='right'
        )
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

def main():
    # Create enhanced database
    print("Creating enhanced database...")
    db_creator = EnhancedSeasonalColorDatabase("/Users/nithyapandurangan/Desktop/colour-analysis-tool/flask/Dataset/Seasons/")
    # Process images and create features
    data = []
    print(f"Looking for images in: {db_creator.image_directory}")

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
    # Train the enhanced model
    print("\nTraining the enhanced model...")
    model = EnhancedSeasonalColorModel('season_db.csv')
    trained_model = model.train_model()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    print(model.feature_importance.head(10))
    
    print("\nModel training completed!")

if __name__ == "__main__":
    main()