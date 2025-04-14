import pandas as pd
import numpy as np
import cv2
import face_recognition
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import joblib
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial

# Feature boosting functions (defined at module level for pickling)
def boost_features_standard(X):
    return boost_features(X, 5.0, 3.0, 1.0, 0.6)

def boost_features_strong_skin(X):
    return boost_features(X, 4.5, 3.2, 0.8, 0.5)

def boost_features_balanced(X):
    return boost_features(X, 5.5, 2.8, 1.2, 0.7)

def boost_features(X, skin_mult=5.0, eye_mult=3.0, hair_mult=1.0, global_mult=0.6):
    """Boost features with seasonal weights"""
    X = X.copy()
    if isinstance(X, pd.DataFrame):
        skin_cols = [col for col in X.columns if 'skin_' in col]
        eye_cols = [col for col in X.columns if 'eye_' in col]
        hair_cols = [col for col in X.columns if 'hair_' in col]
        global_cols = [col for col in ['contrast', 'global_warmth', 'color_clarity', 'depth_score'] if col in X.columns]
        
        X[skin_cols] = X[skin_cols] * skin_mult
        X[eye_cols] = X[eye_cols] * eye_mult
        X[hair_cols] = X[hair_cols] * hair_mult
        if global_cols:
            X[global_cols] = X[global_cols] * global_mult
    return X

class SeasonalColorAnalyzer:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def _get_eye_region(self, image, eye_points):
        eye_points = np.array(eye_points)
        x, y, w, h = cv2.boundingRect(eye_points)
        return image[y:y+h, x:x+w]
    
    def _extract_eye_features(self, image, image_lab, image_hsv, landmarks):
        features = {}
        
        for eye_name, eye_points in [('left_eye', landmarks['left_eye']), 
                                   ('right_eye', landmarks['right_eye'])]:
            eye_region = self._get_eye_region(image, eye_points)
            eye_region_lab = self._get_eye_region(image_lab, eye_points)
            eye_region_hsv = self._get_eye_region(image_hsv, eye_points)
            
            bgr_mean = cv2.mean(eye_region)[:3]
            lab_mean = cv2.mean(eye_region_lab)[:3]
            hsv_mean = cv2.mean(eye_region_hsv)[:3]
            
            features.update({
                f'{eye_name}_b': bgr_mean[0],
                f'{eye_name}_g': bgr_mean[1],
                f'{eye_name}_r': bgr_mean[2],
                f'{eye_name}_l': lab_mean[0],
                f'{eye_name}_a': lab_mean[1],
                f'{eye_name}_b_lab': lab_mean[2],
                f'{eye_name}_h': hsv_mean[0],
                f'{eye_name}_s': hsv_mean[1],
                f'{eye_name}_v': hsv_mean[2]
            })
        
        left_eye = self._get_eye_region(image, landmarks['left_eye'])
        right_eye = self._get_eye_region(image, landmarks['right_eye'])
        
        features.update({
            'eye_contrast': np.std(cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)) + 
                          np.std(cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)),
            'eye_brightness': (features['left_eye_l'] + features['right_eye_l']) / 2,
            'eye_clarity': (features['left_eye_s'] + features['right_eye_s']) / 2,
            'eye_warmth': (features['left_eye_a'] + features['right_eye_a']) / 2,
            'eye_undertone': (features['left_eye_b_lab'] + features['right_eye_b_lab']) / 2
        })
        
        return features
    
    def _extract_skin_features(self, face, face_lab, face_hsv, landmarks):
        features = {}
        regions = {
            'forehead': (0.2, 0.3, 0.8, 0.4),
            'left_cheek': (0.5, 0.2, 0.7, 0.35),
            'right_cheek': (0.5, 0.65, 0.7, 0.8),
            'chin': (0.8, 0.4, 0.9, 0.6)
        }
        
        all_a = []
        all_b = []
        all_s = []
        
        for region_name, (y1, x1, y2, x2) in regions.items():
            h, w = face.shape[:2]
            region = face[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            region_lab = face_lab[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            region_hsv = face_hsv[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]
            
            if region.size > 0:
                lab_mean = cv2.mean(region_lab)[:3]
                hsv_mean = cv2.mean(region_hsv)[:3]
                
                features.update({
                    f'skin_{region_name}_l': lab_mean[0],
                    f'skin_{region_name}_a': lab_mean[1],
                    f'skin_{region_name}_b_lab': lab_mean[2],
                    f'skin_{region_name}_h': hsv_mean[0],
                    f'skin_{region_name}_s': hsv_mean[1],
                    f'skin_{region_name}_v': hsv_mean[2]
                })
                
                all_a.append(lab_mean[1])
                all_b.append(lab_mean[2])
                all_s.append(hsv_mean[1])
        
        if all_a and all_b and all_s:
            avg_a = np.mean(all_a)
            avg_b = np.mean(all_b)
            avg_s = np.mean(all_s)
            
            features.update({
                'skin_warmth': avg_a + (avg_b * 0.5),
                'skin_temperature': avg_a + avg_b,
                'skin_clarity': avg_s,
                'skin_depth': 255 - np.mean([features.get(f'skin_{r}_l', 0) 
                                         for r in regions.keys() 
                                         if f'skin_{r}_l' in features])
            })
        
        features.update({
            'global_warmth': np.mean(face_lab[:,:,1]) - np.mean(face_lab[:,:,2]),
            'color_clarity': np.mean(face_hsv[:,:,1]) * np.std(face_hsv[:,:,2]),
            'depth_score': np.mean(face_lab[:,:,0]) * (1 - np.mean(face_hsv[:,:,1]))
        })
        
        return features
    
    def _extract_hair_features(self, image, image_lab, image_hsv, top, left, right):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
            
        x, y, w, h = faces[0]
        hair_region = image[max(0, y-int(h*0.5)):max(0, y-int(h*0.1)), 
                          max(0, x-int(w*0.3)):min(image.shape[1], x+w+int(w*0.3))]
        
        if hair_region.size == 0:
            return None
            
        lab_region = cv2.cvtColor(hair_region, cv2.COLOR_BGR2LAB)
        hsv_region = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
        
        lab_mean = np.mean(lab_region, axis=(0,1))
        hsv_mean = np.mean(hsv_region, axis=(0,1))
        
        return {
            'hair_l_mean': lab_mean[0],
            'hair_a_mean': lab_mean[1],
            'hair_b_mean': lab_mean[2],
            'hair_h_mean': hsv_mean[0],
            'hair_s_mean': hsv_mean[1],
            'hair_v_mean': hsv_mean[2]
        }
    
    def extract_features(self, image_path):
        try:
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            
            if not face_locations:
                return None
                
            image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
            image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            
            face_landmarks = face_recognition.face_landmarks(image, face_locations)
            if not face_landmarks:
                return None
                
            landmarks = face_landmarks[0]
            top, right, bottom, left = face_locations[0]
            face = image_cv[top:bottom, left:right]
            face_lab = image_lab[top:bottom, left:right]
            face_hsv = image_hsv[top:bottom, left:right]
            
            features = {}
            features.update(self._extract_eye_features(image_cv, image_lab, image_hsv, landmarks))
            features.update(self._extract_skin_features(face, face_lab, face_hsv, landmarks))
            
            hair_features = self._extract_hair_features(image_cv, image_lab, image_hsv, top, left, right)
            if hair_features:
                features.update(hair_features)
            
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            features['contrast'] = np.std(gray)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None


class SeasonalColorModel:
    def __init__(self, csv_path='season_db.csv'):
        self.df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
        self.model = None
    
    def create_database(self, image_directory):
        analyzer = SeasonalColorAnalyzer(image_directory)
        data = []
        
        for image_file in os.listdir(image_directory):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                season_match = re.match(r'([a-zA-Z_]+)_\d+', image_file)
                if season_match:
                    season = season_match.group(1)
                    image_path = os.path.join(image_directory, image_file)
                    features = analyzer.extract_features(image_path)
                    
                    if features:
                        features['image_file'] = image_file
                        features['season'] = season
                        data.append(features)
        
        self.df = pd.DataFrame(data)
        self.df.to_csv('season_db.csv', index=False)
        return self.df
    
   
    def train_model(self, test_size=0.2, random_state=42):
        if self.df is None:
            raise ValueError("No data available. Call create_database() first.")
            
        # Balance classes
        classes = self.df['season'].value_counts()
        majority_size = classes.max()
        balanced_dfs = []
        
        for season in classes.index:
            season_df = self.df[self.df['season'] == season]
            if len(season_df) < majority_size:
                season_df = resample(season_df, replace=True, 
                                   n_samples=majority_size, 
                                   random_state=random_state)
            balanced_dfs.append(season_df)
            
        balanced_df = pd.concat(balanced_dfs)
        X = balanced_df.drop(['season', 'image_file'], axis=1)
        y = balanced_df['season']
        
        # Feature transformations
        scaler = RobustScaler()
        boosted_X = boost_features(X)
        scaled_X = scaler.fit_transform(boosted_X)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        scaled_X_imputed = imputer.fit_transform(scaled_X)

        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(scaled_X_imputed)
        n_components = X_pca.shape[1]
        print(f"Number of components after PCA: {n_components}")
        
        # Adjusted feature selection parameters
        selector_max_features = min(15, n_components)
        min_features = max(1, selector_max_features-5)
        
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('feature_booster', FunctionTransformer(func=boost_features_standard)),
            ('pca', PCA(n_components=0.95)),
            ('selector', SelectFromModel(
                ExtraTreesClassifier(n_estimators=100, max_features='sqrt', random_state=random_state),
                max_features=selector_max_features)),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            ))
        ])
        
        param_dist = {
            'feature_booster__func': [
                boost_features_standard,
                boost_features_strong_skin,
                boost_features_balanced
            ],
            'selector__max_features': np.arange(
                min_features,
                min(selector_max_features+1, n_components)
            ).tolist(),
            'classifier__max_depth': [10, 12, 15]
        }

        search = RandomizedSearchCV(
            pipeline, 
            param_distributions=param_dist,
            n_iter=20,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_state,
            verbose=2
        )

        print("Training model with traditional importance constraints...")
        search.fit(X, y)
        self.model = search.best_estimator_
        
        # Evaluation and feature analysis
        self._evaluate_model(X, y)
        self._analyze_feature_importance(X.columns)
        self._show_top_features(X.columns)
        
        return self.model
    
    def _get_feature_importance(self, model, feature_names):
        if not hasattr(model.named_steps['classifier'], 'feature_importances_'):
            return {}
            
        importance = model.named_steps['classifier'].feature_importances_
        pca = model.named_steps['pca']
        selector = model.named_steps['selector']
        
        # Map importance back to original features
        original_importance = np.zeros(len(feature_names))
        selected_mask = selector.get_support()
        
        for i, (is_selected, imp) in enumerate(zip(selected_mask, importance)):
            if is_selected:
                original_importance += imp * np.abs(pca.components_[i])
        
        return original_importance
    
    def _analyze_feature_importance(self, feature_names):
        importance = self._get_feature_importance(self.model, feature_names)
        
        # Group importance
        groups = {
            'Skin': [i for i, f in enumerate(feature_names) if 'skin_' in f],
            'Eyes': [i for i, f in enumerate(feature_names) if 'eye_' in f],
            'Hair': [i for i, f in enumerate(feature_names) if 'hair_' in f],
            'Global': [i for i, f in enumerate(feature_names) 
                      if f in ['contrast', 'global_warmth', 'color_clarity', 'depth_score']]
        }
        
        total = sum(importance)
        group_importance = {g: sum(importance[indices])/total for g, indices in groups.items() if indices}
        
        print("\nFinal Feature Importance Distribution:")
        print(f"Skin: {group_importance.get('Skin', 0):.2%} (Target: 40-50%)")
        print(f"Eyes: {group_importance.get('Eyes', 0):.2%} (Target: 25-35%)")
        print(f"Hair: {group_importance.get('Hair', 0):.2%} (Target: 15-25%)")
        print(f"Global: {group_importance.get('Global', 0):.2%} (Target: 5-15%)")
        
        # Plot group importance
        plt.figure(figsize=(10, 6))
        groups_list = ['Skin', 'Eyes', 'Hair', 'Global']
        values = [group_importance.get(g, 0) for g in groups_list]
        targets = [0.45, 0.30, 0.20, 0.10]
        
        x = range(len(groups_list))
        plt.bar(x, values, width=0.4, label='Actual')
        plt.bar([i + 0.4 for i in x], targets, width=0.4, label='Target')
        
        plt.xticks([i + 0.2 for i in x], groups_list)
        plt.title("Feature Importance vs Target Distribution")
        plt.ylabel("Importance Percentage")
        plt.legend()
        plt.ylim(0, 0.6)
        plt.tight_layout()
        plt.show()
                           

    def _show_top_features(self, feature_names):
    # Get the PCA components and selector
        pca = self.model.named_steps['pca']
        selector = self.model.named_steps['selector']
        classifier = self.model.named_steps['classifier']
    
        # Get which PCA components were selected
        pca_selected_mask = selector.get_support()
        selected_indices = np.where(pca_selected_mask)[0]
        n_selected = len(selected_indices)
    
    # Calculate importance scores for original features
        importance_scores = np.zeros(len(feature_names))
    
    # Map selected PCA components to classifier feature importances
        for i, pca_idx in enumerate(selected_indices):
            if i < len(classifier.feature_importances_):
            # Add the weighted component importance to original features
                importance_scores += np.abs(pca.components_[pca_idx]) * classifier.feature_importances_[i]
    
    # Create DataFrame of feature importances
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
    
    # Get the top selected features (those with highest importance scores)
        selected_features = feature_imp.head(n_selected)['feature'].tolist()
    
        print(f"\nSelected {n_selected} important features out of {len(feature_names)}")
        print("Selected features:", selected_features)
    
        print("\nTop 20 Feature Importances:")
        for i, row in enumerate(feature_imp.head(20).itertuples(), 1):
            print(f"{i}. {row.feature}: {row.importance:.4f}")
    
    # Plot top features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', 
               data=feature_imp.head(20),
               palette='viridis')
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        plt.show()
        
    
    def _evaluate_model(self, X, y):
        y_pred = self.model.predict(X)
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=np.unique(y),
                   yticklabels=np.unique(y))
        plt.title('Confusion Matrix')
        plt.ylabel('True Season')
        plt.xlabel('Predicted Season')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def predict(self, image_path):
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        analyzer = SeasonalColorAnalyzer(os.path.dirname(image_path))
        features = analyzer.extract_features(image_path)
        
        if features is None:
            return "Unknown", {}
            
        features_df = pd.DataFrame([features])
        missing_cols = set(self.model.feature_names_in_) - set(features_df.columns)
        
        for col in missing_cols:
            features_df[col] = 0
            
        features_df = features_df[self.model.feature_names_in_]
        season = self.model.predict(features_df)[0]
        proba = self.model.predict_proba(features_df)[0]
        
        return season, dict(zip(self.model.classes_, proba))

def main():
    analyzer = SeasonalColorModel()
    
    if not os.path.exists('season_db.csv'):
        print("Creating feature database...")
        analyzer.create_database(r"/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/Dataset/Seasons")
    
    print("Training model with optimal feature weights...")
    analyzer.train_model()
    
    joblib.dump(analyzer.model, 'seasonal_colour_model.joblib')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    main()
