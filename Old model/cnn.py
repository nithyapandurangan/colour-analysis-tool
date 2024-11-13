import pandas as pd
import numpy as np
import cv2
import face_recognition
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re

class SeasonalColorDatabase:
    def __init__(self, image_directory):
        self.image_directory = image_directory
        
    def extract_features(self, image_path):
        """
        Extract eye, hair, and skin colors from an image
        """
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return None
            
        # Convert to BGR
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Get the first face
        top, right, bottom, left = face_locations[0]
        face = image_cv[top:bottom, left:right]
        
        features = {}
        
        # Extract eye region
        face_landmarks = face_recognition.face_landmarks(image, face_locations)
        if face_landmarks:
            landmarks = face_landmarks[0]
            
            # Get eye regions using landmarks
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            # Calculate eye regions
            left_eye_region = self._get_eye_region(image_cv, left_eye)
            right_eye_region = self._get_eye_region(image_cv, right_eye)
            
            # Average color of both eyes
            left_color = cv2.mean(left_eye_region)[:3]
            right_color = cv2.mean(right_eye_region)[:3]
            eye_color = tuple(np.mean([left_color, right_color], axis=0))
            
            features.update({
                'eye_b': eye_color[0],
                'eye_g': eye_color[1],
                'eye_r': eye_color[2]
            })
        
        # Extract skin color from multiple face regions
        skin_regions = [
            face[int(face.shape[0]*0.3):int(face.shape[0]*0.4), 
                 int(face.shape[1]*0.3):int(face.shape[1]*0.7)],  # Forehead
            face[int(face.shape[0]*0.5):int(face.shape[0]*0.7), 
                 int(face.shape[1]*0.2):int(face.shape[1]*0.4)],  # Left cheek
            face[int(face.shape[0]*0.5):int(face.shape[0]*0.7), 
                 int(face.shape[1]*0.6):int(face.shape[1]*0.8)]   # Right cheek
        ]
        
        skin_colors = [cv2.mean(region)[:3] for region in skin_regions if region.size > 0]
        if skin_colors:
            skin_color = tuple(np.mean(skin_colors, axis=0))
            features.update({
                'skin_b': skin_color[0],
                'skin_g': skin_color[1],
                'skin_r': skin_color[2]
            })
        
        # Extract hair color from top of head
        hair_region = image_cv[max(0, top-30):top, left:right]
        if hair_region.size > 0:
            hair_color = cv2.mean(hair_region)[:3]
            features.update({
                'hair_b': hair_color[0],
                'hair_g': hair_color[1],
                'hair_r': hair_color[2]
            })
        
        return features
    
    def _get_eye_region(self, image, eye_points):
        """Helper function to get eye region from landmarks"""
        eye_points = np.array(eye_points)
        x, y, w, h = cv2.boundingRect(eye_points)
        return image[y:y+h, x:x+w]
    
    def create_database(self):
        """
        Process all images and create season_db.csv
        """
        data = []
        
        # Get all image files
        image_files = [f for f in os.listdir(self.image_directory) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in image_files:
            # Extract season from filename (e.g., dark_autumn_8.jpg -> dark_autumn)
            season_match = re.match(r'([a-zA-Z_]+)_\d+', image_file)
            if not season_match:
                continue
                
            season = season_match.group(1)
            image_path = os.path.join(self.image_directory, image_file)
            
            features = self.extract_features(image_path)
            if features:
                features['image_file'] = image_file
                features['season'] = season
                data.append(features)
                print(f"Processed {image_file}")
            else:
                print(f"No face detected in {image_file}")
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv('season_db.csv', index=False)
        print(f"\nDatabase created with {len(df)} entries")
        return df

class SeasonalColorModel:
    def __init__(self, csv_path='season_db.csv'):
        self.df = pd.read_csv(csv_path)
        self.model = None
        
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train the seasonal color classification model
        """
        # Prepare features
        feature_cols = [col for col in self.df.columns 
                       if col not in ['image_file', 'season']]
        X = self.df[feature_cols]
        y = self.df['season']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Print results
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.model.classes_,
                   yticklabels=self.model.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Season')
        plt.xlabel('Predicted Season')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Feature importance
        plt.figure(figsize=(10, 6))
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=importances, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        return self.model
        
    def predict_season(self, features):
        """
        Predict season for new features
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        feature_cols = [col for col in self.df.columns 
                       if col not in ['image_file', 'season']]
        prediction = self.model.predict([features[col] for col in feature_cols])
        return prediction[0]

def main():
    # Step 1: Create database
    print("Creating database...")
    db_creator = SeasonalColorDatabase("Dataset/Seasons/")
    df = db_creator.create_database()
    
    # Step 2: Train model
    print("\nTraining model...")
    model_trainer = SeasonalColorModel()
    model = model_trainer.train_model()
    
    print("\nModel training complete. Results saved in:")
    print("- confusion_matrix.png")
    print("- feature_importance.png")

if __name__ == "__main__":
    main()