from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from PIL import Image
import numpy as np
import io
import os
import requests
from model1 import EnhancedSeasonalColorDatabase, ImprovedSeasonalColorModel

# Define paths
MODEL_PATH = '/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/seasonal_color_model.joblib'
IMAGE_DIR = '/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/Dataset/Seasons'

# Shopify store details
shop_url = "https://suit-you-well.myshopify.com"
access_token = "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" #add your own shopify token to this to fetch data from shopify

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load pre-trained model
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize database creator
db_creator = EnhancedSeasonalColorDatabase(IMAGE_DIR)

# Helper function to preprocess image
def process_image(image):
    """Preprocess the image before making predictions."""
    try:
        image = Image.open(io.BytesIO(image))
        image = image.resize((224, 224))  # Resize to model's input size
        image_array = np.array(image) / 255.0  # Normalize
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return "Colour Analysis Tool API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict seasonal color."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        image = file.read()
        
        # Extract enhanced features from the image
        temp_image_path = os.path.join(IMAGE_DIR, 'temp_image.jpg')
        with open(temp_image_path, 'wb') as temp_file:
            temp_file.write(image)
        
        features = db_creator.extract_enhanced_features(temp_image_path)
        if not features:
            return jsonify({'error': 'No face detected or could not extract features'}), 400
        
        # Convert features to the expected input format
        feature_cols = list(features.keys())
        feature_values = np.array([features[col] for col in feature_cols]).reshape(1, -1)
        
        # Make prediction using the trained model
        prediction = model.predict(feature_values)
        predicted_season = prediction[0]
        print(f"Predicted season: {predicted_season}") 
        
        # Remove temporary image
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        return jsonify({'predicted_season': predicted_season})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the image'}), 500

# New Endpoint: Fetch product recommendations based on season
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """API endpoint to fetch product recommendations for a given season."""
    data = request.get_json()
    season_tag = data.get('season')
    
    if not season_tag:
        return jsonify({'error': 'Season tag is required'}), 400

    # Construct the API endpoint to fetch products with the season tag
    url = f"{shop_url}/admin/api/2024-01/products.json?tag={season_tag}"
    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
    }

    try:
        # Send the request to fetch products
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            products = response.json().get('products', [])
            filtered_products = [
                {
                    'title': product['title'],
                    'tags': product['tags'],
                    'image_url': product['images'][0]['src'] if product['images'] else None,
                    'product_url': f"{shop_url}/products/{product['handle']}"
                }
                for product in products if season_tag in product['tags']
            ]
            
            return jsonify({'recommended_products': filtered_products}), 200
        else:
            return jsonify({'error': f"Failed to fetch products, status code: {response.status_code}"}), 500

    except Exception as e:
        app.logger.error(f"Error fetching recommendations: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
