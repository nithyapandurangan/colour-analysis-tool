from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from PIL import Image
import numpy as np
import pandas as pd
import io
import os
import requests
from MainModel import EnhancedSeasonalColorDatabase, ImprovedSeasonalColorModel

# Define paths
MODEL_PATH = '/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/colour_model.joblib'
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

# Helper function to format season names consistently
def format_season_name(season):
    """Format season name to match frontend expectations.
    
    Example: 'dark_winter' -> 'Dark Winter'
    """
    words = season.replace('_', ' ').split()
    formatted = ' '.join(word.capitalize() for word in words)
    print(f"Formatting season '{season}' to '{formatted}'")
    return formatted

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
        
        # Add the same engineered features as during training
        features['warm_cool_balance'] = (
            features['skin_forehead_a'] - 0.5 * features['skin_forehead_b_lab']
        )
        features['autumn_dark_warmth'] = (
            0.8 * features['skin_forehead_a'] +
            0.6 * features['skin_forehead_l'] +
            0.4 * features['contrast']
        )
        features['dark_season_indicator'] = (
            -0.7 * features['skin_forehead_l'] +
            -0.7 * features['hair_l'] +
            0.3 * features['contrast']
        )
        features['winter_coolness_index'] = (
            -0.6 * features['skin_forehead_a'] +
            -0.6 * features['skin_left_cheek_a'] +
            0.4 * features['winter_contrast_depth']
        )
        features['soft_season_indicator'] = (
            -0.7 * features['contrast'] +
            0.5 * features['autumn_mutedness'] +
            -0.4 * features['v_variance']
        )
        features['winter_features'] = (
            0.5 * features['hair_l'] +
            0.3 * features['contrast'] +
            0.2 * features['winter_coolness']
        )

        # Convert features to the expected input format
        feature_cols = list(features.keys())
        feature_values = pd.DataFrame([features], columns=feature_cols)
        
        # Make prediction using the trained model
        prediction = model.predict(feature_values)
        raw_season = prediction[0]
        
        # Format the season name to match frontend expectations
        predicted_season = format_season_name(raw_season)
        
        print(f"Raw predicted season: {raw_season}")
        print(f"Formatted predicted season: {predicted_season}")
        
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
    """Improved product recommendation endpoint with robust tag matching"""
    data = request.get_json()
    season_tag = data.get('season')
    
    if not season_tag:
        return jsonify({'error': 'Season tag is required'}), 400

    # Generate all possible tag variations
    formatted_tag = format_season_name(season_tag)  # "Dark Autumn"
    tag_variations = {
        'space': formatted_tag.lower(),          # "dark autumn"
        'underscore': formatted_tag.replace(' ', '_').lower(),  # "dark_autumn"
        'hyphen': formatted_tag.replace(' ', '-').lower()       # "dark-autumn"
    }

    headers = {
        "X-Shopify-Access-Token": access_token,
        "Content-Type": "application/json",
    }

    try:
        # Fetch all products (we'll filter locally)
        response = requests.get(f"{shop_url}/admin/api/2024-01/products.json", headers=headers)
        response.raise_for_status()  # Raises exception for 4XX/5XX status
        
        products = response.json().get('products', [])
        matched_products = []
        
        for product in products:
            product_tags = product.get('tags', '').lower()
            
            # Check if any variation matches
            if any(variation in product_tags for variation in tag_variations.values()):
                matched_products.append({
                    'title': product['title'],
                    'tags': product['tags'],
                    'image_url': product['images'][0]['src'] if product.get('images') else None,
                    'product_url': f"{shop_url}/products/{product['handle']}"
                })

        return jsonify({
            'recommended_products': matched_products,
            'meta': {
                'searched_tag': formatted_tag,
                'matched_variations': list(tag_variations.values()),
                'product_count': len(matched_products)
            }
        })

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Shopify API error: {str(e)}")
        return jsonify({
            'error': 'Failed to fetch products from Shopify',
            'details': str(e)
        }), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500
        
if __name__ == '__main__':
    app.run(debug=True, port=5000)
