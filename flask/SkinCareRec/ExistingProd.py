from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the skincare data
skincare_url = 'https://drive.google.com/uc?export=download&id=19LD3liEpL9ZRi77evf5cK-icR0w_qXCx'
skincare = pd.read_csv(skincare_url, low_memory=False)

# Clean and prepare data
skincare['product_name'] = skincare['product_name'].str.lower().str.strip()  # Normalize product names
skincare['notable_effects'] = skincare['notable_effects'].fillna('')  # Handle NaN values
skincare['price'] = skincare['price'].fillna("Price not available")  # Handle missing prices
skincare['picture_src'] = skincare['picture_src'].fillna("")  # Ensure empty strings for missing images

# Initialize and fit TF-IDF on 'notable_effects'
tf = TfidfVectorizer()
tf_matrix = tf.fit_transform(skincare['notable_effects'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tf_matrix)

# Function to get product recommendations
def get_recommendations(product_name, n=5):
    product_name = product_name.lower().strip()  # Normalize input to lowercase

    if product_name not in skincare['product_name'].values:
        return []

    # Get index of the product
    idx_list = skincare.index[skincare['product_name'] == product_name].tolist()
    
    if not idx_list:
        return []
    
    idx = idx_list[0]  # Use first match if multiple

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of similar products
    sim_indices = [i[0] for i in sim_scores[1:n+1]]

    # Return top n recommendations
    return skincare.iloc[sim_indices][['product_name', 'brand', 'notable_effects', 'picture_src', 'price']].to_dict(orient='records')

@app.route('/recommend', methods=['GET'])
def recommend():
    product_name = request.args.get('existing_product', '').strip()

    if not product_name:
        return jsonify({'error': 'No product name provided'}), 400

    recommendations = get_recommendations(product_name)

    if not recommendations:
        return jsonify({'error': 'No matching products found'}), 404

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
