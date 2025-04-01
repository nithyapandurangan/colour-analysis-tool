from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/recommend": {"origins": "*"}})

# Load dataset
file_path = "/Users/nithyapandurangan/Documents/colour-analysis-tool/flask/SkinCareRec/MP-Skin-Care-Product-Recommendation-System3.csv"

try:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower().str.strip()  
    df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x) 
    print("CSV file loaded successfully.")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    df = None  

# Define mapping for user concerns to equivalent database terms
concern_mapping = {
    "hyperpigmentation": ["brightening", "dark spot", "even tone"],
    "dryness": ["moisturizing", "hydrating"],
    "acne": ["anti-acne", "acne", "acne-free", "blemish control", "oil control"],
    "wrinkles": ["anti-aging", "firming", "lifting"],
    "sensitivity": ["soothing", "calming", "gentle"],
}

@app.route('/recommend', methods=['GET'])
def recommend_product():
    if df is None:
        return jsonify({"error": "Failed to load product dataset"}), 500

    # Get user inputs from query parameters
    skin_type = request.args.get("skin_type", "").strip().lower()
    concern = request.args.get("concern", "").strip().lower()
    product_type = request.args.get("product_type", "").strip().lower()

    print(f"Received parameters: skin_type={skin_type}, concern={concern}, product_type={product_type}")

    # Ensure required columns exist in the dataset
    required_columns = {'skintype', 'notable_effects', 'product_type', 'product_name', 'brand', 'price', 'picture_src'}
    if not required_columns.issubset(df.columns):
        return jsonify({"error": "Dataset is missing required columns"}), 500

    # Get mapped keywords for concern
    concern_keywords = concern_mapping.get(concern, [concern])

    # Create a filter condition
    filtered_df = df[
        (df['skintype'].str.lower().str.contains(skin_type, na=False)) &
        (df['product_type'].str.lower().str.contains(product_type, na=False)) &
        (df['notable_effects'].str.lower().apply(lambda x: any(keyword in x for keyword in concern_keywords)))
    ]

    # Select required columns including image URL
    selected_columns = ['product_name', 'product_type', 'brand', 'notable_effects', 'skintype', 'price', 'picture_src']
    filtered_df = filtered_df[selected_columns]

    print(f"Filtered products count: {len(filtered_df)}")

    # Convert to JSON and return response
    if not filtered_df.empty:
        return jsonify(filtered_df.to_dict(orient="records"))
    else:
        return jsonify({"message": "No matching products found. Try adjusting your filters."}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001)  
