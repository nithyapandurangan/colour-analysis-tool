# Colour Analysis Tool

The Colour Analysis Tool is an AI-powered platform that provides personalized colour recommendations based on a user's facial features such as skin tone, hair colour, and eye colour. The tool classifies users into seasonal colour categories and suggests clothing and accessories that complement their unique features. It is designed to help individuals discover their ideal colour palette and elevate their style.

<img width="768" height="768" alt="image" src="https://github.com/user-attachments/assets/af597b2d-126e-4a01-a8ce-3d1aa4ba1491" />

## Features

- **AI-Powered colour Analysis:** The tool uses advanced AI algorithms to analyze the user's facial features and provide accurate colour recommendations. It improves over time by incorporating user feedback.
- **Personalized Seasonal colour Classification:** Based on the analysis, the tool classifies users into one of the seasonal categories (Spring, Summer, Autumn, Winter), which helps identify colors that suit them best.
- **Tailored Product Recommendations:** Users are provided with personalized clothing and accessory suggestions from the catalog, based on their seasonal classification and colour palette.
- **Enhanced User Experience:** Users can upload their photos and instantly receive their personalized colour analysis and seasonal classification.
  
## Technologies Used

- **React.js:** For building the front-end of the application.
- **Framer Motion:** For smooth animations and transitions.
- **Tailwind CSS:** For styling the UI components.
- **Shopify API:** For integrating product recommendations from a catalog of dresses and accessories.
- **Machine Learning Model:**
  - **RandomForestClassifier:** Classifies seasons based on extracted features.
  - **PCA:** Reduces dimensionality while preserving key information.
  - **Hyperparameter Tuning:** Optimized with `RandomizedSearchCV`.

- **Feature Extraction:**
  - **OpenCV:** Image processing (colour space conversion, face detection).
  - **Face Recognition:** Detects facial features (eyes, skin, hair).
  - **Skimage:** Advanced image processing (colour conversion, histogram equalization).

- **Data Processing & Model Training:**
  - **Pandas:** Manages and processes datasets.
  - **Scikit-learn:** Model training, preprocessing, and evaluation.

- **Visualization:**
  - **Matplotlib & Seaborn:** Visualizes model performance (confusion matrix).

- **Database & File Management:**
  - **Joblib:** Saves the trained model for future predictions.
  - **Resampling:** Balances the dataset by upsampling minority classes.
  
## How It Works

1. **User Upload:** The user uploads a clear photo of their face.
2. **AI Analysis:** The AI model analyzes the uploaded photo to identify key facial features like skin tone, hair colour, and eye colour.
3. **colour Classification:** Based on these features, the tool classifies the user into one of the four seasonal colour categories (Spring, Summer, Autumn, Winter).
4. **Personalized Palette:** The user receives a personalized colour palette that includes colors that best complement their features.
5. **Product Recommendations:** Based on the user's seasonal classification, the tool suggests products (clothing and accessories) from the catalog that align with their personalized colour palette.

## Installation

To run the Colour Analysis Tool locally, follow the instructions below:

1. Clone the repository:

    ```bash
    git clone https://github.com/nithyapandurangan/colour-analysis-tool.git
    ```

2. Navigate into the project directory:

    ```bash
    cd colour-analysis-tool
    ```

3. Install the dependencies:

    ```bash
    npm install
    ```

4. Run the project locally:

    ```bash
    npm start
    ```

5. Open your browser and go to `http://localhost:3000` to view the tool in action.

## Usage

1. Go to the website and upload an image of your face.
2. Wait for the tool to analyze your facial features.
3. Receive your personalized seasonal colour classification and colour palette.
4. Browse clothing and accessory recommendations tailored to your unique colour profile.

## Contributing

If you'd like to contribute to the development of the Colour Analysis Tool, feel free to fork the repository, create a new branch, and submit a pull request.
