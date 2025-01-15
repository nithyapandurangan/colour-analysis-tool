import React, { useState } from 'react';

// Product Recommendations Component
const ProductRecommendations = ({ seasonTag }) => {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch product recommendations based on season
  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/get_recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ season: seasonTag }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      setProducts(data.recommended_products || []);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError(error.message || 'An error occurred while fetching recommendations');
    } finally {
      setLoading(false);
    }
  };

  // Fetch recommendations when component loads
  React.useEffect(() => {
    if (seasonTag) {
      fetchRecommendations();
    }
  }, [seasonTag]);

  return (
    <div className="mt-8 flex flex-col items-center">
      <h2 className="text-2xl font-bold mb-4 text-center">Recommended Products for {seasonTag}</h2>
      {loading && <p>Loading recommendations...</p>}
      {error && <p className="text-red-500">{error}</p>}

      {/* Center the product grid, 2 items per row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-5xl justify-items-center">
        {products.length > 0 ? (
          products.map((product, index) => (
            <div
              key={index}
              className="border p-4 rounded-lg shadow-lg w-60 flex flex-col items-center"  
            >
              <h3 className="font-bold text-lg text-center">{product.title}</h3>
              {product.image_url && (
                <img
                  src={product.image_url}
                  alt={product.title}
                  className="w-full h-40 object-cover mb-4 rounded-lg"
                />
              )}
              <a
                href={product.product_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-500 hover:underline"
              >
                View Product
              </a>
            </div>
          ))
        ) : (
          <p>No products found for this season.</p>
        )}
      </div>
    </div>
  );
};

const Tool = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showRecommendations, setShowRecommendations] = useState(false);

  // Handle image upload
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setResult(null);
      setShowRecommendations(false);
    }
  };

  // Analyze color using backend API
  const analyzeColor = async () => {
    if (!selectedImage) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Error analyzing image');
      }
      const data = await response.json();
      setResult(data.predicted_season);
    } catch (error) {
      console.error('Error analyzing image:', error);
      setResult('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Function to format the season name for the palette image
  const getPaletteImagePath = (season) => {
    if (!season) return null;
    const formattedSeason = season.toLowerCase().replace(/\s+/g, '_');
    return `${process.env.PUBLIC_URL}/Palettes/${formattedSeason}_wc.png`;
  };

  return (
    <div className="min-h-screen bg-white p-8 flex flex-col items-center">
      <h1 className="text-4xl font-extrabold text-black mb-8 text-center">
        Our Colour Analysis Tool
      </h1>

      <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-3xl">
        {/* File Upload Section */}
        <div className="flex flex-col items-center mb-6">
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="mb-4 p-4 border-2 border-indigo-500 rounded-lg cursor-pointer"
          />
          <p className="text-gray-500">Upload an image for colour analysis & clothing recommendation</p>
        </div>

        {/* Image Preview and Analysis Section */}
        {selectedImage && (
          <div className="flex flex-col items-center space-y-4 mb-6">
            <h2 className="text-xl font-semibold">Uploaded Image:</h2>
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="Uploaded"
              className="max-w-sm rounded-lg shadow-lg"
            />
            <button
              onClick={analyzeColor}
              disabled={!selectedImage || loading}
              className="mt-4 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Analyzing...' : 'Analyze Colour'}
            </button>
          </div>
        )}

        {/* Results Section */}
        {result && (
          <div className="flex flex-col items-center space-y-6 mt-2">
            <h2 className="text-2xl font-bold text-black">Your Predicted Season:</h2>
            <p className="text-2xl font-semibold text-p">{result}</p>

            {/* Display Color Palette */}
            <div className="flex flex-col items-center">
              <h3 className="text-2xl font-bold mb-4">Your Seasonal Colour Palette:</h3>
              <img
                src={getPaletteImagePath(result)}
                alt={`${result} Colour Palette`}
                className="w-72 h-auto rounded-lg shadow-lg"
                onError={(e) => {
                  e.target.src = `${process.env.PUBLIC_URL}/Palettes/dark_autumn_wc.png`;
                }}
              />
            </div>

            {/* View Recommended Products Button */}
            {!showRecommendations && (
              <button
                onClick={() => setShowRecommendations(true)}
                className="mt-4 px-6 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-semibold transition-colors"
              >
                View Recommended Products
              </button>
            )}

            {/* Display Recommended Products */}
            {showRecommendations && <ProductRecommendations seasonTag={result} />}
          </div>
        )}
      </div>
    </div>
  );
};

export default Tool;
