import React from 'react';
import ProductRecommendations from './ProductRecommendations';

const ResultDisplay = ({ result, showRecommendations, setShowRecommendations }) => {
  const getPaletteImagePath = (season) => {
    if (!season) return null;
    const formattedSeason = season.toLowerCase().replace(/\s+/g, '_');
    return `${process.env.PUBLIC_URL}/Palettes/${formattedSeason}_wc.jpg`;
  };

  return (
    <div className="flex flex-col items-center space-y-6 mt-2">
      <h2 className="text-2xl font-bold text-black">Your Predicted Season:</h2>
      <p className="text-2xl font-semibold text-p">{result}</p>
      <div className="flex flex-col items-center">
        <h3 className="text-2xl font-bold mb-4">Your Seasonal Colour Palette:</h3>
        <img
          src={getPaletteImagePath(result)}
          alt={`${result} Colour Palette`}
          className="w-72 h-auto rounded-lg shadow-lg"
          onError={(e) => {
            e.target.src = `${process.env.PUBLIC_URL}/Palettes/dark_autumn_wc.jpg`;
          }}
        />
      </div>

      {!showRecommendations && (
        <button
          onClick={() => setShowRecommendations(true)}
          className="mt-4 px-6 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-semibold transition-colors"
        >
          View Recommended Products
        </button>
      )}

      {showRecommendations && <ProductRecommendations seasonTag={result} />}
    </div>
  );
};

export default ResultDisplay;
