import React, { useState } from 'react';
import ProductRecommendations from './ProductRecommendations';
import ColourPalette from './ColourPalette';

const ResultDisplay = ({ result, showRecommendations, setShowRecommendations }) => {
  const formatSeasonName = (season) => {
    return season.split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="flex flex-col items-center space-y-6 mt-2">
      <h2 className="text-2xl font-bold text-black">Your Predicted Season:</h2>
      <p className="text-2xl font-semibold text-pink-600">{formatSeasonName(result)}</p>
      
      <div className="flex flex-col items-center">
        <h3 className="text-2xl font-bold">Your Seasonal Colour Palette:</h3>
        <ColourPalette season={result} />
      </div>

      {showRecommendations && <ProductRecommendations seasonTag={result} />}
    </div>
  );
};

export default ResultDisplay;
