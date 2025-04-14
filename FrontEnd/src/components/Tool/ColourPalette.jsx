import React, { useState } from 'react';
import { seasonalPalettes, colourNames } from './Palette.jsx';
import FeedbackLoop from './FeedbackLoop';

const ColourPalette = ({ season }) => {
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [userPreferences, setUserPreferences] = useState({
    likedColors: [],
    dislikedColors: []
  });

  const formattedSeason = season
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  const palette = seasonalPalettes[formattedSeason];

  if (!palette) {
    console.log('Available seasons:', Object.keys(seasonalPalettes));
    console.log('Requested season:', formattedSeason);
    return <p>No palette available for "{season}". Tried: "{formattedSeason}"</p>;
  }

  const renderColorGroup = (colors, title) => (
    <div className="mb-6">
      <h3 className="text-lg font-semibold mb-3 text-center">{title}</h3>
      <div className="flex flex-wrap justify-center gap-4">
        {colors.map((color, index) => (
          <div key={index} className="flex flex-col items-center w-24">
            <div
              className="w-20 h-20 rounded-full border shadow-md transition-transform hover:scale-110"
              style={{ backgroundColor: color }}
            ></div>
            <span className="text-sm mt-2 font-medium text-center">{colourNames[color] || "Unknown"}</span>
            <span className="text-xs font-mono text-gray-500">{color}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const handleCloseFeedback = (feedback, userPreferences) => {
    setFeedback(feedback);
    setUserPreferences(userPreferences);
    setShowFeedback(false);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h2 className="text-2xl font-semibold mb-4 text-pink-600 text-center">{formattedSeason} Palette</h2>
      <div className="bg-gray-50 p-6 rounded-lg shadow-md">
        {renderColorGroup(palette.primary, "Primary Colours")}
        {renderColorGroup(palette.secondary, "Secondary Colours")}
      </div>
      {/* <div className="flex justify-center mt-4">
      <button
        onClick={() => setShowFeedback(true)}
        className="mt-4 px-6 py-2 bg-pink-600 hover:bg-pink-800 text-white rounded-lg font-semibold transition-colors"
      >
        Personalize My Colours
      </button>
      </div> */}
      {showFeedback && <FeedbackLoop season={season} onClose={handleCloseFeedback} />}
      
      {feedback && (
        <div className="mt-6 p-4 bg-pink-50 border border-pink-200 rounded-lg">
          <p className="text-pink-800">{feedback}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div>
              <h4 className="font-medium text-sm mb-2 text-gray-700">Colours You Liked:</h4>
              <div className="flex flex-wrap gap-2">
                {userPreferences.likedColors.length > 0 ? (
                  userPreferences.likedColors.map((color, index) => (
                    <div 
                      key={index} 
                      className="w-10 h-10 rounded-full border shadow" 
                      style={{ backgroundColor: color }}
                      title={colourNames[color] || color}
                    ></div>
                  ))
                ) : (
                  <p className="text-gray-500 text-sm italic">No liked colours yet</p>
                )}
              </div>
            </div>
            <div>
              <h4 className="font-medium text-sm mb-2 text-gray-700">Colours You Disliked:</h4>
              <div className="flex flex-wrap gap-2">
                {userPreferences.dislikedColors.length > 0 ? (
                  userPreferences.dislikedColors.map((color, index) => (
                    <div 
                      key={index} 
                      className="w-10 h-10 rounded-full border shadow" 
                      style={{ backgroundColor: color }}
                      title={colourNames[color] || color}
                    ></div>
                  ))
                ) : (
                  <p className="text-gray-500 text-sm italic">No disliked colours yet</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ColourPalette;
