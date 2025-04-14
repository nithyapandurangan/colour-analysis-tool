import React, { useState, useEffect } from 'react';
import { seasonalPalettes, colourNames } from './Palette.jsx';

const FeedbackLoop = ({ season, onClose }) => {
  const formattedSeason = season
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

  const palette = seasonalPalettes[formattedSeason];

  const [userPreferences, setUserPreferences] = useState({
    likedColors: [],
    dislikedColors: []
  });

  const [feedback, setFeedback] = useState('');
  const [allColorsRated, setAllColorsRated] = useState(false);

  // Combine primary and secondary colors
  const allColors = [...(palette?.primary || []), ...(palette?.secondary || [])];

  // Check if all colors have been rated
  useEffect(() => {
    const ratedColors = [...userPreferences.likedColors, ...userPreferences.dislikedColors];
    if (ratedColors.length === allColors.length) {
      setAllColorsRated(true);
      generateFeedback();
    } else {
      setAllColorsRated(false);
    }
  }, [userPreferences, allColors]);

  const handleColorFeedback = (color, isLiked) => {
    let newLiked = [...userPreferences.likedColors];
    let newDisliked = [...userPreferences.dislikedColors];

    if (isLiked) {
      newLiked.push(color);
      newDisliked = newDisliked.filter(c => c !== color);
    } else {
      newDisliked.push(color);
      newLiked = newLiked.filter(c => c !== color);
    }

    setUserPreferences({
      likedColors: newLiked,
      dislikedColors: newDisliked
    });
  };

  const generateFeedback = () => {
    const liked = userPreferences.likedColors;
    const disliked = userPreferences.dislikedColors;
  
    if (liked.length === 0 && disliked.length === 0) {
      setFeedback(`You haven't rated any colours yet in the ${formattedSeason} palette.`);
      return;
    }
  
    if (liked.length === 0) {
      setFeedback(`You've disliked all rated colours in the ${formattedSeason} palette. Consider trying a different seasonal palette.`);
      return;
    }
  
    // Convert liked colors to HSL
    const likedHSL = liked.map(color => hex2hsl(color));
    
    // Calculate average values
    const avgHue = likedHSL.reduce((sum, [h]) => sum + h, 0) / likedHSL.length;
    const avgSat = likedHSL.reduce((sum, [, s]) => sum + s, 0) / likedHSL.length;
    const avgLight = likedHSL.reduce((sum, [, , l]) => sum + l, 0) / likedHSL.length;
  
    // Get the most represented hue category
    const hueCategories = [
      { name: "reds", range: [0, 15], altRange: [345, 360] },
      { name: "red-oranges", range: [15, 45] },
      { name: "oranges and yellows", range: [45, 70] },
      { name: "yellows and yellow-greens", range: [70, 100] },
      { name: "greens", range: [100, 160] },
      { name: "aquas and cyans", range: [160, 200] },
      { name: "blues", range: [200, 240] },
      { name: "indigos and deep blues", range: [240, 270] },
      { name: "violets and purples", range: [270, 300] },
      { name: "magentas and pinks", range: [300, 330] },
      { name: "red-pinks", range: [330, 345] }
    ];
  
    // Count how many liked colors fall into each category
    const categoryCounts = {};
    hueCategories.forEach(category => {
      categoryCounts[category.name] = 0;
    });
  
    likedHSL.forEach(([h]) => {
      for (const category of hueCategories) {
        if ((h >= category.range[0] && h < category.range[1])) {
          categoryCounts[category.name]++;
          break;
        }
        if (category.altRange && h >= category.altRange[0] && h < category.altRange[1]) {
          categoryCounts[category.name]++;
          break;
        }
      }
    });
  
    // Find the most represented category
    let mostRepresentedCategory = "various colors";
    let maxCount = 0;
    for (const [name, count] of Object.entries(categoryCounts)) {
      if (count > maxCount) {
        maxCount = count;
        mostRepresentedCategory = name;
      }
    }
  
    // If multiple categories tie for top, be more general
    const topCategories = Object.entries(categoryCounts)
      .filter(([_, count]) => count === maxCount)
      .map(([name]) => name);
  
    let hueCategory;
    if (topCategories.length > 1) {
      hueCategory = "a mix of colors";
    } else {
      hueCategory = mostRepresentedCategory;
    }
  
    // Determine saturation category
    let satCategory;
    if (avgSat > 70) satCategory = "very vibrant";
    else if (avgSat > 50) satCategory = "vibrant";
    else if (avgSat > 30) satCategory = "moderately saturated";
    else if (avgSat > 15) satCategory = "muted";
    else satCategory = "very muted";
  
    // Determine lightness category
    let lightCategory;
    if (avgLight > 80) lightCategory = "very light";
    else if (avgLight > 60) lightCategory = "light";
    else if (avgLight > 40) lightCategory = "medium";
    else if (avgLight > 20) lightCategory = "dark";
    else lightCategory = "very dark";
  
    // Detect if preferences are monochromatic (narrow hue range)
    const hueRange = Math.max(...likedHSL.map(hsl => hsl[0])) - Math.min(...likedHSL.map(hsl => hsl[0]));
    const isMonochromatic = hueRange < 30;
  
    // Detect if preferences are neutral (low saturation)
    const isNeutral = avgSat < 20;
  
    // Build feedback message
    let feedbackMessage = `Based on your preferences in the ${formattedSeason} palette:`;
  
    if (isMonochromatic) {
      feedbackMessage += ` You have a strong preference for ${hueCategory}.`;
    } else if (topCategories.length > 1) {
      feedbackMessage += ` You tend to prefer ${hueCategory} including ${topCategories.join(' and ')}.`;
    } else {
      feedbackMessage += ` You tend to prefer ${hueCategory}.`;
    }
  
    feedbackMessage += ` Your choices are ${satCategory} and ${lightCategory}.`;
  
    if (isNeutral) {
      feedbackMessage += ` You seem drawn to more neutral, subtle tones.`;
    }
  
    // Add specific notes about pink preferences if detected
    const pinkCount = liked.filter(color => {
      const [h] = hex2hsl(color);
      return (h >= 330 || h < 15) && hex2hsl(color)[1] > 20; // Reds/pinks with some saturation
    }).length;
  
    if (pinkCount > liked.length * 0.5) { // If majority are pinks
      feedbackMessage += ` You particularly favour pink tones.`;
    }
  
    // Check for contrast preferences
    const contrastRange = Math.max(...likedHSL.map(hsl => hsl[2])) - Math.min(...likedHSL.map(hsl => hsl[2]));
    if (contrastRange > 50) {
      feedbackMessage += ` You like a wide range of lightness values.`;
    } else if (contrastRange < 20) {
      feedbackMessage += ` You prefer colours with similar lightness levels.`;
    }
  
    setFeedback(feedbackMessage);
  };

  const hex2hsl = (hex) => {
    let r = parseInt(hex.slice(1, 3), 16) / 255;
    let g = parseInt(hex.slice(3, 5), 16) / 255;
    let b = parseInt(hex.slice(5, 7), 16) / 255;

    let max = Math.max(r, g, b);
    let min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
      h = s = 0;
    } else {
      let d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
        default: h = 0;
      }

      h /= 6;
    }

    return [h * 360, s * 100, l * 100];
  };

  const handleClose = () => {
    onClose(feedback, userPreferences);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white p-6 rounded-lg shadow-lg max-w-4xl w-full">
        <h2 className="text-2xl font-bold mb-6 text-center text-pink-600">Personalize Your {formattedSeason} Colours</h2>
        
        {allColorsRated && feedback && (
          <div className="p-4 bg-pink-50 border border-pink-200 rounded-lg mb-6">
            <p className="text-pink-800">{feedback}</p>
          </div>
        )}

        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-center text-gray-800">Primary Colours</h3>
          <div className="flex flex-wrap gap-10 justify-center">
            {palette?.primary.map((color, index) => (
              <div key={index} className="flex flex-col items-center w-24">
                <div 
                  className="w-20 h-20 rounded-full border shadow-md transition-transform hover:scale-105"
                  style={{ backgroundColor: color }}
                ></div>
                <span className="text-sm mt-2 font-medium text-center">{colourNames[color] || "Unknown"}</span>
                <span className="text-xs font-mono text-gray-500">{color}</span>
                <div className="flex mt-2">
                  <button 
                    className="px-3 py-1 bg-green-500 text-white text-xs rounded-l hover:bg-green-600 transition-colors"
                    onClick={() => handleColorFeedback(color, true)}
                  >
                    Like
                  </button>
                  <button 
                    className="px-3 py-1 bg-red-500 text-white text-xs rounded-r hover:bg-red-600 transition-colors"
                    onClick={() => handleColorFeedback(color, false)}
                  >
                    Dislike
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-3 text-center text-gray-800">Secondary Colours</h3>
          <div className="flex flex-wrap gap-10 justify-center">
            {palette?.secondary.map((color, index) => (
              <div key={index} className="flex flex-col items-center w-24">
                <div 
                  className="w-20 h-20 rounded-full border shadow-md transition-transform hover:scale-105"
                  style={{ backgroundColor: color }}
                ></div>
                <span className="text-sm mt-2 font-medium text-center">{colourNames[color] || "Unknown"}</span>
                <span className="text-xs font-mono text-gray-500">{color}</span>
                <div className="flex mt-2">
                  <button 
                    className="px-3 py-1 bg-green-500 text-white text-xs rounded-l hover:bg-green-600 transition-colors"
                    onClick={() => handleColorFeedback(color, true)}
                  >
                    Like
                  </button>
                  <button 
                    className="px-3 py-1 bg-red-500 text-white text-xs rounded-r hover:bg-red-600 transition-colors"
                    onClick={() => handleColorFeedback(color, false)}
                  >
                    Dislike
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-white p-4 rounded-lg shadow-sm mb-6">
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

        <div className="flex justify-center space-x-4">
          <button 
            className="px-6 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600 transition-colors font-medium"
            onClick={handleClose}
          >
            Close
          </button>
          {/* <button 
            className="px-6 py-2 bg-pink-500 text-white rounded-lg hover:bg-pink-600 transition-colors font-medium"
            onClick={generateFeedback}
            disabled={!allColorsRated}
          >
            Update Recommendations
          </button> */}
        </div>
      </div>
    </div>
  );
};

export default FeedbackLoop;
