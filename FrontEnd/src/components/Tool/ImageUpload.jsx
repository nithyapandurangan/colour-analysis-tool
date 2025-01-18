import React from 'react';

const ImageUpload = ({ handleImageUpload, analyzeColor, selectedImage, loading }) => {
  return (
    <div>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <button onClick={analyzeColor} disabled={!selectedImage || loading}>
      </button>
    </div>
  );
};

export default ImageUpload;
