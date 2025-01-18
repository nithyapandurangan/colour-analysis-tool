import React, { useState, useRef } from 'react';
import WebcamCapture from '../components/Tool/WebcamCapture';
import ImageUpload from '../components/Tool/ImageUpload';
import ResultDisplay from '../components/Tool/ResultDisplay';
import ProductRecommendations from '../components/Tool/ProductRecommendations';

const Tool = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [isWebcamOpen, setIsWebcamOpen] = useState(false);
  const webcamRef = useRef(null);

  const captureImageFromWebcam = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    console.log(imageSrc);
    const blob = dataURItoBlob(imageSrc);
    const file = new File([blob], "webcam_photo.jpg", { type: "image/jpeg" });
    setSelectedImage(file);
    setIsWebcamOpen(false);
  };

  const dataURItoBlob = (dataURI) => {
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    const buffer = new Uint8Array(byteString.length);
    for (let i = 0; i < byteString.length; i++) {
      buffer[i] = byteString.charCodeAt(i);
    }
    return new Blob([buffer], { type: mimeString });
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setResult(null);
      setShowRecommendations(false);
    }
  };

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

  return (
    <div className="min-h-screen bg-white p-8 flex flex-col items-center">
      <h1 className="text-4xl font-extrabold text-black mb-8 text-center">
        Our Colour Analysis Tool
      </h1>

      <div className="bg-white p-6 rounded-lg shadow-lg w-full max-w-3xl">
        <div className="flex flex-col items-center mb-6">
          <div className="flex items-center space-x-4 mb-4">
            {/* Open Webcam Button with same style as Upload File button */}
            <button
              onClick={() => setIsWebcamOpen((prev) => !prev)}
              className="px-1 py-1 text-[15px] bg-gray-100 border border-gray-500 text-black rounded-md hover:bg-gray-200"
            >
              {isWebcamOpen ? "Close Webcam" : "Open Webcam"}
            </button>

            {/* Upload File Button */}
            <ImageUpload
              handleImageUpload={handleImageUpload}
              selectedImage={selectedImage}
            />
          </div>

          {/* Webcam Capture */}
          <WebcamCapture
            isWebcamOpen={isWebcamOpen}
            captureImageFromWebcam={captureImageFromWebcam}
            webcamRef={webcamRef}
          />

          <p className="text-gray-500 mb-4">
            Upload an image or take a photo for colour analysis & clothing recommendation
          </p>

          {/* Display Image Upload & Analyze Colour button */}
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
                disabled={loading}
                className="mt-4 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? "Analyzing..." : "Analyze Colour"}
              </button>
            </div>
          )}

          {result && (
            <ResultDisplay result={result} showRecommendations={showRecommendations} setShowRecommendations={setShowRecommendations} />
          )}
        </div>
      </div>
    </div>
  );
};

export default Tool;
