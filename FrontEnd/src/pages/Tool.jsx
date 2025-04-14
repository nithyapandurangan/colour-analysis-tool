import React, { useState, useRef } from 'react';
import { Camera, Upload, RefreshCw, Check, Info, ChevronDown, AlertTriangle, ChevronRight, ChevronLeft } from 'lucide-react';
import WebcamCapture from '../components/Tool/WebcamCapture';
import ImageUpload from '../components/Tool/ImageUpload';
import ResultDisplay from '../components/Tool/ResultDisplay';
import FeedbackLoop from '../components/Tool/FeedbackLoop';
import { seasonalPalettes, colourNames } from '../components/Tool/Palette';

const Tool = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [isWebcamOpen, setIsWebcamOpen] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [userPreferences, setUserPreferences] = useState({
    likedColors: [],
    dislikedColors: []
  });
  const [styleMessageOpen, setStyleMessageOpen] = useState(false);
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const toolSectionRef = useRef(null);

  const captureImageFromWebcam = () => {
    const imageSrc = webcamRef.current.getScreenshot();
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

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      setSelectedImage(event.dataTransfer.files[0]);
      setResult(null);
      setShowRecommendations(false);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
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
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setResult(data.predicted_season);
      setShowRecommendations(false); // Initially hide recommendations
    } catch (error) {
      console.error('Error analyzing image:', error);
      setResult('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setResult(null);
    setShowRecommendations(false);
    setFeedback('');
    setUserPreferences({
      likedColors: [],
      dislikedColors: []
    });
  };

  const scrollToTool = () => {
    toolSectionRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle the closure of the feedback component
  const handleCloseFeedback = (feedbackMsg, preferences) => {
    setFeedback(feedbackMsg);
    setUserPreferences(preferences);
    setShowFeedback(false);
  };

  // Format the season key for display (ensure proper capitalization)
  const formatSeasonName = (season) => {
    if (!season) return '';
    return season.replace(/(^\w|\s\w)/g, m => m.toUpperCase());
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-pink-50 to-white">
      {/* Hero Section */}
      <div className="py-16 px-4 sm:px-6 lg:px-8 flex flex-col items-center text-center">
        <h1 className="text-5xl font-bold text-pink-800 mb-6">
          Discover Your Perfect Palette
        </h1>
        <p className="text-xl text-black max-w-4xl mb-8">
          Our AI-powered 12-season colour analysis tool helps you find your precise seasonal colour palette for fashion, makeup, and personal style.
        </p>
        
        {/* 12-Season Palette Preview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 w-full max-w-5xl mb-12">
          {Object.entries({
            'Spring': ['Light Spring', 'True Spring', 'Bright Spring'],
            'Summer': ['Light Summer', 'True Summer', 'Soft Summer'],
            'Autumn': ['Soft Autumn', 'True Autumn', 'Dark Autumn'],
            'Winter': ['Dark Winter', 'True Winter', 'Bright Winter']
          }).map(([family, seasons]) => (
            <div key={family} className="bg-white p-4 rounded-lg shadow-md">
              <h3 className="capitalize text-lg font-medium text-pink-800 mb-3">{family}</h3>
              <div className="space-y-3">
                {seasons.map(season => (
                  <div key={season} className="flex flex-col">
                    <span className="text-sm capitalize text-gray-600 mb-1">{season}</span>
                    <div className="flex">
                      {seasonalPalettes[season].primary.slice(0, 5).map((color, idx) => (
                        <div 
                          key={idx} 
                          className="w-8 h-6 first:rounded-l-md last:rounded-r-md"
                          style={{ backgroundColor: color }}
                          title={colourNames[color]}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
        
        {/* Scroll Down Button */}
        <button 
          onClick={scrollToTool}
          className="mt-6 flex flex-col items-center text-pink-800 hover:text-pink-600 transition-colors animate-bounce"
        >
          <span className="mb-2">Go to Colour Analysis Tool</span>
          <ChevronDown size={24} />
        </button>
      </div>

      {/* Main Tool Section */}
      <div ref={toolSectionRef} className="max-w-4xl mx-auto px-4 pb-24 scroll-mt-8">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Tool Header */}
          <div className="bg-gradient-to-r from-pink-600 to-pink-800 py-6 px-8">
            <h2 className="text-3xl font-bold text-white">12-Season Colour Analysis Tool</h2>
            <p className="text-pink-100">Upload your photo to discover your precise seasonal colour palette</p>
          </div>

          <div className="p-8">
            {/* Step Indicator */}
            <div className="flex items-center justify-center mb-10">
              <div className={`flex items-center ${selectedImage ? 'text-gray-400' : 'text-pink-800'}`}>
                <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${selectedImage ? 'border-gray-300' : 'border-pink-800'}`}>
                  <Upload size={20} />
                </div>
                <span className="ml-2 font-medium">Upload</span>
              </div>
              <div className="w-16 h-1 mx-4 bg-gray-200"></div>
              <div className={`flex items-center ${result ? 'text-gray-400' : selectedImage ? 'text-pink-800' : 'text-gray-400'}`}>
                <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${result ? 'border-gray-300' : selectedImage ? 'border-pink-800' : 'border-gray-300'}`}>
                  <RefreshCw size={20} />
                </div>
                <span className="ml-2 font-medium">Analyse</span>
              </div>
              <div className="w-16 h-1 mx-4 bg-gray-200"></div>
              <div className={`flex items-center ${result ? 'text-pink-800' : 'text-gray-400'}`}>
                <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${result ? 'border-pink-800' : 'border-gray-300'}`}>
                  <Check size={20} />
                </div>
                <span className="ml-2 font-medium">Results</span>
              </div>
            </div>

            {/* Image Upload Area */}
            {!selectedImage && !result && (
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-pink-500 transition-colors"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <div className="flex flex-col items-center space-y-6">
                  <div className="p-4 bg-pink-100 rounded-full">
                    <Upload size={36} className="text-pink-800" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-gray-700">Upload your photo</h3>
                    <p className="text-gray-500 mt-1">Drag and drop your image here or click to browse</p>
                    <p className="text-xs text-gray-400 mt-2">Your image will be analyzed for colour features but will not be stored or shared.</p>
                  </div>
                  <div className="flex flex-wrap justify-center gap-4">
                    <button
                      onClick={() => fileInputRef.current.click()}
                      className="px-6 py-3 bg-pink-800 text-white rounded-full hover:bg-pink-700 transition-colors flex items-center shadow-md"
                    >
                      <Upload size={18} className="mr-2" />
                      Choose File
                    </button>
                    <button
                      onClick={() => setIsWebcamOpen((prev) => !prev)}
                      className="px-6 py-3 bg-pink-600 text-white rounded-full hover:bg-pink-700 transition-colors flex items-center shadow-md"
                    >
                      <Camera size={18} className="mr-2" />
                      Use Webcam
                    </button>
                  </div>
                  <input 
                    type="file" 
                    ref={fileInputRef}
                    onChange={handleImageUpload}
                    accept="image/*" 
                    className="hidden" 
                  />
                </div>
              </div>
            )}

            {/* Webcam Capture */}
            {isWebcamOpen && (
              <div className="mb-8 p-4 border border-gray-200 rounded-lg">
                <WebcamCapture
                  isWebcamOpen={isWebcamOpen}
                  captureImageFromWebcam={captureImageFromWebcam}
                  webcamRef={webcamRef}
                />
                {/* Added centered capture button */}
                <div className="mt-4 flex justify-center">
                  <button
                    onClick={captureImageFromWebcam}
                    className="px-6 py-3 bg-pink-600 text-white rounded-full hover:bg-pink-700 transition-colors flex items-center shadow-md"
                  >
                    <Camera size={18} className="mr-2" />
                    Capture Photo
                  </button>
                </div>
              </div>
              
            )}
            <div className="mt-6 w-full bg-red-50 border-l-4 border-red-400 p-4 rounded-lg">
              <div className="flex items-start text-red-800">
                <AlertTriangle size={20} className="flex-shrink-0 mr-3 mt-0.5" />
                <div>
                  <h4 className="font-medium mb-2">Photo Guidelines for Best Results</h4>
                  <ul className="list-disc space-y-2 pl-5 text-sm">
                    <li>No hats or head coverings</li>
                    <li>Use natural lighting - avoid harsh artificial light</li>
                    <li>Remove glasses/spectacles</li>
                    <li>Face should be clearly visible</li>
                    <li>Avoid heavy makeup if possible</li>
                  </ul>
                </div>
              </div>
            </div>


            {/* Selected Image Preview */}
            {selectedImage && !result && (
              <div className="flex flex-col items-center">
                <div className="mb-6 relative">
                  <img
                    src={URL.createObjectURL(selectedImage)}
                    alt="Preview"
                    className="max-h-80 rounded-lg shadow-lg"
                  />
                  <button 
                    onClick={resetAnalysis}
                    className="absolute -top-3 -right-3 bg-white rounded-full p-1 shadow-md hover:bg-gray-100"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18"></line>
                      <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                  </button>
                </div>
                
                <button
                  onClick={analyzeColor}
                  disabled={loading}
                  className="px-8 py-3 bg-gradient-to-r from-pink-600 to-pink-800 text-white text-lg rounded-full hover:from-pink-700 hover:to-pink-900 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg flex items-center"
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analysing...
                    </>
                  ) : (
                    <>Analyse My Colours</>
                  )}
                </button>
              </div>
            )}

            {/* Results Display */}
            {result && (
              <div className="mt-6">
                {/* 1. Season Title */}
                <div className="text-center mb-6">
                  <h3 className="text-2xl font-bold text-pink-800">Your Colour Season</h3>
                  <div className="mt-4 mb-8">
                    <div className="inline-block bg-gradient-to-r from-pink-600 to-pink-800 text-white text-xl font-bold py-3 px-8 rounded-lg shadow-lg capitalize">
                      {formatSeasonName(result)}
                    </div>
                  </div>
                  <img
                    src={URL.createObjectURL(selectedImage)}
                    alt="Analyzed"
                    className="max-h-64 rounded-lg shadow-lg mx-auto mb-6"
                  />
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  {/* Message about Color Recommendation Accordion */}
                  <div className="mb-8 bg-pink-50 p-4 rounded-lg border-l-4 border-pink-400">
                    <button 
                      onClick={() => setStyleMessageOpen(!styleMessageOpen)}
                      className="w-full flex justify-between items-center"
                    >
                      <div className="flex items-center">
                        <Info size={20} className="text-pink-600 mr-3" />
                        <h4 className="text-med font-bold text-pink-800">Your Style, Your Choice</h4>
                      </div>
                      <ChevronRight 
                        size={20} 
                        className={`text-pink-800 transition-transform ${styleMessageOpen ? 'rotate-90' : ''}`} 
                      />
                    </button>
                    
                    {styleMessageOpen && (
                      <div className="mt-3 text-pink-800">
                        <p className="text-pink-700 mt-1">
                          These colour recommendations are meant to guide you toward shades that naturally complement your features, 
                          but they're not strict rules! If you love a colour outside your season, you can still wear it—just consider 
                          incorporating it as an accent, in patterns, or through accessories. <b>Confidence and personal style matter 
                          just as much as colour theory.</b> 
                        </p>
                      </div>
                    )}
                  </div>

                  {/* 2. Colour Palette Display */}
                  <ResultDisplay 
                    result={result}
                    showRecommendations={showRecommendations}
                    setShowRecommendations={setShowRecommendations}
                  />

                    {/* 3. User Feedback Section */}
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

                  {/* Banner for personalization */}
                  <div className="my-8 bg-gradient-to-r from-pink-100 to-pink-200 p-6 rounded-lg text-center border-l-4 border-pink-500 shadow-md">
                    <h4 className="text-lg font-medium text-pink-800 mb-2">
                      Not completely connecting with your colour palette?
                    </h4>
                    <p className="text-gray-600 mb-4">
                      Everyone's unique! Fine-tune your palette by selecting colours that resonate with you personally.
                    </p>
                    <button
                      onClick={() => setShowFeedback(true)}
                      className="px-6 py-3 bg-white text-pink-700 border border-pink-400 rounded-lg hover:bg-pink-50 transition-colors shadow-md"
                    >
                      Personalize My Colours
                    </button>
                  </div>

                  {/* Banner for users to click for clothing recommendations */}
                  {!showRecommendations && (
                    <div className="my-8 bg-gradient-to-r from-pink-600 to-pink-800  p-6 rounded-lg text-center shadow-lg">
                      <h4 className="text-xl font-medium text-white mb-2">
                        Curious which clothing styles would complement your colours?
                      </h4>
                      <p className="text-pink-100 mb-4">
                        Get personalized wardrobe recommendations based on your seasonal colour palette
                      </p>
                      <button
                        onClick={() => setShowRecommendations(true)}
                        className="px-6 py-3 bg-white text-pink-800 rounded-lg hover:bg-pink-50 transition-colors shadow-md font-medium"
                      >
                        View Recommended Products
                      </button>
                    </div>
                  )}
                  
                  <div className="flex flex-wrap justify-center mt-8 gap-4">
                    <button
                      onClick={resetAnalysis}
                      className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      Start Over
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Feedback Loop Component */}
            {showFeedback && (
              <FeedbackLoop 
                season={result} 
                onClose={handleCloseFeedback}
              />
            )}

            {/* Informational Footer */}
            {!result && (
              <div className="mt-8 pt-6 border-t border-gray-100">
                <h4 className="text-pink-800 font-medium mb-2">Understanding the 12-Season Colour System:</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
                  <div className="flex flex-col p-3 bg-pink-50 rounded-lg">
                    <h5 className="font-medium text-pink-800 mb-1">Spring Family</h5>
                    <ul className="list-disc pl-5 text-gray-600">
                      <li>Light Spring</li>
                      <li>True Spring</li>
                      <li>Bright Spring</li>
                    </ul>
                  </div>
                  <div className="flex flex-col p-3 bg-pink-50 rounded-lg">
                    <h5 className="font-medium text-pink-800 mb-1">Summer Family</h5>
                    <ul className="list-disc pl-5 text-gray-600">
                      <li>Light Summer</li>
                      <li>True Summer</li>
                      <li>Soft Summer</li>
                    </ul>
                  </div>
                  <div className="flex flex-col p-3 bg-pink-50 rounded-lg">
                    <h5 className="font-medium text-pink-800 mb-1">Autumn Family</h5>
                    <ul className="list-disc pl-5 text-gray-600">
                      <li>Soft Autumn</li>
                      <li>True Autumn</li>
                      <li>Dark Autumn</li>
                    </ul>
                  </div>
                  <div className="flex flex-col p-3 bg-pink-50 rounded-lg">
                    <h5 className="font-medium text-pink-800 mb-1">Winter Family</h5>
                    <ul className="list-disc pl-5 text-gray-600">
                      <li>Dark Winter</li>
                      <li>True Winter</li>
                      <li>Bright Winter</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="flex flex-col items-center p-4 bg-pink-50 rounded-lg">
                    <div className="p-2 bg-pink-100 rounded-full mb-3">
                      <Camera size={24} className="text-pink-800" />
                    </div>
                    <h5 className="font-medium text-gray-700 mb-1">Upload Photo</h5>
                    <p className="text-sm text-center text-gray-500">Take a clear photo in natural lighting for best results</p>
                  </div>
                  <div className="flex flex-col items-center p-4 bg-pink-50 rounded-lg">
                    <div className="p-2 bg-pink-100 rounded-full mb-3">
                      <RefreshCw size={24} className="text-pink-800" />
                    </div>
                    <h5 className="font-medium text-gray-700 mb-1">AI Analysis</h5>
                    <p className="text-sm text-center text-gray-500">Our algorithm identifies your precise season among 12 possibilities</p>
                  </div>
                  <div className="flex flex-col items-center p-4 bg-pink-50 rounded-lg">
                    <div className="p-2 bg-pink-100 rounded-full mb-3">
                      <Check size={24} className="text-pink-800" />
                    </div>
                    <h5 className="font-medium text-gray-700 mb-1">Get Results</h5>
                    <p className="text-sm text-center text-gray-500">Receive your personalized colour recommendations tailored just for you</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Tool;
