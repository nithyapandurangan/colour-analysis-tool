import React from 'react';
import Webcam from "react-webcam";

const WebcamCapture = ({ isWebcamOpen, captureImageFromWebcam, webcamRef }) => {
  return (
    <div className="mb-4">
      {isWebcamOpen && (
        <>
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="rounded-lg shadow-lg"
          />
          <button
            onClick={captureImageFromWebcam}
            className="mt-4 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-700"
          >
            Capture Photo
          </button>
        </>
      )}
    </div>
  );
};

export default WebcamCapture;
