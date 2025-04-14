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
        </>
      )}
    </div>
  );
};

export default WebcamCapture;
