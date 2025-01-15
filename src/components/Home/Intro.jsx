import React from "react";
import Colourwheel from "../../assets/images/colourwheel.jpeg"
import line from "../../assets/images/Vector-light.png"
import linemain from "../../assets/images/Vector.png"

const Intro = (props) => {
  return (
    <div className="w-full flex min-h-screen items-center flex-col bg-white">
        <div className="relative w-full flex flex-col items-end p-16">

          {/* Title */}
          <div className="absolute top-10 left-10 flex flex-col items-center w-full">
            <span className="text-6xl font-bold text-black text-center">
              What exactly is seasonal <br />
              colour analysis?
            </span>
            <img src={linemain} alt="Vector" className="absolute -bottom-4" />


        {/* Intro */}
          <span className="absolute top-[180px] left-[20%] text-lg text-center w-[900px] text-black">
            Our Colour analysis tool determines which colours flatter you most based on your eye, hair, and skin colour. You will then be assigned one of 4 seasons- Spring, Summer, Autumn & Winter. Each colour season comes with a colour palette, specifically designed to harmonize with your features.
          </span>
          </div>

          {/* Spring Section */}
          <div className="absolute top-[350px] left-[200px] flex items-start">
            <img src={line} alt="Vector" className="absolute -bottom-3 left-1" />
            <span className="text-4xl font-bold text-black">Spring</span>
          </div>
          <span className="absolute top-[450px] left-[50px] text-lg text-black w-[500px] text-center">
            People who fall under the Spring category typically have warm undertones with light, clear features. Their overall colouring is warm, fresh, and often delicate, meaning they look best in bright, warm colours that bring out their natural radiance.
          </span>

          {/* Summer Section */}
          <div className="absolute top-[350px] left-[1100px] flex items-start">
            <img src={line} alt="Vector" className="absolute w-[170px] -bottom-3 left-1" />
            <span className="text-4xl font-bold text-black">Summer</span>
          </div>
          <span className="absolute top-[450px] left-[900px] text-lg text-black w-[500px] text-center">
            Summer individuals have cool, soft, and muted colouring. They typically look best in cool-toned colours that have a soft, subdued quality. This season is characterized by a gentle, cool, and soft appearance.
          </span>


          {/* Autumn Section */}
          <div className="absolute top-[650px] left-[200px] flex items-start">
            <img src={line} alt="Vector" className="absolute w-[170px] -bottom-3 left-1" />
            <span className="text-4xl font-bold text-black">Autumn</span>
          </div>
          <span className="absolute top-[750px] left-[50px] text-lg text-black w-[500px] text-center">
            Autumn types have warm, rich, and earthy tones in their overall colouring. Their skin, hair, and eyes are often characterized by warmth and depth, making them look stunning in deep, golden, and earthy shades.
          </span>

          {/* Winter Section */}
          <div className="absolute top-[650px] left-[1100px] flex items-start">
            <img src={line} alt="Vector" className="absolute w-[170px] -bottom-3 left-1" />
            <span className="text-4xl font-bold text-black">Winter</span>
          </div>
          <span className="absolute top-[750px] left-[900px] text-lg text-black w-[500px] text-center">
            Winter types have cool, vivid, and high-contrast features, often looking striking in bold, cool colours. The Winter palette is made up of intense and clear shades that complement their naturally sharp, cool-toned appearance.
          </span>

          {/* Colour Wheel */}
          <img src={Colourwheel} alt="colourwheel" className="absolute top-[450px] left-[550px] w-[350px] h-[350px]" />

        </div>
      </div>
  );
};

export default Intro;
