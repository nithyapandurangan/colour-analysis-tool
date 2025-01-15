import React from "react";
import { Link } from "react-router-dom";  // Import Link from react-router-dom

import hands1 from "../../assets/images/image.png";
import hands2 from "../../assets/images/hands_2.png";
import line1 from "../../assets/images/line.png";
import line2 from "../../assets/images/line-2.png";
import circleline from "../../assets/images/line.svg";
import bulb1 from "../../assets/images/remove-bg-1-2.png";
import bulb2 from "../../assets/images/remove-bg-1.png";

export const Hero = () => {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-40 py-[140px] relative overflow-hidden">
      {/* Left line */}
      <img className="absolute w-[440px] h-[800px] top-[2%] right-[80%]" alt="Line" src={line1} />
      {/* Left bulb */}
      <img className="absolute w-[90px] h-[115px] top-[30%] left-[13%]" alt="bulb" src={bulb1} />
      {/* Left hand */}
      <img className="absolute w-[414px] h-[217px] -top-2 right-[78%]" alt="Hands" src={hands1} />
      {/* Right line */}
      <img className="absolute w-[376px] h-[507px] top-[220px] left-[1250px]" alt="Line" src={line2} />
      {/* Right hand */}
      <img className="absolute w-[403px] h-[426px] top-[415px] left-[1200px]" alt="Hands" src={hands2} />
      {/* Right bulb */}
      <img className="absolute w-[82px] h-[114px] top-[260px] left-[1200px]" alt="Remove bg" src={bulb2} />

      <div className="inline-flex flex-col items-center justify-center gap-8 relative">
        <img className="absolute w-[703px] h-[132px] top-[103px] left-[68px]" alt="Line" src={circleline} />
        <p className="relative w-[977px] font-bold text-black text-[110px] text-center leading-[112px]">
          Discover your true colours &amp; elevate your style
        </p>
        <p className="relative w-[800px] font-normal text-[#15291f] text-xl text-center tracking-[0] leading-8">
          Our Colour Analysis Tool blends innovation, style & AI to reveal your perfect colours, using your skin, hair, eye colours,
          and undertones.
        </p>
      </div>

      {/* Wrap the button inside Link component */}
      <Link to="/tool">
        <button className="mt-5 box-border items-center justify-center gap-3 px-8 py-3 bg-p border-2 border-pink-400 rounded-3xl  hover:bg-white hover:text-pink-600 hover:font-bold transition-colors duration-300">
          <div className="relative w-fit font-semibold text-black text-lg text-center leading-6 whitespace-nowrap">
            Try Our Tool!
          </div>
        </button>
      </Link>
    </div>
  );
};

export default Hero;
