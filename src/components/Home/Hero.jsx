import React from "react";

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
      <img className="absolute w-[440px] h-[804px] top-[5px] left-[-142px]" alt="Line" src={line1} />
      <img className="absolute w-[403px] h-[426px] top-[415px] left-[1544px]" alt="Hands" src={hands2} />
      <img className="absolute w-[414px] h-[217px] -top-2 left-[-142px]" alt="Hands" src={hands1} />
      <div className="inline-flex flex-col items-center justify-center gap-8 relative flex-[0_0_auto]">
        <img className="absolute w-[703px] h-[132px] top-[103px] left-[68px]" alt="Line" src={circleline} />
        <p className="relative w-[977px] mt-[-1.00px] [font-family:'Inter-Bold',Helvetica] font-bold text-black text-[112px] text-center tracking-[0] leading-[112px]">
          Discover your true colours &amp; elevate your style
        </p>
        <p className="relative w-[800px] font-normal text-[#15291f] text-xl text-center tracking-[0] leading-8">
          Our Colour Analysis Tool blends innovation, style & AI to reveal your perfect colours, using your skin, hair, eye colours,
          and undertones. 
        </p>
      </div>
      <button className="mt-10 box-border items-center justify-center gap-3 px-8 py-3 bg-p border-2 border-pink-400 rounded-3xl  hover:bg-white hover:text-pink-600 hover:font-bold transition-colors duration-300">
        <div className="relative w-fit font-semibold text-black text-lg text-center leading-6 whitespace-nowrap">
            Try Our Tool!
        </div>
      </button>
      <img className="absolute w-[82px] h-[114px] top-[236px] left-[1455px]" alt="Remove bg" src={bulb2} />
      <img className="absolute w-[376px] h-[507px] top-[209px] left-[1571px]" alt="Line" src={line2} />
      <img className="absolute w-[91px] h-[115px] top-[257px] left-[185px]" alt="Remove bg" src={bulb1} />
    </div>
  );
};

export default Hero 