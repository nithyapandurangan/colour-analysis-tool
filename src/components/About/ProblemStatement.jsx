import hands1 from "../../assets/images/image.png";
import hands2 from "../../assets/images/hands_2.png";
import line1 from "../../assets/images/line.png";
import line2 from "../../assets/images/line-2.png";
import circleline from "../../assets/images/line.svg";

const ProblemStatement = () => {
  return (
    <>
      <div className="relative flex flex-col h-screen items-center justify-center gap-16 overflow-hidden">
        {/* Background and Decorations */}
        <img className="absolute w-[440px] h-[804px] top-[5px] left-[-142px]" alt="Line" src={line1} />
        <img className="absolute w-[414px] h-[217px] -top-2 left-[-142px]" alt="Hands" src={hands1} />
        <img className="absolute w-[403px] h-[426px] top-[415px] left-[1544px]" alt="Hands" src={hands2} />
        <img className="absolute w-[376px] h-[507px] top-[209px] left-[1571px]" alt="Line" src={line2} />
        <img className="absolute w-[715px] h-[145px] top-24 left-404" src={circleline} alt="circle line" />

        {/* Problem Statement Heading */}
        <p className="mt-[-120px] relative font-extrabold text-black text-[56px] text-center">
          Our Problem Statement
        </p>

        {/* Problem Statement Text Content */}
        <div className="flex flex-col gap-6 text-center max-w-[1000px]">
          <p className="font-normal text-black text-3xl tracking-normal leading-relaxed">
            Many individuals <span className="font-semibold text-pink-600">struggle to find colours that complement</span> their features like skin tone, hair, and eye colour, often <span className="font-semibold text-pink-600">feeling overwhelmed by choices in clothing and cosmetics.</span>
          </p>
          <p className="mt-2 font-normal text-black text-3xl tracking-normal leading-relaxed">
            This project aims to solve this by developing an AI-assisted colour analysis tool that <span className="font-semibold text-pink-600">offers personalized colour palettes and product recommendations,</span> enhancing users' style and confidence.
          </p>
        </div>
      </div>
    </>
  );
};

export default ProblemStatement;
