import eyecolour from "../../assets/images/eyecolour1.png";
import haircolour from "../../assets/images/haircolour.png";
import skincolour from "../../assets/images/skincolour.png";
import undertones from '../../assets/images/undertones.png';
import circleline from "../../assets/images/line.svg";

export default function Factors() {
    return (
      <div className="min-h-screen flex flex-col items-center">

        {/* Header Section */}
        <div className="w-full py-5">
          <div className="container mx-auto px-4 text-center">
            <h1 className="font-extrabold text-black text-[56px] text-center mb-2">Factors considered</h1>
            <img className="relative w-[650px] h-[134px] top-[-120px] mx-auto" src={circleline} alt="circle line"/>
            <p className="mt-[-60px] font-normal text-2xl w-[70%] mx-auto leading-normal">
              To provide truly personalized colour recommendations, our AI-assisted colour analysis tool takes into account key aspects of your appearance such as <span className="font-semibold text-pink-600">skin colour, undertone, hair colour, and eye colour </span>using mediapipe landmarker and gives your personalized colour palette & product recommendations.
            </p>
          </div>
        </div>

        {/* Factors Grid */}
        <div className="container mx-auto px-4 md:px-8 lg:px-16 xl:px-32 py-10">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-36 max-w-4xl mx-auto">

            {/* Skin Colour Section */}
            <div className="flex flex-col items-center text-center space-y-10">
              <h2 className="text-3xl font-bold">1# Skin Colour</h2>
              <p className="text-lg leading-normal">
                Skin colour forms the foundation for colour analysis. The tool identifies the general tone of an individual's skin which is crucial in determining which colours enhance your natural beauty rather than overshadow it.
              </p>
              <div className="relative flex items-center justify-center">
              <div className="absolute top-[30px] left-[30px] bg-pink-500 h-[250px] w-[350px] rounded-2xl z-0"/>
                <img
                  src={skincolour}
                  alt="Skin Colour Analysis"
                  className="relative z-10 rounded-lg w-[350px] h-[250px] mx-auto"
                />
              </div>
            </div>

            {/* Undertones Section */}
            <div className="flex flex-col items-center text-center space-y-10">
              <h2 className="text-3xl font-bold">2# Undertones</h2>
              <p className="text-lg leading-normal">
                Undertones are subtle hues beneath the skin's surface and are classified as warm, cool, or neutral. Undertones ensures that the recommended colours work harmoniously with your complexion and features.
              </p>
              <div className="relative flex items-center justify-center">
              <div className='absolute top-[30px] left-[30px] bg-pink-500 h-[250px] w-[320px] rounded-2xl'/>
                <img
                  src={undertones}
                  alt="Undertones Analysis"
                  className="z-10 rounded-lg w-[320px] h-[250px] mx-auto"
                />
              </div>
            </div>

            {/* Hair Colour Section */}
            <div className="flex flex-col items-center text-center space-y-10">
              <h2 className="text-3xl font-bold">3# Hair Colour</h2>
              <p className="text-lg leading-normal">
                Hair colour plays a vital role in colour matching. Whether you have jet black or fiery red hair, the analysis tool takes this into account to recommend shades that complement the overall contrast between your hair and skin.
              </p>
              <div className="relative flex items-center justify-center">
              <div className='absolute top-[30px] left-[30px] bg-pink-500 h-[250px] w-[300px] rounded-2xl'></div>
              <img
                  src={haircolour}
                  alt="Hair Colour Analysis"
                  className="z-10 rounded-lg w-[300px] h-[250px] mx-auto"
                />
              </div>
            </div>

            {/* Eye Colour Section */}
            <div className="flex flex-col items-center text-center space-y-10">
              <h2 className="text-3xl font-bold">4# Eye Colour</h2>
              <p className="text-lg leading-normal">
                Eyes are one of the most defining facial features and their natural hue can influence how colours look on you. Our tool examines your eye colour to determine which shades will make your eyes pop!
              </p>
              <div className="relative flex items-center justify-center">
              <div className='absolute top-[30px] left-[30px] bg-pink-500 h-[250px] w-[380px] rounded-2xl'/>
                <img
                  src={eyecolour}
                  alt="Eye Colour Analysis"
                  className="z-10 rounded-lg w-[380px] h-[250px] mx-auto"
                />
              </div>
            </div>

          </div>
        </div>
      </div>
    );
}
