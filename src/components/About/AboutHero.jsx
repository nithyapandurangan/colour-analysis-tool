import { IoMdCheckboxOutline } from "react-icons/io";
import HeroImages from "../About/HeroImages";

const AboutHero = () => {
    return (
        <div id='hero' className='relative flex px-12 pt-10 h-screen'>
            <div className='flex-1 relative flex flex-col items-start z-5'>
                <h1 className='font-extrabold pb-8 py-8 text-6xl font-bold leading-tight text-left'>
                    Unlock your <span className="text-pink-600">true colours</span> with our expert analysis – tailored to bring out the <span className="text-pink-600">best in you.</span>
                </h1>

                <div className='py-3 flex items-center justify-start z-10'>
                    <IoMdCheckboxOutline size={50} className='mr-4 text-black' />
                    <p className='text-base xl:text-lg text-left'>AI-Powered Precision: Using AI, the tool ensures accurate color analysis and improves <br /> over time by incorporating user feedback to refine colour palette recommendations.</p>
                </div>
                <div className='py-3 flex items-center justify-start z-10'>
                    <IoMdCheckboxOutline size={50} className='mr-4 text-black' />
                    <p className='text-base xl:text-lg text-left'>Personalized Seasonal Color Classification: The tool analyzes facial features such as <br /> skin tone, hair color, and eye color to determine your seasonal colour palette.</p>
                </div>
                <div className='py-3 flex items-center justify-start z-10'>
                    <IoMdCheckboxOutline size={50} className='mr-4 text-black' />
                    <p className='text-base xl:text-lg text-left'>Tailored Fashion and Cosmetic Recommendations: Based on the user's seasonal <br /> classification, the tool suggests clothing and cosmetic colors that harmonize <br /> with your natural features.</p>
                </div>

                <div className='pt-4 flex flex-col items-center md:items-start justify-start w-full'>
                    <div className='w-full py-5 flex justify-start'>
                        <p className='text-base md:text-lg lg:text-xl font-bold'>Our AI-Assisted Colour Analysis Tool is your perfect partner <br /> for your self-expression & style.​</p>
                    </div>
                    <div className='w-full py-2 flex flex-row justify-start'>
                        <button className="mt-3 box-border items-center justify-center gap-3 px-8 py-3 bg-p border-2 border-pink-400 rounded-3xl  hover:bg-white hover:text-pink-600 hover:font-bold transition-colors duration-300">
                            <div className="relative w-fit font-semibold text-black text-lg text-center leading-6 whitespace-nowrap">
                                Try Our Tool!
                            </div>
                        </button>
                    </div>
                </div>
            </div>  
            <div className='max-2xl:hidden flex justify-center w-[50%] z-20 py-40'>
                <HeroImages />
            </div>
        </div>
    );
};

export default AboutHero;
