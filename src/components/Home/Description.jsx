import line1 from "../../assets/images/line.png";
import line2 from "../../assets/images/line-2.png";


const Description = () => {
    return (
        <div className='relative w-full p-10'>
            <div className='w-full flex flex-col items-center mt-10'>
            <img className="absolute w-[440px] h-[804px] top-[5px] left-[-200px]" alt="Line" src={line1} />
                <div className='xl:w-3/4 flex flex-col xl:flex-row py-10'>
                <div className="uppercase text-left xl:w-1/3 text-3xl lg:text-4xl xl:text-4xl md:text-4xl font-semibold xl:text-right px-10 py-5 lg:border-r-4 lg:border-black">
                        CHALLENGE <br /> 
                        <span className="text-b-900 font-medium">FACED</span>
                    </div>
                    <div className="md:w-4/5 xl:w-1/2 text-lg text-left pl-10">
                    Many individuals struggle to identify colours that flatter their natural features. Choosing the wrong colours in clothing or cosmetics can make them look tired or washed out, leading to a lack of confidence in personal style. Additionally, with the overwhelming variety of products available, finding the perfect match becomes increasingly difficult.​                        <br />

                    </div>
                </div>

                <div className='xl:w-3/4 flex flex-col xl:flex-row py-10'>
                    <div className="uppercase text-left xl:w-1/3 text-3xl lg:text-4xl xl:text-4xl md:text-4xl font-medium xl:text-right px-10 py-5 lg:border-r-4 lg:border-black">
                        OUR <br /> 
                        <span className="text-b-900 font-semibold">SOLUTION</span>
                    </div>
                    <div className="md:w-4/5 xl:w-1/2 text-lg text-left pl-10">
                    Our AI-assisted tool leverages facial features such as skin tone, hair, and eye colour to provide personalized seasonal colour classifications. By offering tailored product recommendations that align with the user’s unique colour profile, the system simplifies decision-making in fashion and cosmetics, ensuring users always look their best.​
                    </div>
                </div>
                <div className='xl:w-3/4 flex flex-col xl:flex-row py-10'>
                    <div className="uppercase text-left xl:w-1/3 text-3xl lg:text-4xl xl:text-4xl md:text-4xl font-medium xl:text-right px-10 py-5 lg:border-r-4 lg:border-black">
                        OUR <br /> 
                        <span className="text-b-900 font-semibold">GOAL</span>
                    </div>
                    <div className="md:w-4/5 xl:w-1/2 text-lg text-left pl-10">
                    To empower individuals with a tool that enhances their personal style by providing precise, real-time colour analysis and product recommendations. The ultimate goal is to bridge technology and fashion, making personalized styling accessible and intuitive for everyone.​
                    </div>
                </div>
            </div>
            <img className="absolute w-[376px] h-[507px] top-[50px] left-[1571px]" alt="Line" src={line2} />

        </div>
    );
};

export default Description;
