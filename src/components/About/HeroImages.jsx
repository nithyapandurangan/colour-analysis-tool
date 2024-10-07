import { ScrollParallax } from "react-just-parallax"
import Colouranalysis from "../../assets/images/colouranaly.gif"
import { useRef } from 'react';

const HeroImages = () => {
  const parallaxRef = useRef(null);

  return (
    <div className='relative w-[100%]' ref={parallaxRef}>
        <ScrollParallax isAbsolutelyPositioned>
            <img src={Colouranalysis} alt='gif' className='absolute z-10 rounded-2xl object-cover h-[450px] w-[800px]'/>
            <div className='absolute left-[30px] top-[30px] z-5 bg-pink-700 h-[450px] w-[800px] rounded-2xl'></div>
            {/* <img src={heroImg2} className='absolute top-[200px] left-[200px] z-10 rounded-3xl object-cover h-[300px] w-[300px]'/>
            <div className='absolute left-[230px] top-[230px] z-5 bg-pink-700 h-[300px] w-[300px] rounded-3xl'></div> */}
        </ScrollParallax>
    </div>
  )
}

export default HeroImages
