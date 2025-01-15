import AboutHero from "../components/About/AboutHero";
import ProblemStatement from "../components/About/ProblemStatement";
import Factors from "../components/About/Factors";
import Banner from "../components/Banner";

const About = () => {
    return (
        <div className='relative h-screen'>
            <AboutHero />
            <ProblemStatement/>
            <Factors />
            <Banner />
        </div>
    );
};

export default About;
