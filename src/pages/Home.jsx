import Hero from "../components/Home/Hero"
import Description from "../components/Home/Description"
import Intro from "../components/Home/Intro"
import Banner from "../components/Banner"

const Home = () => {
    return (
        <div className="relative">
            <Hero />
            <Intro />
            <Description />
            <Banner />

        </div>
    )
}

export default Home 