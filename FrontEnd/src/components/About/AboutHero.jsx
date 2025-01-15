import { motion } from 'framer-motion'
import { CheckCircle } from 'lucide-react'
import { Link } from 'react-router-dom'  // Import Link from react-router-dom

const FeatureItem = ({ children }) => (
  <motion.div 
    className='flex items-start space-x-4 mb-6'
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
    viewport={{ once: true }}
  >
    <CheckCircle className='w-6 h-6 text-pink-600 flex-shrink-0 mt-1' />
    <p className='text-base xl:text-lg'>{children}</p>
  </motion.div>
)

const Button = ({ children, className, size }) => {
  return (
    <button 
      className={`px-6 py-4 bg-p text-black border-2 border-p rounded-full hover:bg-white hover:text-pink-600 transition-colors duration-300 ${className} ${size === 'lg' ? 'text-lg' : ''}`}
    >
      {children}
    </button>
  )
}

const AboutHero = () => {
  return (
    <section id='hero' className='relative px-4 md:px-12 md:py-10 min-h-screen bg-white'>
      <div className='max-w-7xl mx-auto'>
        <motion.h1 
          className='font-extrabold text-4xl md:text-5xl lg:text-6xl leading-tight text-left mb-12'
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Unlock your <span className="text-pink-600">true colours</span> <br className="hidden md:inline" />
          with our expert analysis– tailored <br className="hidden md:inline" />
          to bring out the <span className="text-pink-600">best in you.</span>
        </motion.h1>

        <div className='space-y-6'>
          <FeatureItem>
            AI-Powered Precision: Using AI, the tool ensures accurate color analysis and improves 
            over time by <br /> incorporating user feedback to refine colour palette recommendations.
          </FeatureItem>
          <FeatureItem>
            Personalized Seasonal Color Classification: The tool analyzes facial features such as 
            skin tone,<br /> hair color, and eye color to determine your seasonal colour palette.
          </FeatureItem>
          <FeatureItem>
            Tailored Fashion and Cosmetic Recommendations: Based on the user's seasonal 
            classification, the <br />tool suggests clothing and cosmetic colors that harmonize 
            with your natural features.
          </FeatureItem>
        </div>

        <motion.div 
          className='mt-12 space-y-6'
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          viewport={{ once: true }}
        >
          <p className='text-lg md:text-xl lg:text-2xl font-bold'>
            Our AI-Assisted Colour Analysis Tool is your perfect partner <br className="hidden md:inline" />
            for your self-expression & style.​
          </p>

          {/* Wrap the button inside Link component */}
          <Link to="/tool">  {/* Change '/tool' to your actual route if it's different */}
            <Button 
              className="mt-6"
              size="lg"
            >
              <span className="text-lg font-semibold">Try Our Tool!</span>
            </Button>
          </Link>
        </motion.div>
      </div>
    </section>
  )
}

export default AboutHero
