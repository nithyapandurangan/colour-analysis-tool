import { motion } from 'framer-motion';
import { Sparkles } from 'lucide-react';

const ProblemStatement = () => {
  return (
    <motion.section 
      className="relative flex flex-col min-h-screen items-center justify-center gap-16 overflow-hidden py-20"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1 }}
    >


      {/* Problem Statement Heading */}
      <motion.div 
        className="relative z-10 flex items-center gap-4"
        initial={{ y: -20 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Sparkles className="w-8 h-8 text-pink-500" />
        <h1 className="font-extrabold text-black text-4xl md:text-5xl lg:text-6xl text-center leading-tight">
          Our Problem Statement
        </h1>
        <Sparkles className="w-8 h-8 text-pink-500" />
      </motion.div>

      {/* Problem Statement Text Content */}
      <div className="flex flex-col gap-8 text-center max-w-4xl px-4">
        <motion.div
          className="bg-white/80 backdrop-blur-sm p-8 rounded-2xl shadow-xl transition-all duration-300 hover:shadow-2xl hover:-translate-y-1"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <p className="font-normal text-gray-800 text-xl md:text-2xl tracking-wide leading-relaxed">
            Many individuals <span className="font-semibold text-pink-600">struggle to find colours that complement</span> their features like skin tone, hair, and eye colour, often <span className="font-semibold text-pink-600">feeling overwhelmed by choices in clothing and cosmetics.</span>
          </p>
        </motion.div>
        
        <motion.div
          className="bg-white/80 backdrop-blur-sm p-8 rounded-2xl shadow-xl transition-all duration-300 hover:shadow-2xl hover:-translate-y-1"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          <p className="font-normal text-gray-800 text-xl md:text-2xl tracking-wide leading-relaxed">
            This project aims to solve this by developing an AI-assisted colour analysis tool that <span className="font-semibold text-pink-600">offers personalized colour palettes and product recommendations,</span> enhancing users' style and confidence.
          </p>
        </motion.div>
      </div>

      {/* Decorative elements */}
      
    </motion.section>
  );
};

export default ProblemStatement;
