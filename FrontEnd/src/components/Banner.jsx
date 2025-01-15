import React from 'react'
import { motion } from 'framer-motion'
import { Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'  // Import Link from react-router-dom

export default function Banner() {
  return (
    <div className="bg-gradient-to-r from-pink-200 to-purple-200 py-20 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      <div className="absolute inset-0 bg-white bg-opacity-10 backdrop-filter backdrop-blur-sm"></div>
      <motion.div 
        className="max-w-7xl mx-auto text-center relative z-10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <motion.h1 
          className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-gray-900 mb-8 leading-tight"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Discover your perfect colour palette{' '}
          <br className="hidden sm:inline" />
          <span className="text-pink-600">
            in seconds
          </span>{' '}
          with our advanced{' '}
          <br className="hidden sm:inline" />
          seasonal colour analysis.
        </motion.h1>
        <motion.p 
          className="mt-4 text-xl text-gray-700 max-w-3xl mx-auto mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          Unlock your unique style and enhance your wardrobe with personalized colour recommendations.
        </motion.p>
        
        {/* Wrap the button inside Link component */}
        <Link to="/tool">  {/* Update the path to the actual route for your tool page */}
          <motion.button 
            className="mt-8 inline-flex items-center justify-center px-8 py-4 border border-transparent text-base font-medium rounded-full text-white bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 md:py-4 md:text-lg md:px-10 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Sparkles className="w-5 h-5 mr-2" />
            Try Our Tool!
          </motion.button>
        </Link>
      </motion.div>
      
      {/* Decorative elements */}
      <motion.div
        className="absolute -bottom-16 -left-16 w-64 h-64 bg-pink-300 rounded-full mix-blend-multiply filter blur-3xl opacity-70"
        animate={{
          scale: [1, 1.2, 1],
          rotate: [0, 90, 0],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />
      <motion.div
        className="absolute -top-16 -right-16 w-64 h-64 bg-purple-300 rounded-full mix-blend-multiply filter blur-3xl opacity-70"
        animate={{
          scale: [1, 1.3, 1],
          rotate: [0, -90, 0],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />
      <motion.div
        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-yellow-200 rounded-full mix-blend-multiply filter blur-3xl opacity-30"
        animate={{
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />
    </div>
  )
}
