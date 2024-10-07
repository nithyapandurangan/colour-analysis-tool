import React from 'react'

export default function Banner() {
  return (
    <div className="bg-gradient-to-r from-[#FFADF2] to-[#FF60E6] py-10 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      <div className="absolute inset-0 bg-white bg-opacity-10 backdrop-filter backdrop-blur-sm"></div>
      <div className="max-w-7xl mx-auto text-center relative z-10">
        <h1 className="text-4xl sm:text-5xl md:text-6xl [font-family:'Inter-Bold',Helvetica] font-bold text-black mb-8 leading-tight">
          Discover your perfect colour palette <br className="hidden sm:inline" />
          <span className="text-pink-800">in seconds</span> with our advanced <br className="hidden sm:inline" />
          seasonal colour analysis.
        </h1>
        <p className="mt-4 text-xl text-black text-opacity-90 max-w-3xl mx-auto mb-8">
          Unlock your unique style and enhance your wardrobe with personalized color recommendations.
        </p>
        <button className="mt-2 box-border items-center justify-center gap-3 px-8 py-4 bg-white border-2 border-pink-400 rounded-full hover:bg-pink-600 hover:text-white hover:font-bold transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl">
          <div className="relative w-fit font-semibold text-pink-600 hover:text-white text-lg text-center leading-6 whitespace-nowrap">
            Try Our Tool!
          </div>
        </button>
      </div>
      <div className="absolute bottom-0 left-0 w-64 h-64 bg-pink-300 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
      <div className="absolute top-0 right-0 w-64 h-64 bg-purple-300 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000"></div>
    </div>
  )
}