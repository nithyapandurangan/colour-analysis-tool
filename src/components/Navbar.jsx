import React from 'react'
import Logo from '../assets/Logo.png'

export default function Navbar() {
  return (
    <nav className="bg-white shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <img href="/" src={Logo} alt='logo' />
          </div>

          {/* Main navigation items */}
          <div className="hidden md:flex items-center justify-center flex-1">
            <a href="/" className="mx-6 text-lg font-normal text-black hover:text-pink-600 hover:font-bold">
              Home
            </a>
            <a href="/about" className="mx-6 text-lg font-normal text-black hover:text-pink-600 hover:font-bold">
              About Us
            </a>
            <a href="/work" className="mx-6 text-lg font-normal text-black hover:text-pink-600 hover:font-bold">
              How we work
            </a>
            <a href="/team" className="mx-6 text-lg font-normal text-black hover:text-pink-600 hover:font-bold">
              Our Team
            </a>
          </div>

          {/* Our Tool button */}
          <div className="hidden md:block">
            <a
              href="/tool"
              className="inline-block px-4 py-2 border-2 border-pink-400 rounded-full bg-p text-black font-normal hover:bg-white hover:text-pink-600 hover:font-bold transition-colors duration-300"
            >
              Our Tool
            </a>
          </div>


          {/* //not optimized yet */}

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button className="text-black hover:text-pink-600 focus:outline-none focus:text-pink-600">
              <span className="sr-only">Open main menu</span>
              <svg
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu, show/hide based on menu state */}
      <div className="md:hidden">
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
          <a href="/" className="block px-3 py-2 text-base font-medium text-gray-500 hover:text-blue-600">
            Home
          </a>
          <a href="/about" className="block px-3 py-2 text-base font-medium text-gray-500 hover:text-blue-600">
            About Us
          </a>
          <a href="/how-we-work" className="block px-3 py-2 text-base font-medium text-gray-500 hover:text-blue-600">
            How we work
          </a>
          <a href="/team" className="block px-3 py-2 text-base font-medium text-gray-500 hover:text-blue-600">
            Our Team
          </a>
          <a href="/tool" className="block px-3 py-2 text-base font-medium text-gray-500 hover:text-blue-600">
            Our Tool
          </a>
        </div>
      </div>
    </nav>
  )
}