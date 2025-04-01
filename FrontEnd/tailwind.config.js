/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./public/index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
    "./components/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)"],
      },
      fontSize: {
        sm: ['13.3px', '16px'],
        base: ['16px', '16px'],
        lg: ['19.2px', '24px'],
        xl: ['23.04px', '24px'],
        '2xl': ['27px', '32px'],
        '3xl': ['33px', '40px'],
        '4xl': ['40px', '40px'],
        '5xl': ['48px', '48px']
      },
      colors: {
        'p':'#FF60E6',
        'p-light':'#FFADF2'
      }
    },
  },
  plugins: [],
}

