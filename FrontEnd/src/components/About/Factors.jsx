import { motion } from 'framer-motion';
import { Palette, Droplet, Scissors, Eye } from 'lucide-react';

import eyecolour from "../../assets/images/eyecolour1.png";
import haircolour from "../../assets/images/haircolour.png";
import skincolour from "../../assets/images/skincolour.png";
import undertones from '../../assets/images/undertones.png';

const FactorCard = ({ title, description, image, icon: Icon, delay }) => (
  <motion.div 
    className="bg-white rounded-2xl shadow-xl overflow-hidden transition-all duration-300 hover:shadow-2xl hover:-translate-y-1"
    initial={{ opacity: 0, y: 50 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
  >
    <div className="p-6 space-y-4">
      <div className="flex items-center space-x-3">
        <div className="bg-pink-100 p-2 rounded-full">
          <Icon className="w-6 h-6 text-pink-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800">{title}</h2>
      </div>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </div>
    <div className="relative h-48 w-full mb-4"> {/* Added margin bottom here */}
      <img 
        src={image} 
        alt={title} 
        className="absolute inset-0 object-contain w-full h-full" 
      />
    </div>
  </motion.div>
);

export default function Factors() {
  const factors = [
    {
      title: "Skin Colour",
      description: "Skin colour forms the foundation for colour analysis. The tool identifies the general tone of an individual's skin which is crucial in determining which colours enhance your natural beauty rather than overshadow it.",
      image: skincolour,
      icon: Palette,
    },
    {
      title: "Undertones",
      description: "Undertones are subtle hues beneath the skin's surface and are classified as warm, cool, or neutral. Undertones ensures that the recommended colours work harmoniously with your complexion and features.",
      image: undertones,
      icon: Droplet,
    },
    {
      title: "Hair Colour",
      description: "Hair colour plays a vital role in colour matching. Whether you have jet black or fiery red hair, the analysis tool takes this into account to recommend shades that complement the overall contrast between your hair and skin.",
      image: haircolour,
      icon: Scissors,
    },
    {
      title: "Eye Colour",
      description: "Eyes are one of the most defining facial features and their natural hue can influence how colours look on you. Our tool examines your eye colour to determine which shades will make your eyes pop!",
      image: eyecolour,
      icon: Eye,
    },
  ];

  return (
    <div className="min-h-screen bg-white py-20"> 
      <div className="container mx-auto px-4">
        <motion.div 
          className="text-center mb-10"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="font-extrabold text-black text-5xl md:text-6xl mb-6"> 
            Factors Considered
          </h1>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {factors.map((factor, index) => (
            <FactorCard key={factor.title} {...factor} delay={index * 0.1} />
          ))}
        </div>
      </div>

      {/* Decorative elements */}
      <motion.div
        className="fixed top-20 left-10 w-64 h-64 rounded-full opacity-20 blur-3xl"
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
        className="fixed bottom-20 right-10 w-96 h-96 rounded-full opacity-20 blur-3xl"
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
    </div>
  );
}
