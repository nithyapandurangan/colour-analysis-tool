import { motion } from 'framer-motion';
import { Users } from 'lucide-react';

import niths from '../assets/images/niths.jpeg';
import sud from '../assets/images/sudi.jpeg';
import trish from '../assets/images/trish.jpeg';
import shri from '../assets/images/shri.jpeg';

const teamMembers = [
  { 
    name: 'Nithya P', 
    image: niths, 
    role: 'Full Stack Developer', 
    github: 'https://github.com/nithyapandurangan', 
    linkedin: 'www.linkedin.com/in/nithya-pandurangan' 
  },
  { 
    name: 'Sudiksaa SV', 
    image: sud, 
    role: 'Data Scientist', 
    github: 'https://github.com/Sudiksaa2510', 
    linkedin: 'https://www.linkedin.com/in/sudiksaa-velan-662269322/' 
  },
  { 
    name: 'Trisha B', 
    image: trish, 
    role: 'ML Engineer', 
    github: 'https://github.com/trishabala', 
    linkedin: 'https://www.linkedin.com/in/trishabalakrishnan/' 
  },
  { 
    name: 'Shrinithi V', 
    image: shri, 
    role: 'Data Scientist', 
    github: 'https://github.com/shrinithi', 
    linkedin: 'https://www.linkedin.com/in/shrinithi-vijayaraghavan-101126231/' 
  },
];

const TeamMember = ({ name, image, role, github, linkedin, index }) => (
  <motion.div 
    className="flex flex-col items-center p-6 bg-white rounded-xl shadow-lg transition-all duration-300 hover:shadow-xl hover:-translate-y-1"
    initial={{ opacity: 0, y: 50 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay: index * 0.1 }}
  >
    <div className="relative w-48 h-48 mb-6 overflow-hidden rounded-full">
      <img src={image || "/placeholder.svg"} alt={name} className="w-full h-full object-cover" />
    </div>
    <h2 className="text-2xl font-bold text-gray-800 mb-2">{name}</h2>
    <p className="text-lg text-gray-600 mb-4">{role}</p>
    <div className="flex space-x-4">
      {/* LinkedIn Icon */}
      <a href={linkedin} className="text-gray-700 hover:text-gray-900" target="_blank" rel="noopener noreferrer">
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
        </svg>
      </a>
      {/* GitHub Icon */}
      <a href={github} className="text-gray-700 hover:text-gray-900" target="_blank" rel="noopener noreferrer">
        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
        </svg>
      </a>
    </div>
  </motion.div>
);

export default function Team() {
  return (
    <div className="min-h-screen py-20">
      <div className="container mx-auto px-4">
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center justify-center mb-4">
            <Users className="w-12 h-12 text-black mr-4" />
            <h1 className="text-5xl font-extrabold">
              Our Team
            </h1>
          </div>

        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {teamMembers.map((member, index) => (
            <TeamMember key={member.name} {...member} index={index} />
          ))}
        </div>
      </div>

      {/* Decorative elements */}
      <motion.div
        className="fixed top-20 left-10 w-64 h-64 rounded-full blur-3xl"
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
