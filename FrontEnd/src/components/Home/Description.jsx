import { motion } from 'framer-motion'
import React, { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from './Card'
import { AlertCircle, Lightbulb, Target } from 'lucide-react'
import line from "../../assets/images/Vector-light.png"

const DescriptionCard = ({ title, subtitle, content, icon: Icon, isHovered, onMouseEnter, onMouseLeave }) => {
  return (
    <motion.div
      initial={{ opacity: 1 }}
      whileHover={{ opacity: 1, scale: 1.05 }}  // Slight scaling effect on hover
      className={`flex justify-center items-center ${isHovered ? 'opacity-100' : 'opacity-30'}`} // Hide others by reducing opacity
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <Card className="h-full bg-white border-p text-black shadow-lg transform hover:translate-y-2 transition-all duration-300 ease-in-out">
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Icon className="w-6 h-6 text-pink-600" />
            <CardTitle className="font-semibold">
              {title} <span className="text-pink-600">{subtitle}</span>
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-black text-lg">{content}</p>
        </CardContent>
      </Card>
    </motion.div>
  )
}

const Description = () => {
  const [hoveredCard, setHoveredCard] = useState(null);

  return (
    <section className="w-full min-h-screen flex items-center justify-center py-12 md:py-24 lg:py-32 bg-white mt-20">
      <div className="absolute container px-4 md:px-6">
        <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl text-center mb-12 text-black">
          Our Vision
        </h2>
        <img src={line} alt="Vector" className="absolute top-[10%] left-[45%]" />
        
        <div className="mt-20 grid gap-6 md:gap-8 md:grid-cols-2 lg:grid-cols-3">
          <DescriptionCard
            title="CHALLENGE"
            subtitle="FACED"
            content="Many individuals struggle to identify colours that flatter their natural features. Choosing the wrong colours in clothing or cosmetics can make them look tired or washed out, leading to a lack of confidence in personal style. Additionally, with the overwhelming variety of products available, finding the perfect match becomes increasingly difficult."
            icon={AlertCircle}
            isHovered={hoveredCard === "challenge"}
            onMouseEnter={() => setHoveredCard("challenge")}
            onMouseLeave={() => setHoveredCard(null)}
          />
          <DescriptionCard
            title="OUR"
            subtitle="SOLUTION"
            content="Our AI-assisted tool leverages facial features such as skin tone, hair, and eye colour to provide personalized seasonal colour classifications. By offering tailored product recommendations that align with the user's unique colour profile, the system simplifies decision-making in fashion and cosmetics, ensuring users always look their best."
            icon={Lightbulb}
            isHovered={hoveredCard === "solution"}
            onMouseEnter={() => setHoveredCard("solution")}
            onMouseLeave={() => setHoveredCard(null)}
          />
          <DescriptionCard
            title="OUR"
            subtitle="GOAL"
            content="To empower individuals with a tool that enhances their personal style by providing precise, real-time colour analysis and product recommendations. The ultimate goal is to bridge technology and fashion, making personalized styling accessible and intuitive for everyone."
            icon={Target}
            isHovered={hoveredCard === "goal"}
            onMouseEnter={() => setHoveredCard("goal")}
            onMouseLeave={() => setHoveredCard(null)}
          />
        </div>
      </div>
    </section>
  )
}

export default Description
