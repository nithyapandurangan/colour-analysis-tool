import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "../components/ui/Card"

import FormOptions from "../components/Skincare/FormOptions"
import NewForm from "../components/Skincare/NewForm"
import ExistingForm from "../components/Skincare/ExistingForm"
import RecommendationList from "../components/Skincare/RecommendationList"

export default function Skincare() {
  const [option, setOption] = useState("")
  const [recommendations, setRecommendations] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const handleSubmit = async (formData) => {
    setError("")
    setIsLoading(true)
    try {
      let url = ""
  
      if (option === "new") {
        // API for Start from Scratch (port 5001)
        url = `http://127.0.0.1:5001/recommend?skin_type=${formData.skinType}&concern=${formData.concern}&product_type=${formData.productType}`
      } else if (option === "existing") {
        // API for Find Similar Products (new API on port 5002)
        url = `http://127.0.0.1:5002/recommend?existing_product=${formData.existingProduct}`
      }
  
      const response = await fetch(url)
      if (!response.ok) throw new Error("Failed to fetch recommendations")
  
      const data = await response.json()
      setRecommendations(data.length > 0 ? data : [])
      if (data.length === 0) setError("No matching products found. Try adjusting your filters.")
    } catch (error) {
      setError("Something went wrong while fetching recommendations.")
    } finally {
      setIsLoading(false)
    }
  }
  

  return (
    <div className="min-h-screen bg-gradient-to-b from-pink-50 to-white">
      <div className="container mx-auto py-12 px-4 max-w-6xl">
      <div className="relative mb-12 text-center">
        <h1 className="text-4xl md:text-5xl font-bold text-pink-800 mb-2">Skin Care Recommender</h1>
        <p className="text-black text-lg max-w-2xl mx-auto my-6">
          Discover your perfect skincare product match with our personalized recommendation engine
        </p>
      </div>
        <Card className="border-none shadow-2xl overflow-hidden bg-white rounded-2xl">
          <CardHeader className="bg-gradient-to-r from-pink-600 to-pink-800 rounded-t-2xl py-8">
            <div className="flex items-center justify-center mb-2">
              <CardTitle className="text-3xl md:text-4xl font-bold text-white">🌸 Discover Your Skin's Match</CardTitle>
            </div>
            <CardDescription className="text-center text-pink-100 text-[16px] mt-5">
              Let us help you find your perfect skincare routine
            </CardDescription>
          </CardHeader>
          <CardContent className="pt-8 px-6 md:px-10">
            <FormOptions option={option} setOption={setOption} />
            {option === "new" && <NewForm handleSubmit={handleSubmit} />}
            {option === "existing" && <ExistingForm handleSubmit={handleSubmit} />}
            <RecommendationList recommendations={recommendations} isLoading={isLoading} error={error} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
