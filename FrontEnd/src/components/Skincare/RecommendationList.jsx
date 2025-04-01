//Product Recommendations Template

import { Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "../ui/Alert"
import { Card, CardContent, CardFooter } from "../ui/Card"

export default function RecommendationList({ recommendations, isLoading, error }) {
  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-6">
        <Loader2 className="animate-spin h-8 w-8 text-pink-500" />
      </div>
    )
  }

  if (error) {
    return (
      <Alert className="bg-red-100 text-red-700">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 py-6">
      {recommendations.map((product, index) => (
        <Card key={index} className="shadow-lg border-pink-100">
          <CardContent className="p-4">
            {product.picture_src && (
              <img
                src={product.picture_src}
                alt={product.name}
                loading="lazy"
                className="w-full h-48 object-cover rounded-lg mb-4"
              />
            )}
            <h4 className="text-lg font-semibold text-black">{product.product_name}</h4>
            <p className="text-sm text-black">{product.notable_effects}</p>
          </CardContent>

          <CardFooter className="flex justify-between items-center p-4 border-t">
            <span className="text-pink-500 font-bold">{product.price}</span>
            <button className="text-sm bg-pink-500 text-white py-1 px-3 rounded-md">Buy Now</button>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
