// Recommendation Based on Current Product

import { Input } from "../ui/Input"
import { Label } from "../ui/Label"
import { Button } from "../ui/Button"
import { useState } from "react"

export default function ExistingForm({ handleSubmit }) {
  const [existingProduct, setExistingProduct] = useState("")

  return (
    <form onSubmit={(e) => {
      e.preventDefault()
      handleSubmit({ existingProduct })
    }}
      className="space-y-6"
    >
      <div className="space-y-2">
        <Label htmlFor="existingProduct" className="text-pink-800 text-[16px]">
          Enter Product That You're Currently Using
        </Label>
        <Input
          id="existingProduct"
          placeholder="Enter Product Name"
          value={existingProduct}
          onChange={(e) => setExistingProduct(e.target.value)}
          className={'w-full'}
          required
        />
      </div>
      <Button type="submit" className="bg-pink-600 hover:bg-pink-700 w-full">
        Find Similar Products
      </Button>
    </form>
  )
}
