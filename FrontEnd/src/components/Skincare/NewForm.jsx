//Start From Scratch Option

import { Select, SelectItem,} from "../ui/Select";
import { Label } from "../ui/Label";
import { Button } from "../ui/Button";
import { useState } from "react";

export default function NewForm({ handleSubmit }) {
  const [formData, setFormData] = useState({
    skinType: "",
    concern: "",
    productType: "",
  });

  const handleChange = (field, value) => {
    setFormData({ ...formData, [field]: value });
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        handleSubmit(formData);
      }}
      className="space-y-6"
    >
      <div className="grid gap-6 md:grid-cols-3">
        {/* Skin Type */}
        <div className="space-y-2">
          <Label className="text-pink-800 text-[16px]">Skin Type</Label>
          <Select
            value={formData.skinType}
            onChange={(e) => handleChange("skinType", e.target.value)}
            placeholder="Select Skin Type"
          >
            <SelectItem value="oily">Oily</SelectItem>
            <SelectItem value="dry">Dry</SelectItem>
            <SelectItem value="combination">Combination</SelectItem>
            <SelectItem value="sensitive">Sensitive</SelectItem>
            <SelectItem value="normal">Normal</SelectItem>
          </Select>
        </div>

        {/* Main Concern */}
        <div className="space-y-2">
          <Label className="text-pink-800 text-[16px]">Main Concern</Label>
          <Select
            value={formData.concern}
            onChange={(e) => handleChange("concern", e.target.value)}
            placeholder="Select Concern"
          >
            <SelectItem value="acne">Acne</SelectItem>
            <SelectItem value="wrinkles">Wrinkles</SelectItem>
            <SelectItem value="hyperpigmentation">Hyperpigmentation</SelectItem>
            <SelectItem value="dryness">Dryness</SelectItem>
            <SelectItem value="sensitivity">Sensitivity</SelectItem>
          </Select>
        </div>

        {/* Product Type */}
        <div className="space-y-2">
          <Label className="text-pink-800 text-[16px]">Product Type</Label>
          <Select
            value={formData.productType}
            onChange={(e) => handleChange("productType", e.target.value)}
            placeholder="Select Product Type"
          >
            <SelectItem value="moisturizer">Moisturizer</SelectItem>
            <SelectItem value="serum">Serum</SelectItem>
            <SelectItem value="cleanser">Face Wash</SelectItem>
            <SelectItem value="sunscreen">Sunscreen</SelectItem>
          </Select>
        </div>
      </div>

      {/* Submit Button */}
      <Button type="submit" className="bg-pink-600 hover:bg-pink-700 w-full">
        Get Recommendations
      </Button>
    </form>
  );
}
