// Form Options for - Starting from scratch or Based on Current Product

import { RadioGroup, RadioGroupItem } from "../ui/RadioGroup";
import { Label } from "../ui/Label";
import { Heart } from "lucide-react";

export default function FormOptions({ option, setOption }) {
  return (
    <div className="mb-8 bg-pink-50 p-6 rounded-xl border border-pink-100">
      <h3 className="text-xl font-bold mb-4 text-pink-800 flex items-center">
        <Heart className="h-5 w-5 mr-2 text-pink-500" />
        How would you like recommendations?
      </h3>

      <RadioGroup value={option} onValueChange={setOption} className="flex gap-4">
        {/* Start from Scratch */}
        <div
          className={`flex items-center border rounded-lg p-4 w-full max-w-md cursor-pointer transition-all ${
            option === "new" ? "border-pink-500 bg-pink-100" : "border-pink-200"
          }`}
          onClick={() => setOption("new")}
        >
          <RadioGroupItem
            id="new"
            value="new"
            checked={option === "new"}
            onChange={() => setOption("new")}
            className="mr-3"
          />
          <div className="flex flex-col">
            <Label htmlFor="new" className="text-[18px] font-bold text-pink-800 cursor-pointer">
              Start from Scratch
            </Label>
            <p className="text-pink-600 text-[14px] leading-8">Tell us about your skin and concerns</p>
          </div>
        </div>

        {/* Based on Current Product */}
        <div
          className={`flex items-center border rounded-lg p-4 w-full max-w-md cursor-pointer transition-all ${
            option === "existing" ? "border-pink-500 bg-pink-100" : "border-pink-200"
          }`}
          onClick={() => setOption("existing")}
        >
          <RadioGroupItem
            id="existing"
            value="existing"
            checked={option === "existing"}
            onChange={() => setOption("existing")}
            className="mr-3"
          />
          <div className="flex flex-col">
            <Label htmlFor="existing" className="text-[18px] font-bold text-pink-800 cursor-pointer">
              Based on Current Product
            </Label>
            <p className="text-pink-600 text-[14px] leading-8">Find similar products to what you already love</p>
          </div>
        </div>
      </RadioGroup>
    </div>
  );
}
