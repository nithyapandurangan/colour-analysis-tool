import React, { useState } from "react";

const SkinCare = () => {
  const [option, setOption] = useState(""); 
  const [skinType, setSkinType] = useState("");
  const [concern, setConcern] = useState("");
  const [productType, setProductType] = useState("");
  const [existingProduct, setExistingProduct] = useState("");
  const [recommendations, setRecommendations] = useState([]); 
  const [error, setError] = useState(""); 

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(""); 
  
    try {
      let response;
      if (option === "new") {
        response = await fetch(`http://127.0.0.1:5001/recommend?skin_type=${skinType.toLowerCase()}&concern=${concern.toLowerCase()}&product_type=${productType.toLowerCase()}`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
      } else if (option === "existing") {
        response = await fetch(`http://127.0.0.1:5001/recommend?existing_product=${existingProduct}`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
      }
  
      if (!response.ok) {
        throw new Error(`Failed to fetch recommendations. Status: ${response.status}`);
      }
  
      const data = await response.json();
      console.log("Recommendations:", data);
  
      if (Array.isArray(data) && data.length > 0) {
        setRecommendations(data);
      } else {
        setRecommendations([]);
        setError("No matching products found. Try adjusting your filters.");
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setError("Something went wrong while fetching recommendations.");
    }
  };
  

  return (
    <div className="p-6 bg-white shadow-lg rounded-lg">
      <h2 className="text-2xl font-semibold mb-4">Skin Care Recommendations</h2>

      {/* Select Recommendation Type */}
      <div className="mb-4">
        <label className="block font-medium">How would you like recommendations?</label>
        <div className="mt-2">
          <label className="mr-4">
            <input
              type="radio"
              value="new"
              checked={option === "new"}
              onChange={() => setOption("new")}
              className="mr-2"
            />
            Start from Scratch
          </label>
          <label>
            <input
              type="radio"
              value="existing"
              checked={option === "existing"}
              onChange={() => setOption("existing")}
              className="mr-2"
            />
            Based on Current Product
          </label>
        </div>
      </div>

      {/* Form Based on Selection */}
      <form onSubmit={handleSubmit} className="space-y-4">
        {option === "new" && (
          <>
            <div>
              <label className="block font-medium">Skin Type</label>
              <select
                value={skinType}
                onChange={(e) => setSkinType(e.target.value)}
                className="w-full border p-2 rounded"
                required
              >
                <option value="">Select Skin Type</option>
                <option value="oily">Oily</option>
                <option value="dry">Dry</option>
                <option value="combination">Combination</option>
                <option value="sensitive">Sensitive</option>
                <option value="normal">Normal</option>
              </select>
            </div>

            <div>
              <label className="block font-medium">Main Concern</label>
              <select
                value={concern}
                onChange={(e) => setConcern(e.target.value)}
                className="w-full border p-2 rounded"
                required
              >
                <option value="">Select Concern</option>
                <option value="acne">Acne</option>
                <option value="wrinkles">Wrinkles</option>
                <option value="hyperpigmentation">Hyperpigmentation</option>
                <option value="dryness">Dryness</option>
                <option value="sensitivity">Sensitivity</option>
              </select>
            </div>

            <div>
              <label className="block font-medium">Product Type</label>
              <select
                value={productType}
                onChange={(e) => setProductType(e.target.value)}
                className="w-full border p-2 rounded"
                required
              >
                <option value="">Select Product Type</option>
                <option value="moisturizer">Moisturizer</option>
                <option value="serum">Serum</option>
                <option value="cleanser">Face Wash</option>
                <option value="sunscreen">Sunscreen</option>
              </select>
            </div>
          </>
        )}

        {option === "existing" && (
          <div>
            <label className="block font-medium">Current Product Name</label>
            <input
              type="text"
              value={existingProduct}
              onChange={(e) => setExistingProduct(e.target.value)}
              className="w-full border p-2 rounded"
              placeholder="Enter product name"
              required
            />
          </div>
        )}

        {/* Submit Button */}
        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
          Get Recommendations
        </button>
      </form>

      {/* Display API Results */}
      {error && <p className="text-red-600 mt-4">{error}</p>}

      {recommendations.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-4">Recommended Products:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {recommendations.map((product, index) => (
              <div key={index} className="border p-4 rounded-lg shadow-md">
                <img
                  src={product.picture_src}
                  alt={product.product_name}
                  className="w-full h-40 object-cover rounded"
                />
                <h4 className="text-lg font-medium mt-2">{product.product_name}</h4>
                <p className="text-gray-700"><strong>Brand:</strong> {product.brand}</p>
                <p className="text-gray-700"><strong>Type:</strong> {product.product_type}</p>
                <p className="text-gray-700"><strong>Skin Type:</strong> {product.skintype}</p>
                <p className="text-gray-700"><strong>Effect:</strong> {product.notable_effects}</p>
                <p className="text-gray-900 font-semibold"><strong>Price:</strong> {product.price}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SkinCare;
