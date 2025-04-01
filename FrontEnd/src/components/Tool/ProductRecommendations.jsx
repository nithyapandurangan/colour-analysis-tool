import React, { useState, useEffect } from 'react';

const ProductRecommendations = ({ seasonTag }) => {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/get_recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ season: seasonTag }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }

      setProducts(data.recommended_products || []);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError(error.message || 'An error occurred while fetching recommendations');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (seasonTag) fetchRecommendations();
  }, [seasonTag]);

  return (
    <div className="mt-8 flex flex-col items-center">
      <h2 className="text-2xl font-bold mb-4 text-center">Recommended Products for {seasonTag}</h2>
      {loading && <p>Loading recommendations...</p>}
      {error && <p className="text-red-500">{error}</p>}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 w-full max-w-5xl justify-items-center">
        {products.length > 0 ? (
          products.map((product, index) => (
            <div key={index} className="border p-4 rounded-lg shadow-lg w-60 flex flex-col items-center">
              <h3 className="font-bold text-lg text-center">{product.title}</h3>
              {product.image_url && (
                <img src={product.image_url} alt={product.title} className="w-full h-40 object-cover mb-4 rounded-lg" />
              )}
              <a href={product.product_url} target="_blank" rel="noopener noreferrer" className="text-pink-600 hover:underline">
                View Product
              </a>
            </div>
          ))
        ) : (
          <p>No products found for this season.</p>
        )}
      </div>
    </div>
  );
};

export default ProductRecommendations;
