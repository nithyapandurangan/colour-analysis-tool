import React from 'react';

const Card = ({ children, className }) => {
  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      {children}
    </div>
  );
};

const CardHeader = ({ children }) => {
  return <div className="mb-4">{children}</div>;
};

const CardTitle = ({ children, className }) => {
  return (
    <h3 className={`text-3xl font-semibold ${className}`}>
      {children}
    </h3>
  );
};

const CardContent = ({ children }) => {
  return <div className="text-sm text-gray-600">{children}</div>;
};

export { Card, CardHeader, CardTitle, CardContent };
