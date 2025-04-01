import React from "react";
import { cn } from "../../lib/utils";

export function Select({ className, children, ...props }) {
  return (
    <div className={cn("relative w-full", className)}>
      <select
        className="block w-full p-2 border border-pink-300 rounded focus:ring focus:ring-pink-400 focus:border-pink-500 text-black"
        {...props}
      >
        <option value="" disabled>
          {props.placeholder || "Select an option"}
        </option>
        {children}
      </select>
    </div>
  );
}

export function SelectTrigger({ children }) {
  return <div className="w-full p-2 border border-pink-300 rounded bg-pink-50 text-pink-700 focus:ring focus:ring-pink-400 cursor-pointe">{children}</div>;
}

export function SelectContent({ children }) {
  return (
    <div className="absolute text-black mt-1 w-full bg-white border border-gray-300 rounded shadow-lg">
      {children}
    </div>
  );
}

export function SelectItem({ value, children }) {
  return (
    <option
      value={value}
      className="px-4 py-2 text-sm text-black hover:bg-pink-100 cursor-pointer"
    >
      {children}
    </option>
  );
}

export function SelectValue({ value, placeholder }) {
  return (
    <div className="p-2 border border-pink-300 rounded bg-pink-50 text-black">
      {value ? value : <span className="text-pink-400">{placeholder}</span>}
      </div>
  );
}
