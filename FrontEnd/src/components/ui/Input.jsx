import React from "react";
import { cn } from "../../lib/utils";

export function Input({ className, ...props }) {
  return (
    <input
      className={cn(
        "px-3 py-2 border border-pink-300 rounded focus:outline-none focus:ring focus:ring-pink-300",
        className
      )}
      {...props}
    />
  );
}
