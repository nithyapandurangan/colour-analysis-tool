import React from "react";
import { cn } from "../../lib/utils";

export function RadioGroup({ className, children, ...props }) {
  return (
    <div
      className={cn(
        "flex flex-col space-y-2 md:flex-row md:space-y-0 md:gap-4",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function RadioGroupItem({ id, name, value, checked, onChange }) {
  return (
    <div className="flex items-start gap-3">
      <input
        id={id}
        name={name}
        type="radio"
        value={value}
        checked={checked}
        onChange={onChange}
        className="h-4 w-4 border-pink-500 focus:ring-pink-500 checked:bg-pink-500 cursor-pointer"
      />
      <label
        htmlFor={id}
        className="text-sm font-medium text-pink-700 cursor-pointer"
      >
      </label>
    </div>
  );
}
