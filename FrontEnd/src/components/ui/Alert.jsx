import React from "react";
import { cn } from "../../lib/utils";

export function Alert({ className, children, ...props }) {
  return (
    <div
      className={cn(
        "flex items-start p-4 space-x-4 border border-yellow-300 bg-yellow-50 rounded-md",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function AlertDescription({ className, children, ...props }) {
  return (
    <div
      className={cn("text-sm text-yellow-700", className)}
      {...props}
    >
      {children}
    </div>
  );
}
