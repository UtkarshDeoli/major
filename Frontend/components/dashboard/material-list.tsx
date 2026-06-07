"use client";

import { useRef } from "react";
import { FileText, Trash2, Upload } from "lucide-react";
import { Material } from "@/lib/context/dashboard-context";
import { formatFileSize } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface MaterialListProps {
  materials: Material[];
  collectionId: string;
}

export function MaterialList({ materials, collectionId }: MaterialListProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      // Placeholder for upload logic
      console.log("Uploading files for collection", collectionId, files);
    }
  };

  return (
    <div className="space-y-2">
      {materials.map((material) => (
        <div
          key={material.id}
          className="group flex items-center gap-3 p-2 rounded-md hover:bg-muted/50 transition-colors"
        >
          <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
          <div className="flex-1 min-w-0">
            <span className="text-sm truncate block">{material.name}</span>
          </div>
          <span className="text-xs text-muted-foreground shrink-0">
            {formatFileSize(material.size)}
          </span>
          <button
            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:text-destructive"
            aria-label="Delete material"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      ))}

      <Button
        variant="outline"
        className="w-full justify-center gap-2 border-dashed"
        onClick={handleUpload}
      >
        <Upload className="h-4 w-4" />
        Upload Material
      </Button>

      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        onChange={handleFileChange}
        multiple
      />
    </div>
  );
}
