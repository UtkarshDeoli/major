"use client";

import { useState } from "react";
import { FolderOpen } from "lucide-react";
import { Collection } from "@/lib/context/dashboard-context";
import { MaterialList } from "./material-list";

interface CollectionItemProps {
  collection: Collection;
}

export function CollectionItem({ collection }: CollectionItemProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border rounded-md overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-3 w-full p-3 text-left hover:bg-muted/50 transition-colors"
      >
        <FolderOpen className="h-4 w-4 text-primary shrink-0" />
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium truncate">{collection.name}</span>
        </div>
        <span className="text-xs text-muted-foreground shrink-0">
          {collection.materials?.length || 0} materials
        </span>
      </button>
      {expanded && (
        <div className="px-3 pb-3">
          <MaterialList
            materials={collection.materials || []}
            collectionId={collection.id}
          />
        </div>
      )}
    </div>
  );
}
