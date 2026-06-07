"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Subject } from "@/lib/context/dashboard-context";
import { CollectionItem } from "./collection-item";

interface SubjectAccordionProps {
  subject: Subject;
}

export function SubjectAccordion({ subject }: SubjectAccordionProps) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger className="flex items-center justify-between w-full py-3 text-left hover:text-primary transition-colors">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{subject.name}</span>
          <span className="text-xs text-muted-foreground">
            {subject.collections?.length || 0} collections
          </span>
        </div>
        {isOpen ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        )}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-2 pb-2">
        {subject.collections?.map((collection) => (
          <CollectionItem key={collection.id} collection={collection} />
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}
