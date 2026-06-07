"use client";

import React, { useState } from "react";
import {
  BookOpen,
  FolderOpen,
  FileText,
  ChevronDown,
  ChevronRight,
  MessageSquare,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Exam,
  Subject,
  Collection,
  Material,
} from "@/lib/context/dashboard-context";

interface CollectionsChatSidebarProps {
  exam: Exam | null;
  selectedMaterial: Material | null;
  onSelectMaterial: (material: Material, collection: Collection, subject: Subject) => void;
}

export function CollectionsChatSidebar({
  exam,
  selectedMaterial,
  onSelectMaterial,
}: CollectionsChatSidebarProps) {
  const [expandedSubjects, setExpandedSubjects] = useState<Set<string>>(() => {
    // Auto-expand first subject if only one
    if (exam?.subjects?.length === 1) {
      return new Set([exam.subjects[0].id]);
    }
    return new Set();
  });

  const [expandedCollections, setExpandedCollections] = useState<Set<string>>(
    () => new Set()
  );

  const toggleSubject = (id: string) => {
    setExpandedSubjects((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleCollection = (id: string) => {
    setExpandedCollections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  if (!exam) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-4 text-center">
        <BookOpen className="h-10 w-10 text-muted-foreground/50 mb-3" />
        <p className="text-sm text-muted-foreground">
          No active exam. Add one from the dashboard to start chatting.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Exam Header */}
      <div className="px-4 py-3 border-b">
        <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
          <MessageSquare className="h-3.5 w-3.5" />
          Chat Context
        </div>
        <h3 className="font-semibold text-sm truncate">{exam.name}</h3>
      </div>

      {/* Subjects */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {exam.subjects?.map((subject) => (
          <div key={subject.id}>
            {/* Subject Row */}
            <button
              onClick={() => toggleSubject(subject.id)}
              className={cn(
                "w-full flex items-center gap-2 px-2 py-2 rounded-lg text-sm font-medium transition-colors text-left",
                "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
              )}
            >
              {expandedSubjects.has(subject.id) ? (
                <ChevronDown className="h-3.5 w-3.5 shrink-0" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 shrink-0" />
              )}
              <span className="truncate">{subject.name}</span>
              <span className="ml-auto text-xs text-muted-foreground/60">
                {subject.collections?.length || 0}
              </span>
            </button>

            {/* Collections */}
            {expandedSubjects.has(subject.id) && (
              <div className="ml-5 mt-1 space-y-1">
                {subject.collections?.map((collection) => (
                  <div key={collection.id}>
                    <button
                      onClick={() => toggleCollection(collection.id)}
                      className={cn(
                        "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs font-medium transition-colors text-left",
                        "text-muted-foreground/80 hover:bg-muted/40 hover:text-foreground"
                      )}
                    >
                      {expandedCollections.has(collection.id) ? (
                        <ChevronDown className="h-3 w-3 shrink-0" />
                      ) : (
                        <ChevronRight className="h-3 w-3 shrink-0" />
                      )}
                      <FolderOpen className="h-3.5 w-3.5 shrink-0" />
                      <span className="truncate">{collection.name}</span>
                      <span className="ml-auto text-[10px] text-muted-foreground/50">
                        {collection.materials?.length || 0}
                      </span>
                    </button>

                    {/* Materials */}
                    {expandedCollections.has(collection.id) && (
                      <div className="ml-5 mt-0.5 space-y-0.5">
                        {collection.materials?.map((material) => (
                          <button
                            key={material.id}
                            onClick={() =>
                              onSelectMaterial(material, collection, subject)
                            }
                            className={cn(
                              "w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors text-left",
                              selectedMaterial?.id === material.id
                                ? "bg-primary/10 text-primary"
                                : "text-muted-foreground/70 hover:bg-muted/30 hover:text-foreground"
                            )}
                          >
                            <FileText className="h-3 w-3 shrink-0" />
                            <span className="truncate">{material.name}</span>
                          </button>
                        ))}
                        {collection.materials?.length === 0 && (
                          <div className="px-2 py-1 text-[10px] text-muted-foreground/40 italic">
                            No materials yet
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
                {subject.collections?.length === 0 && (
                  <div className="px-2 py-1 text-xs text-muted-foreground/40 italic">
                    No collections
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
