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
      <div className="h-full flex flex-col items-center justify-center p-3 text-center">
        <BookOpen className="h-8 w-8 text-muted-foreground/50 mb-2" />
        <p className="text-xs text-muted-foreground">
          No active exam. Add one from the dashboard to start chatting.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Exam Header */}
      <div className="px-3 py-2 border-b">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-0.5">
          <MessageSquare className="h-3 w-3" />
          Chat Context
        </div>
        <h3 className="font-medium text-[13px] truncate">{exam.name}</h3>
      </div>

      {/* Subjects */}
      <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
        {exam.subjects?.map((subject) => (
          <div key={subject.id}>
            {/* Subject Row */}
            <button
              onClick={() => toggleSubject(subject.id)}
              className={cn(
                "w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md text-[13px] font-medium transition-colors text-left",
                "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
              )}
            >
              {expandedSubjects.has(subject.id) ? (
                <ChevronDown className="h-3 w-3 shrink-0" />
              ) : (
                <ChevronRight className="h-3 w-3 shrink-0" />
              )}
              <span className="truncate">{subject.name}</span>
              <span className="ml-auto text-[10px] text-muted-foreground/60">
                {subject.collections?.length || 0}
              </span>
            </button>

            {/* Collections */}
            {expandedSubjects.has(subject.id) && (
              <div className="ml-4 mt-0.5 space-y-0.5">
                {subject.collections?.map((collection) => (
                  <div key={collection.id}>
                    <button
                      onClick={() => toggleCollection(collection.id)}
                      className={cn(
                        "w-full flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium transition-colors text-left",
                        "text-muted-foreground/80 hover:bg-muted/40 hover:text-foreground"
                      )}
                    >
                      {expandedCollections.has(collection.id) ? (
                        <ChevronDown className="h-2.5 w-2.5 shrink-0" />
                      ) : (
                        <ChevronRight className="h-2.5 w-2.5 shrink-0" />
                      )}
                      <FolderOpen className="h-3 w-3 shrink-0" />
                      <span className="truncate">{collection.name}</span>
                      <span className="ml-auto text-[10px] text-muted-foreground/50">
                        {collection.materials?.length || 0}
                      </span>
                    </button>

                    {/* Materials */}
                    {expandedCollections.has(collection.id) && (
                      <div className="ml-4 mt-0.5 space-y-0.5">
                        {collection.materials?.map((material) => (
                          <button
                            key={material.id}
                            onClick={() =>
                              onSelectMaterial(material, collection, subject)
                            }
                            className={cn(
                              "w-full flex items-center gap-1.5 px-2 py-1 rounded-md text-xs transition-colors text-left",
                              selectedMaterial?.id === material.id
                                ? "bg-secondary text-foreground"
                                : "text-muted-foreground/70 hover:bg-muted/30 hover:text-foreground"
                            )}
                          >
                            <FileText className="h-3 w-3 shrink-0" />
                            <span className="truncate">{material.name}</span>
                          </button>
                        ))}
                        {collection.materials?.length === 0 && (
                          <div className="px-2 py-0.5 text-[10px] text-muted-foreground/40 italic">
                            No materials yet
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
                {subject.collections?.length === 0 && (
                  <div className="px-2 py-0.5 text-[10px] text-muted-foreground/40 italic">
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
