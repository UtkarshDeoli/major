"use client";

import { BookOpen } from "lucide-react";
import { MagicCard } from "@/components/ui/magic-card";
import Container from "@/components/global/container";
import { Subject } from "@/lib/context/dashboard-context";
import { formatRelativeTime } from "@/lib/utils";

interface SubjectCardProps {
  subject: Subject;
  index: number;
  onClick: () => void;
}

export function SubjectCard({ subject, index, onClick }: SubjectCardProps) {
  const collectionCount = subject.collections?.length || 0;

  return (
    <Container delay={0.2 + index * 0.1}>
      <MagicCard
        className="cursor-pointer hover:-translate-y-1 hover:shadow-lg transition-all duration-300"
        onClick={onClick}
      >
        <div className="p-4 flex flex-col gap-3">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
              <BookOpen className="h-5 w-5 text-primary" />
            </div>
            <div className="flex flex-col">
              <span className="font-medium text-sm">{subject.name}</span>
              <span className="text-xs text-muted-foreground">
                {collectionCount} {collectionCount === 1 ? "collection" : "collections"}
              </span>
            </div>
          </div>
          {subject.lastStudiedAt && (
            <span className="text-xs text-muted-foreground">
              {formatRelativeTime(subject.lastStudiedAt)}
            </span>
          )}
          <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-1000 ease-out"
              style={{ width: `${subject.progress || 0}%` }}
            />
          </div>
        </div>
      </MagicCard>
    </Container>
  );
}
