"use client";

import { MessageSquare } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Exam } from "@/lib/context/dashboard-context";
import { SubjectAccordion } from "./subject-accordion";

interface CollectionsPanelProps {
  exam: Exam | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChat: (examId: string) => void;
}

export function CollectionsPanel({
  exam,
  open,
  onOpenChange,
  onChat,
}: CollectionsPanelProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full sm:w-[480px] sm:max-w-[480px] bg-background border-l p-0 flex flex-col"
      >
        {/* Sticky Header */}
        <div className="sticky top-0 z-10 bg-background border-b px-6 py-4 flex items-center justify-between">
          <SheetHeader className="space-y-0 text-left">
            <SheetTitle className="text-base">
              {exam?.name || "Exam Details"}
            </SheetTitle>
          </SheetHeader>
        </div>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {exam?.subjects?.map((subject) => (
            <SubjectAccordion key={subject.id} subject={subject} />
          ))}
        </div>

        {/* Sticky Footer */}
        <div className="sticky bottom-0 z-10 bg-background border-t px-6 py-4">
          <Button
            className="w-full gap-2"
            onClick={() => exam && onChat(exam.id)}
            disabled={!exam}
          >
            <MessageSquare className="h-4 w-4" />
            Chat with this Exam
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}
