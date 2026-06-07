"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
} from "react";

// ─── Types ──────────────────────────────────────────────────────────────────

export interface Material {
  id: string;
  collectionId: string;
  name: string;
  type: string;
  size: number;
  url: string;
  uploadedAt: string;
  ragIndexed: boolean;
}

export interface Collection {
  id: string;
  subjectId: string;
  name: string;
  description?: string;
  materials: Material[];
  createdAt: string;
}

export interface Subject {
  id: string;
  examId: string;
  name: string;
  icon?: string;
  collections: Collection[];
  progress: number;
  lastStudiedAt?: string;
}

export interface Exam {
  id: string;
  name: string;
  description?: string;
  icon?: string;
  color?: string;
  subjects: Subject[];
  isActive: boolean;
  createdAt: string;
}

// ─── Context Shape ──────────────────────────────────────────────────────────

interface DashboardContextValue {
  exams: Exam[];
  activeExam: Exam | null;
  setExams: React.Dispatch<React.SetStateAction<Exam[]>>;
  setActiveExam: React.Dispatch<React.SetStateAction<Exam | null>>;
  refreshExams: () => Promise<void>;
  isLoading: boolean;
}

const DashboardContext = createContext<DashboardContextValue | undefined>(
  undefined
);

// ─── Provider ─────────────────────────────────────────────────────────────────

export function DashboardProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [exams, setExams] = useState<Exam[]>([]);
  const [activeExam, setActiveExam] = useState<Exam | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const refreshExams = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/exams");
      if (!res.ok) {
        throw new Error(`Failed to fetch exams: ${res.status}`);
      }
      const data: Exam[] = await res.json();
      setExams(data);
      setActiveExam((prev) => {
        if (prev) {
          const stillActive = data.find((e) => e.id === prev.id);
          return stillActive ?? (data.find((e) => e.isActive) || data[0] || null);
        }
        return data.find((e) => e.isActive) || data[0] || null;
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <DashboardContext.Provider
      value={{
        exams,
        activeExam,
        setExams,
        setActiveExam,
        refreshExams,
        isLoading,
      }}
    >
      {children}
    </DashboardContext.Provider>
  );
}

// ─── Hook ───────────────────────────────────────────────────────────────────

export function useDashboard(): DashboardContextValue {
  const ctx = useContext(DashboardContext);
  if (ctx === undefined) {
    throw new Error("useDashboard must be used within a DashboardProvider");
  }
  return ctx;
}
