"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  FileText,
  ChevronDown,
  ChevronRight,
  Loader2,
  Upload,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
import { pdfAPI } from "@/lib/api";
import type { BookMaterial } from "./chat-books-sidebar";

interface UploadedPDF {
  id: string;
  filename: string;
  title?: string;
  size: number;
  processed: boolean;
  chunk_count?: number;
  page_count?: number;
  subject?: string;
  upload_date: string;
}

interface SubjectGroup {
  name: string;
  documents: UploadedPDF[];
}

interface UploadedPdfsPanelProps {
  selectedMaterialId?: string;
  onSelectMaterial: (material: BookMaterial, sourceName: string, subjectName: string) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function UploadedPdfsPanel({
  selectedMaterialId,
  onSelectMaterial,
}: UploadedPdfsPanelProps) {
  const { toast } = useToast();
  const [subjectGroups, setSubjectGroups] = useState<SubjectGroup[]>([]);
  const [others, setOthers] = useState<SubjectGroup | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);

  const fetchUploads = useCallback(async () => {
    setLoading(true);
    try {
      const data = await pdfAPI.listBySubject();
      setSubjectGroups(data.subjects || []);
      setOthers(data.others || null);
      // Auto-expand subjects that have a selected doc
      const autoExpand = new Set<string>();
      (data.subjects || []).forEach((g) => {
        if (g.documents.some((d) => d.id === selectedMaterialId)) {
          autoExpand.add(g.name);
        }
      });
      if ((data.others?.documents || []).some((d) => d.id === selectedMaterialId)) {
        autoExpand.add("Others");
      }
      setExpanded((prev) => new Set([...prev, ...autoExpand]));
    } catch (e) {
      console.error("Failed to load uploaded PDFs:", e);
      toast({
        title: "Couldn't load uploads",
        description: getErrorMessage(e),
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [selectedMaterialId, toast]);

  useEffect(() => {
    fetchUploads();
  }, [fetchUploads]);

  const toggleGroup = (name: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const handleSelect = (pdf: UploadedPDF, groupName: string) => {
    const material: BookMaterial = {
      id: pdf.id,
      collectionId: "uploads",
      name: pdf.title || pdf.filename,
      type: "pdf",
      size: pdf.size,
      url: "",
      ragIndexed: pdf.processed && (pdf.chunk_count || 0) > 0,
      docId: pdf.id,
    };
    onSelectMaterial(material, groupName, groupName);
  };

  const renderGroup = (group: SubjectGroup) => {
    const isExpanded = expanded.has(group.name);
    const hasDocs = group.documents.length > 0;
    if (!hasDocs) return null;

    return (
      <div key={group.name} className="mb-0.5">
        <button
          onClick={() => toggleGroup(group.name)}
          className={cn(
            "w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md text-[13px] font-medium text-left",
            "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
          )}
        >
          {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          <span className="truncate">{group.name}</span>
          <span className="ml-auto text-[10px] text-muted-foreground/70 tabular-nums">
            {group.documents.length}
          </span>
        </button>

        {isExpanded && (
          <div className="ml-4 mt-0.5 space-y-0.5">
            {group.documents.map((pdf) => (
              <button
                key={pdf.id}
                onClick={() => handleSelect(pdf, group.name)}
                className={cn(
                  "w-full flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-left",
                  selectedMaterialId === pdf.id
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground/70 hover:bg-muted/30 hover:text-foreground"
                )}
                title={`${pdf.filename} • ${formatSize(pdf.size)}`}
              >
                <FileText className="h-3 w-3 shrink-0" />
                <span className="truncate flex-1">{pdf.title || pdf.filename}</span>
                {!pdf.processed || (pdf.chunk_count || 0) === 0 ? (
                  <span className="ml-auto text-[9px] text-amber-500">idx…</span>
                ) : null}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  const totalDocs =
    subjectGroups.reduce((acc, g) => acc + g.documents.length, 0) +
    (others?.documents.length || 0);

  return (
    <div className="h-full flex flex-col bg-background">
      <div className="px-3 py-2 border-b">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">My Uploads</div>
        <h3 className="font-medium text-[13px] truncate">Uploaded PDFs</h3>
      </div>

      <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
        {loading && (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        )}

        {!loading && totalDocs === 0 && (
          <div className="px-2 py-3 text-[11px] text-muted-foreground text-center space-y-2">
            <p>No uploaded PDFs yet.</p>
            <p className="text-[10px]">
              Upload study material from the dashboard; it will appear here subject-wise (or under Others if no subject is set).
            </p>
          </div>
        )}

        {!loading && subjectGroups.map(renderGroup)}
        {!loading && others && renderGroup(others)}
      </div>

      <div className="border-t p-2 text-[10px] text-muted-foreground/80 text-center">
        <Upload className="h-3 w-3 inline-block mr-1" />
        Upload from dashboard to categorize by subject.
      </div>
    </div>
  );
}
