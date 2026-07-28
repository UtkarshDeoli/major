"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  BookOpen,
  FolderOpen,
  FileText,
  ChevronDown,
  ChevronRight,
  Plus,
  Upload,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
import {
  subjectAPI,
  collectionAPI,
  materialAPI,
  sampleMaterialAPI,
} from "@/lib/api";
import { UploadedPdfsPanel } from "./uploaded-pdfs-panel";

export interface BookMaterial {
  id: string;
  collectionId: string;
  name: string;
  type: string;
  size: number;
  url: string;
  ragIndexed: boolean;
  docId?: string;
}

interface ChatBooksSidebarProps {
  examId: string | undefined;
  examName?: string;
  selectedMaterialId?: string;
  onSelectMaterial: (material: BookMaterial, collectionName: string, subjectName: string) => void;
  /** When true (student has no teacher), show the sample-material hint. */
  showSampleHint?: boolean;
}

interface SubjectNode {
  id: string;
  name: string;
  collections?: CollectionNode[];
}
interface CollectionNode {
  id: string;
  name: string;
  materials?: BookMaterial[];
}

export function ChatBooksSidebar({
  examId,
  examName,
  selectedMaterialId,
  onSelectMaterial,
  showSampleHint,
}: ChatBooksSidebarProps) {
  const { toast } = useToast();
  const [subjects, setSubjects] = useState<SubjectNode[]>([]);
  const [expandedSubjects, setExpandedSubjects] = useState<Set<string>>(new Set());
  const [expandedCollections, setExpandedCollections] = useState<Set<string>>(new Set());
  const [loadingSubjects, setLoadingSubjects] = useState(false);
  const [loadingCol, setLoadingCol] = useState<string | null>(null);
  const [loadingMat, setLoadingMat] = useState<string | null>(null);
  const [newSectionSubject, setNewSectionSubject] = useState<string | null>(null);
  const [newSectionName, setNewSectionName] = useState("");
  const [seeding, setSeeding] = useState(false);
  const [activeTab, setActiveTab] = useState<"books" | "uploads">("books");
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadTargetCollection = useRef<string | null>(null);

  const handleSeedSample = async () => {
    setSeeding(true);
    try {
      const res = await sampleMaterialAPI.seed();
      toast({ title: "Sample material loaded", description: `Seeded ${res?.count ?? 0} NCERT/PYQ materials. Expand a subject to view.` });
      await fetchSubjects();
    } catch (e) {
      toast({ title: "Couldn't load sample material", description: getErrorMessage(e), variant: "destructive" });
    } finally {
      setSeeding(false);
    }
  };

  const fetchSubjects = useCallback(async () => {
    if (!examId) return;
    setLoadingSubjects(true);
    try {
      const subs = await subjectAPI.listSubjects(examId);
      setSubjects((subs || []).map((s: any) => ({ id: s.id, name: s.name })));
      if ((subs || []).length === 1) setExpandedSubjects(new Set([subs[0].id]));
    } catch (e) {
      console.error("Failed to load subjects:", e);
    } finally {
      setLoadingSubjects(false);
    }
  }, [examId]);

  useEffect(() => {
    fetchSubjects();
  }, [fetchSubjects]);

  const toggleSubject = async (id: string) => {
    setExpandedSubjects((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    // Lazy-load collections once
    const subj = subjects.find((s) => s.id === id);
    if (subj && !subj.collections) {
      setLoadingCol(id);
      try {
        const cols = await collectionAPI.listCollections(id);
        setSubjects((prev) => prev.map((s) => s.id === id ? { ...s, collections: (cols || []).map((c: any) => ({ id: c.id, name: c.name })) } : s));
      } catch (e) {
        console.error("Failed to load collections:", e);
      } finally {
        setLoadingCol(null);
      }
    }
  };

  const toggleCollection = async (subjectName: string, colId: string) => {
    setExpandedCollections((prev) => {
      const next = new Set(prev);
      if (next.has(colId)) next.delete(colId);
      else next.add(colId);
      return next;
    });
    for (const s of subjects) {
      const col = s.collections?.find((c) => c.id === colId);
      if (col && !col.materials) {
        setLoadingMat(colId);
        try {
          const mats = await materialAPI.listMaterials(colId);
          const mapped: BookMaterial[] = (mats || []).map((m: any) => ({
            id: m.id,
            collectionId: m.collection_id,
            name: m.name,
            type: m.type,
            size: m.size,
            url: m.url,
            ragIndexed: m.rag_indexed,
            docId: m.doc_id,
          }));
          setSubjects((prev) => prev.map((ss) => ss.id === s.id ? { ...ss, collections: ss.collections?.map((cc) => cc.id === colId ? { ...cc, materials: mapped } : cc) } : ss));
        } catch (e) {
          console.error("Failed to load materials:", e);
        } finally {
          setLoadingMat(null);
        }
        break;
      }
    }
  };

  const handleCreateSection = async (subjectId: string) => {
    const name = newSectionName.trim();
    if (!name) return;
    try {
      await collectionAPI.createCollection(subjectId, name);
      setNewSectionSubject(null);
      setNewSectionName("");
      // Refresh collections for this subject
      const cols = await collectionAPI.listCollections(subjectId);
      setSubjects((prev) => prev.map((s) => s.id === subjectId ? { ...s, collections: (cols || []).map((c: any) => ({ id: c.id, name: c.name })) } : s));
      toast({ title: "Section added", description: `“${name}” created.` });
    } catch (e) {
      toast({ title: "Couldn't add section", description: getErrorMessage(e), variant: "destructive" });
    }
  };

  const triggerUpload = (colId: string) => {
    uploadTargetCollection.current = colId;
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    const colId = uploadTargetCollection.current;
    if (!file || !colId) {
      e.target.value = "";
      return;
    }
    try {
      await materialAPI.uploadMaterial(colId, file);
      const mats = await materialAPI.listMaterials(colId);
      const mapped: BookMaterial[] = (mats || []).map((m: any) => ({
        id: m.id, collectionId: m.collection_id, name: m.name, type: m.type,
        size: m.size, url: m.url, ragIndexed: m.rag_indexed, docId: m.doc_id,
      }));
      setSubjects((prev) => prev.map((s) => s.collections ? { ...s, collections: s.collections.map((cc) => cc.id === colId ? { ...cc, materials: mapped } : cc) } : s));
      toast({ title: "Material uploaded", description: "Indexed and ready to chat." });
    } catch (err) {
      toast({ title: "Upload failed", description: getErrorMessage(err), variant: "destructive" });
    } finally {
      e.target.value = "";
      uploadTargetCollection.current = null;
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange} accept=".pdf,.txt,.md" />

      <div className="px-2 py-2 border-b space-y-2">
        <div className="flex items-center gap-1 bg-muted/50 rounded-md p-0.5">
          <button
            onClick={() => setActiveTab("books")}
            className={cn(
              "flex-1 text-[11px] font-medium px-2 py-1 rounded-[4px] transition-colors",
              activeTab === "books"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            Books
          </button>
          <button
            onClick={() => setActiveTab("uploads")}
            className={cn(
              "flex-1 text-[11px] font-medium px-2 py-1 rounded-[4px] transition-colors",
              activeTab === "uploads"
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            My Uploads
          </button>
        </div>
        {activeTab === "books" && (
          <div>
            <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-0.5">Your Books</div>
            <h3 className="font-medium text-[13px] truncate">{examName || "Exam"}</h3>
          </div>
        )}
      </div>

      {activeTab === "uploads" && (
        <UploadedPdfsPanel
          selectedMaterialId={selectedMaterialId}
          onSelectMaterial={onSelectMaterial}
        />
      )}

      {activeTab === "books" && (
        <>
          <div className="flex-1 overflow-y-auto p-1.5 space-y-0.5">
        {!examId && (
          <div className="h-full flex flex-col items-center justify-center p-3 text-center">
            <BookOpen className="h-8 w-8 text-muted-foreground/50 mb-2" />
            <p className="text-xs text-muted-foreground">
              No active exam. Add one from the dashboard to start chatting.
            </p>
          </div>
        )}
        {examId && loadingSubjects && (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        )}
        {examId && !loadingSubjects && subjects.length === 0 && (
          <div className="px-2 py-2 text-[11px] text-muted-foreground">
            No subjects yet. Add subjects from the dashboard.
          </div>
        )}

        {examId && subjects.map((subject) => (
          <div key={subject.id}>
            <div className="flex items-center">
              <button
                onClick={() => toggleSubject(subject.id)}
                className={cn(
                  "flex-1 flex items-center gap-1.5 px-2 py-1.5 rounded-md text-[13px] font-medium text-left",
                  "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                )}
              >
                {expandedSubjects.has(subject.id) ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                <span className="truncate">{subject.name}</span>
              </button>
              <Button
                variant="ghost" size="icon" className="h-5 w-5"
                title="Add section"
                onClick={() => setNewSectionSubject(newSectionSubject === subject.id ? null : subject.id)}
              >
                <Plus className="h-3 w-3" />
              </Button>
            </div>

            {newSectionSubject === subject.id && (
              <div className="ml-4 mt-1 mb-1 flex gap-1">
                <input
                  autoFocus
                  value={newSectionName}
                  onChange={(e) => setNewSectionName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") handleCreateSection(subject.id); if (e.key === "Escape") setNewSectionSubject(null); }}
                  placeholder="Section/chapter name"
                  className="flex-1 rounded-md border bg-background px-2 py-1 text-[11px]"
                />
                <Button size="sm" className="h-6 text-[11px]" onClick={() => handleCreateSection(subject.id)}>Add</Button>
              </div>
            )}

            {expandedSubjects.has(subject.id) && (
              <div className="ml-4 mt-0.5 space-y-0.5">
                {loadingCol === subject.id && (
                  <div className="px-2 py-1"><Loader2 className="h-3 w-3 animate-spin text-muted-foreground" /></div>
                )}
                {subject.collections?.map((col) => (
                  <div key={col.id}>
                    <div className="flex items-center">
                      <button
                        onClick={() => toggleCollection(subject.name, col.id)}
                        className={cn(
                          "flex-1 flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium text-left",
                          "text-muted-foreground/80 hover:bg-muted/40 hover:text-foreground"
                        )}
                      >
                        {expandedCollections.has(col.id) ? <ChevronDown className="h-2.5 w-2.5" /> : <ChevronRight className="h-2.5 w-2.5" />}
                        <FolderOpen className="h-3 w-3" />
                        <span className="truncate">{col.name}</span>
                      </button>
                      <Button variant="ghost" size="icon" className="h-5 w-5" title="Upload material" onClick={() => triggerUpload(col.id)}>
                        <Upload className="h-3 w-3" />
                      </Button>
                    </div>

                    {expandedCollections.has(col.id) && (
                      <div className="ml-4 mt-0.5 space-y-0.5">
                        {loadingMat === col.id && <div className="px-2 py-1"><Loader2 className="h-3 w-3 animate-spin text-muted-foreground" /></div>}
                        {col.materials?.map((m) => (
                          <button
                            key={m.id}
                            onClick={() => onSelectMaterial(m, col.name, subject.name)}
                            className={cn(
                              "w-full flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-left",
                              selectedMaterialId === m.id ? "bg-secondary text-foreground" : "text-muted-foreground/70 hover:bg-muted/30 hover:text-foreground"
                            )}
                          >
                            <FileText className="h-3 w-3 shrink-0" />
                            <span className="truncate">{m.name}</span>
                            {!m.ragIndexed && <span className="ml-auto text-[9px] text-amber-500">idx…</span>}
                          </button>
                        ))}
                        {col.materials?.length === 0 && (
                          <div className="px-2 py-0.5 text-[10px] text-muted-foreground/40 italic">No materials — upload one</div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
                {subject.collections?.length === 0 && (
                  <div className="px-2 py-0.5 text-[10px] text-muted-foreground/40 italic">No sections — add one with +</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {showSampleHint && (
        <div className="border-t p-2 space-y-2">
          <p className="text-[10px] text-muted-foreground/80">
            Not enrolled with a teacher? Load free NCERT excerpts & JEE/NEET PYQs to start chatting, or join a class from the dashboard.
          </p>
          <Button
            size="sm"
            variant="outline"
            disabled={seeding}
            onClick={handleSeedSample}
            className="w-full h-7 text-[11px]"
          >
            {seeding ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <BookOpen className="h-3 w-3 mr-1" />}
            Load sample NCERT & PYQ
          </Button>
        </div>
      )}
        </>
      )}
    </div>
  );
}