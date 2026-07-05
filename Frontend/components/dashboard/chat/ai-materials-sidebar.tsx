"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Sparkles,
  FileText,
  Layers,
  ClipboardList,
  Trash2,
  Loader2,
  PanelRightClose,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { getErrorMessage } from "@/lib/errors";
import { useRouter } from "next/navigation";
import { aiMaterialAPI, flashcardAPI } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface AIMaterialsSidebarProps {
  /** Currently selected source material (the "ground" for AI actions). */
  selectedMaterial: { id: string; name: string; docId?: string } | null;
  selectedSubjectName?: string;
  onClose?: () => void;
}

interface AIMaterial {
  id: string;
  kind: string;
  title: string;
  subject?: string;
  content: string;
  created_at?: string;
}

export function AIMaterialsSidebar({ selectedMaterial, selectedSubjectName, onClose }: AIMaterialsSidebarProps) {
  const { toast } = useToast();
  const router = useRouter();
  const [materials, setMaterials] = useState<AIMaterial[]>([]);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [viewing, setViewing] = useState<AIMaterial | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const list = await aiMaterialAPI.list();
      setMaterials((list || []) as AIMaterial[]);
    } catch (e) {
      console.error("Failed to load AI materials:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const requireMaterial = () => {
    if (!selectedMaterial) {
      toast({ title: "Select a material first", description: "Pick a book/section material to ground AI generation.", variant: "destructive" });
      return false;
    }
    return true;
  };

  const handleSummarize = async () => {
    if (!requireMaterial() || !selectedMaterial) return;
    setBusy("summary");
    try {
      const mat = await aiMaterialAPI.summarize({
        material_ids: [selectedMaterial.id],
        subject: selectedSubjectName,
        title: `Summary — ${selectedMaterial.name}`,
        style: "detailed",
      });
      toast({ title: "Summary ready", description: "Saved to your AI materials." });
      setViewing({ id: mat.id, kind: "summary", title: mat.title, subject: mat.subject, content: mat.content, created_at: mat.created_at });
      refresh();
    } catch (e) {
      toast({ title: "Summary failed", description: getErrorMessage(e), variant: "destructive" });
    } finally {
      setBusy(null);
    }
  };

  const handleFlashcards = async () => {
    if (!requireMaterial() || !selectedMaterial) return;
    setBusy("flashcards");
    try {
      const res = await flashcardAPI.generate({
        material_ids: [selectedMaterial.id],
        subject: selectedSubjectName,
        title: `Deck — ${selectedMaterial.name}`,
        num_cards: 15,
      });
      toast({ title: "Flashcards ready", description: `${res.card_count} cards generated.` });
      router.push(`/flashcards?deck=${res.deck_id}`);
    } catch (e) {
      toast({ title: "Flashcards failed", description: getErrorMessage(e), variant: "destructive" });
    } finally {
      setBusy(null);
    }
  };

  const handleMockTest = async () => {
    if (!requireMaterial() || !selectedMaterial) return;
    // Route to the mock-test generator, preloading context via query params.
    const params = new URLSearchParams({
      material: selectedMaterial.id,
      subject: selectedSubjectName || "",
    });
    router.push(`/mock-tests?${params.toString()}`);
  };

  const handleDelete = async (id: string) => {
    try {
      await aiMaterialAPI.delete(id);
      setMaterials((prev) => prev.filter((m) => m.id !== id));
    } catch (e) {
      toast({ title: "Delete failed", description: getErrorMessage(e), variant: "destructive" });
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      <div className="px-3 py-2 border-b flex items-center gap-1.5">
        <Sparkles className="h-3.5 w-3.5 text-primary" />
        <span className="text-[13px] font-medium">AI Materials</span>
        {onClose && (
          <Button variant="ghost" size="icon" className="ml-auto h-6 w-6" onClick={onClose}>
            <PanelRightClose className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      {/* AI actions grounded on the selected material */}
      <div className="px-2 py-2 border-b space-y-1.5">
        <div className="text-[10px] text-muted-foreground">
          {selectedMaterial ? <>From <span className="font-medium text-foreground">{selectedMaterial.name}</span></> : "Select a material to enable"}
        </div>
        <div className="grid grid-cols-3 gap-1.5">
          <ActionBtn icon={<FileText className="h-3.5 w-3.5" />} label="Summary" busy={busy === "summary"} disabled={!selectedMaterial || !!busy} onClick={handleSummarize} />
          <ActionBtn icon={<Layers className="h-3.5 w-3.5" />} label="Flashcards" busy={busy === "flashcards"} disabled={!selectedMaterial || !!busy} onClick={handleFlashcards} />
          <ActionBtn icon={<ClipboardList className="h-3.5 w-3.5" />} label="Mock Test" busy={busy === "mocktest"} disabled={!selectedMaterial || !!busy} onClick={handleMockTest} />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-1.5 space-y-1">
        {loading && <div className="flex justify-center py-4"><Loader2 className="h-4 w-4 animate-spin text-muted-foreground" /></div>}
        {!loading && materials.length === 0 && (
          <div className="px-2 py-3 text-[11px] text-muted-foreground text-center">
            No AI-generated material yet. Generate a summary from a selected material.
          </div>
        )}
        {materials.map((m) => (
          <button
            key={m.id}
            onClick={() => setViewing(m)}
            className="w-full text-left rounded-md border p-2 hover:bg-muted/40 transition-colors group"
          >
            <div className="flex items-center gap-1.5">
              <Sparkles className="h-3 w-3 text-primary/70 shrink-0" />
              <span className="text-[12px] font-medium truncate flex-1">{m.title}</span>
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => { e.stopPropagation(); handleDelete(m.id); }}
                onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); handleDelete(m.id); } }}
                className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="h-3 w-3" />
              </span>
            </div>
            {m.subject && <div className="text-[10px] text-muted-foreground mt-0.5">{m.subject}</div>}
          </button>
        ))}
      </div>

      <Dialog open={!!viewing} onOpenChange={(o) => !o && setViewing(null)}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2"><Sparkles className="h-4 w-4 text-primary" /> {viewing?.title}</DialogTitle>
          </DialogHeader>
          <div className="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap text-sm leading-relaxed">
            {viewing?.content}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function ActionBtn({ icon, label, busy, disabled, onClick }: { icon: React.ReactNode; label: string; busy?: boolean; disabled?: boolean; onClick: () => void }) {
  return (
    <Button
      variant="outline"
      size="sm"
      disabled={disabled}
      onClick={onClick}
      className={cn("h-auto flex-col py-1.5 px-1 text-[10px] gap-1")}
    >
      {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : icon}
      <span>{label}</span>
    </Button>
  );
}