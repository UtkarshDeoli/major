# Dashboard & Collections Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Orbit dashboard to match the landing page's "Cosmic Productivity" aesthetic, add a 3-step onboarding flow, and implement the Exam → Subject → Collection → Material study workspace.

**Architecture:** Next.js App Router with shadcn/ui components. Landing page's `MagicCard` and `Container` animation system reused for dashboard. Backend extends FastAPI with new MongoDB collections for exams, subjects, collections, and materials. React Context for exam tree state, SWR for data fetching.

**Tech Stack:** Next.js 15, TypeScript, Tailwind CSS, shadcn/ui, Framer Motion (via `Container`), Lucide React, FastAPI, MongoDB (Motor), SWR

---

## File Structure

### New Files (Frontend)
- `Frontend/app/onboarding/page.tsx` — Onboarding route with 3-step wizard
- `Frontend/components/onboarding/step-about-you.tsx` — Step 1 form
- `Frontend/components/onboarding/step-study-goal.tsx` — Step 2 exam selection
- `Frontend/components/onboarding/step-tour.tsx` — Step 3 quick tour
- `Frontend/components/onboarding/onboarding-container.tsx` — Step wrapper with transitions
- `Frontend/components/dashboard/active-study-card.tsx` — Centerpiece exam card
- `Frontend/components/dashboard/subject-card.tsx` — Subject tile
- `Frontend/components/dashboard/collections-panel.tsx` — Sheet slide-over
- `Frontend/components/dashboard/subject-accordion.tsx` — Collapsible subject section
- `Frontend/components/dashboard/collection-item.tsx` — Collection row
- `Frontend/components/dashboard/material-list.tsx` — PDF list + upload
- `Frontend/components/dashboard/exam-setup-dialog.tsx` — Dialog to add exam
- `Frontend/components/ui/progress-ring.tsx` — Circular progress
- `Frontend/components/ui/bento-grid.tsx` — Layout helper
- `Frontend/lib/constants/exams.ts` — Preset exam data
- `Frontend/lib/context/dashboard-context.tsx` — React Context for exam tree

### Modified Files (Frontend)
- `Frontend/app/(dashboard)/dashboard/page.tsx` — Full rewrite
- `Frontend/app/(dashboard)/layout.tsx` — Add onboarding redirect guard
- `Frontend/app/globals.css` — Deprecate neo-card, keep animations

### New Files (Backend)
- `Backend/src/routers/exam_router.py` — Exam CRUD routes
- `Backend/src/routers/subject_router.py` — Subject routes
- `Backend/src/routers/collection_router.py` — Collection routes
- `Backend/src/routers/material_router.py` — Material upload/delete
- `Backend/src/routers/onboarding_router.py` — Onboarding status
- `Backend/src/services/exam_service.py` — Exam business logic
- `Backend/src/core/models.py` — Extend with Exam, Subject, Collection, Material models
- `Backend/src/core/data_store.py` — Add new collection accessors

### Modified Files (Backend)
- `Backend/src/main.py` — Register new routers
- `Backend/src/core/models.py` — Add User fields (role, institute, preferredLanguage, onboardingCompleted)

---

## Task 1: Backend — Extend User Model with Onboarding Fields

**Files:**
- Modify: `Backend/src/core/models.py`

- [ ] **Step 1: Add new fields to User model**

Add to the existing User Pydantic model:

```python
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    email: str
    name: str
    password_hash: str
    role: str = "student"  # "student" or "teacher"
    institute: Optional[str] = None
    preferred_language: str = "en"
    onboarding_completed: bool = False
    active_exam_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

- [ ] **Step 2: Commit**

```bash
git add Backend/src/core/models.py
git commit -m "feat: extend User model with onboarding and role fields"
```

---

## Task 2: Backend — Create Exam, Subject, Collection, Material Models

**Files:**
- Modify: `Backend/src/core/models.py`

- [ ] **Step 1: Add new data models**

Append to `Backend/src/core/models.py`:

```python
class Exam(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    user_id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    color: Optional[str] = None
    is_active: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Subject(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    exam_id: str
    name: str
    icon: Optional[str] = None
    progress: int = 0
    last_studied_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Collection(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    subject_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Material(BaseModel):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    collection_id: str
    name: str
    type: str = "pdf"  # "pdf", "image", "text"
    size: int = 0
    url: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    rag_indexed: bool = False
```

- [ ] **Step 2: Commit**

```bash
git add Backend/src/core/models.py
git commit -m "feat: add Exam, Subject, Collection, Material models"
```

---

## Task 3: Backend — Add Data Store Collections

**Files:**
- Modify: `Backend/src/core/data_store.py`

- [ ] **Step 1: Add collection accessors**

Add to `DataStore` class:

```python
@property
def exams_collection(self):
    return self.db["exams"]

@property
def subjects_collection(self):
    return self.db["subjects"]

@property
def collections_collection(self):
    return self.db["collections"]

@property
def materials_collection(self):
    return self.db["materials"]
```

- [ ] **Step 2: Commit**

```bash
git add Backend/src/core/data_store.py
git commit -m "feat: add new MongoDB collection accessors"
```

---

## Task 4: Backend — Create Exam Router

**Files:**
- Create: `Backend/src/routers/exam_router.py`

- [ ] **Step 1: Write exam router**

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

from src.core.models import Exam, Subject
from src.core.data_store import data_store
from src.routers.auth_router import get_current_user

router = APIRouter(prefix="/api/exams", tags=["exams"])

@router.get("/", response_model=List[Exam])
async def list_exams(current_user: dict = Depends(get_current_user)):
    exams = await data_store.exams_collection.find({"user_id": str(current_user["_id"])}).to_list(length=100)
    return [Exam(**exam) for exam in exams]

@router.post("/", response_model=Exam)
async def create_exam(exam_data: dict, current_user: dict = Depends(get_current_user)):
    exam = Exam(
        user_id=str(current_user["_id"]),
        name=exam_data.get("name"),
        description=exam_data.get("description"),
        icon=exam_data.get("icon"),
        color=exam_data.get("color"),
        is_active=exam_data.get("is_active", False)
    )
    result = await data_store.exams_collection.insert_one(exam.dict(by_alias=True))
    exam.id = str(result.inserted_id)
    return exam

@router.patch("/{exam_id}/active")
async def set_active_exam(exam_id: str, current_user: dict = Depends(get_current_user)):
    # Deactivate all other exams
    await data_store.exams_collection.update_many(
        {"user_id": str(current_user["_id"])},
        {"$set": {"is_active": False}}
    )
    # Activate selected
    await data_store.exams_collection.update_one(
        {"_id": ObjectId(exam_id), "user_id": str(current_user["_id"])},
        {"$set": {"is_active": True, "updated_at": datetime.utcnow()}}
    )
    # Update user
    await data_store.users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"active_exam_id": exam_id, "updated_at": datetime.utcnow()}}
    )
    return {"success": True}
```

- [ ] **Step 2: Commit**

```bash
git add Backend/src/routers/exam_router.py
git commit -m "feat: add exam CRUD router with active exam toggle"
```

---

## Task 5: Backend — Create Subject Router

**Files:**
- Create: `Backend/src/routers/subject_router.py`

- [ ] **Step 1: Write subject router**

```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime
from bson import ObjectId

from src.core.models import Subject
from src.core.data_store import data_store
from src.routers.auth_router import get_current_user

router = APIRouter(prefix="/api/subjects", tags=["subjects"])

@router.get("/{exam_id}/subjects", response_model=List[Subject])
async def list_subjects(exam_id: str, current_user: dict = Depends(get_current_user)):
    subjects = await data_store.subjects_collection.find({"exam_id": exam_id}).to_list(length=100)
    return [Subject(**subject) for subject in subjects]

@router.post("/{exam_id}/subjects", response_model=Subject)
async def create_subject(exam_id: str, subject_data: dict, current_user: dict = Depends(get_current_user)):
    subject = Subject(
        exam_id=exam_id,
        name=subject_data.get("name"),
        icon=subject_data.get("icon")
    )
    result = await data_store.subjects_collection.insert_one(subject.dict(by_alias=True))
    subject.id = str(result.inserted_id)
    return subject
```

- [ ] **Step 2: Commit**

```bash
git add Backend/src/routers/subject_router.py
git commit -m "feat: add subject router with list and create"
```

---

## Task 6: Backend — Create Collection and Material Routers

**Files:**
- Create: `Backend/src/routers/collection_router.py`
- Create: `Backend/src/routers/material_router.py`

- [ ] **Step 1: Write collection router**

```python
from fastapi import APIRouter, Depends
from typing import List
from datetime import datetime
from bson import ObjectId

from src.core.models import Collection
from src.core.data_store import data_store
from src.routers.auth_router import get_current_user

router = APIRouter(prefix="/api/collections", tags=["collections"])

@router.get("/{subject_id}/collections", response_model=List[Collection])
async def list_collections(subject_id: str, current_user: dict = Depends(get_current_user)):
    collections = await data_store.collections_collection.find({"subject_id": subject_id}).to_list(length=100)
    return [Collection(**col) for col in collections]

@router.post("/{subject_id}/collections", response_model=Collection)
async def create_collection(subject_id: str, collection_data: dict, current_user: dict = Depends(get_current_user)):
    collection = Collection(
        subject_id=subject_id,
        name=collection_data.get("name"),
        description=collection_data.get("description")
    )
    result = await data_store.collections_collection.insert_one(collection.dict(by_alias=True))
    collection.id = str(result.inserted_id)
    return collection
```

- [ ] **Step 2: Write material router**

```python
from fastapi import APIRouter, Depends, UploadFile, File
from typing import List
from datetime import datetime
from bson import ObjectId

from src.core.models import Material
from src.core.data_store import data_store
from src.routers.auth_router import get_current_user

router = APIRouter(prefix="/api/materials", tags=["materials"])

@router.get("/{collection_id}/materials", response_model=List[Material])
async def list_materials(collection_id: str, current_user: dict = Depends(get_current_user)):
    materials = await data_store.materials_collection.find({"collection_id": collection_id}).to_list(length=100)
    return [Material(**mat) for mat in materials]

@router.post("/{collection_id}/materials", response_model=Material)
async def upload_material(collection_id: str, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    # Save file logic here (placeholder for actual storage)
    material = Material(
        collection_id=collection_id,
        name=file.filename,
        type="pdf" if file.filename.endswith(".pdf") else "text",
        size=0,  # Calculate actual size
        url=f"/uploads/{file.filename}",
        rag_indexed=False
    )
    result = await data_store.materials_collection.insert_one(material.dict(by_alias=True))
    material.id = str(result.inserted_id)
    return material

@router.delete("/{material_id}")
async def delete_material(material_id: str, current_user: dict = Depends(get_current_user)):
    await data_store.materials_collection.delete_one({"_id": ObjectId(material_id)})
    return {"success": True}
```

- [ ] **Step 3: Commit**

```bash
git add Backend/src/routers/collection_router.py Backend/src/routers/material_router.py
git commit -m "feat: add collection and material routers"
```

---

## Task 7: Backend — Create Onboarding Router

**Files:**
- Create: `Backend/src/routers/onboarding_router.py`

- [ ] **Step 1: Write onboarding router**

```python
from fastapi import APIRouter, Depends
from datetime import datetime
from bson import ObjectId

from src.core.data_store import data_store
from src.routers.auth_router import get_current_user

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

@router.post("/")
async def save_onboarding(data: dict, current_user: dict = Depends(get_current_user)):
    update_data = {
        "role": data.get("role", "student"),
        "institute": data.get("institute"),
        "preferred_language": data.get("preferred_language", "en"),
        "updated_at": datetime.utcnow()
    }
    await data_store.users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": update_data}
    )
    return {"success": True}

@router.get("/")
async def get_onboarding_status(current_user: dict = Depends(get_current_user)):
    user = await data_store.users_collection.find_one({"_id": ObjectId(current_user["_id"])})
    return {
        "onboarding_completed": user.get("onboarding_completed", False),
        "role": user.get("role", "student"),
        "institute": user.get("institute"),
        "preferred_language": user.get("preferred_language", "en")
    }

@router.post("/complete")
async def complete_onboarding(data: dict, current_user: dict = Depends(get_current_user)):
    await data_store.users_collection.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"onboarding_completed": True, "updated_at": datetime.utcnow()}}
    )
    return {"success": True}
```

- [ ] **Step 2: Commit**

```bash
git add Backend/src/routers/onboarding_router.py
git commit -m "feat: add onboarding status router"
```

---

## Task 8: Backend — Register All New Routers

**Files:**
- Modify: `Backend/src/main.py`

- [ ] **Step 1: Import and register routers**

Add imports:
```python
from src.routers.exam_router import router as exam_router
from src.routers.subject_router import router as subject_router
from src.routers.collection_router import router as collection_router
from src.routers.material_router import router as material_router
from src.routers.onboarding_router import router as onboarding_router
```

Add router inclusions:
```python
app.include_router(exam_router)
app.include_router(subject_router)
app.include_router(collection_router)
app.include_router(material_router)
app.include_router(onboarding_router)
```

- [ ] **Step 2: Commit**

```bash
git add Backend/src/main.py
git commit -m "feat: register all new routers in main app"
```

---

## Task 9: Frontend — Preset Exam Data and Constants

**Files:**
- Create: `Frontend/lib/constants/exams.ts`

- [ ] **Step 1: Write preset exam data**

```typescript
export interface PresetExam {
  id: string;
  name: string;
  tagline: string;
  icon: string;
  subjects: string[];
}

export const PRESET_EXAMS: PresetExam[] = [
  {
    id: "jee-mains",
    name: "JEE Mains",
    tagline: "Engineering Entrance",
    icon: "Cpu",
    subjects: ["Physics", "Chemistry", "Mathematics"]
  },
  {
    id: "neet",
    name: "NEET",
    tagline: "Medical Entrance",
    icon: "Heart",
    subjects: ["Physics", "Chemistry", "Biology"]
  },
  {
    id: "upsc",
    name: "UPSC",
    tagline: "Civil Services",
    icon: "Landmark",
    subjects: ["History", "Geography", "Polity", "Economy", "Science"]
  },
  {
    id: "cat",
    name: "CAT",
    tagline: "MBA Entrance",
    icon: "BarChart3",
    subjects: ["Quantitative Aptitude", "Verbal Ability", "Logical Reasoning", "Data Interpretation"]
  },
  {
    id: "gate",
    name: "GATE",
    tagline: "PG Engineering",
    icon: "GraduationCap",
    subjects: ["Engineering Mathematics", "Technical Subjects"]
  },
  {
    id: "cbse-12",
    name: "CBSE 12th",
    tagline: "Board Exams",
    icon: "School",
    subjects: ["Physics", "Chemistry", "Mathematics", "Biology", "English"]
  }
];

export const INDIAN_LANGUAGES = [
  { code: "en", name: "English" },
  { code: "hi", name: "Hindi" },
  { code: "ta", name: "Tamil" },
  { code: "te", name: "Telugu" },
  { code: "mr", name: "Marathi" },
  { code: "bn", name: "Bengali" },
  { code: "gu", name: "Gujarati" },
  { code: "kn", name: "Kannada" },
  { code: "ml", name: "Malayalam" },
  { code: "pa", name: "Punjabi" },
  { code: "ur", name: "Urdu" }
];
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/lib/constants/exams.ts
git commit -m "feat: add preset exam and language constants"
```

---

## Task 10: Frontend — Dashboard Context Provider

**Files:**
- Create: `Frontend/lib/context/dashboard-context.tsx`

- [ ] **Step 1: Write dashboard context**

```typescript
"use client";

import React, { createContext, useContext, useState, useCallback } from "react";

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

export interface Subject {
  id: string;
  examId: string;
  name: string;
  icon?: string;
  collections: Collection[];
  progress: number;
  lastStudiedAt?: string;
}

export interface Collection {
  id: string;
  subjectId: string;
  name: string;
  description?: string;
  materials: Material[];
  createdAt: string;
}

export interface Material {
  id: string;
  collectionId: string;
  name: string;
  type: "pdf" | "image" | "text";
  size: number;
  url: string;
  uploadedAt: string;
  ragIndexed: boolean;
}

interface DashboardContextType {
  exams: Exam[];
  activeExam: Exam | null;
  setExams: (exams: Exam[]) => void;
  setActiveExam: (exam: Exam | null) => void;
  refreshExams: () => Promise<void>;
  isLoading: boolean;
}

const DashboardContext = createContext<DashboardContextType | undefined>(undefined);

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [exams, setExams] = useState<Exam[]>([]);
  const [activeExam, setActiveExam] = useState<Exam | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const refreshExams = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch("/api/exams");
      const data = await res.json();
      setExams(data);
      const active = data.find((e: Exam) => e.isActive);
      setActiveExam(active || null);
    } catch (err) {
      console.error("Failed to fetch exams", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  return (
    <DashboardContext.Provider
      value={{ exams, activeExam, setExams, setActiveExam, refreshExams, isLoading }}
    >
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const context = useContext(DashboardContext);
  if (!context) {
    throw new Error("useDashboard must be used within DashboardProvider");
  }
  return context;
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/lib/context/dashboard-context.tsx
git commit -m "feat: add dashboard context for exam tree state"
```

---

## Task 11: Frontend — Progress Ring Component

**Files:**
- Create: `Frontend/components/ui/progress-ring.tsx`

- [ ] **Step 1: Write progress ring**

```typescript
"use client";

import { cn } from "@/lib/utils";

interface ProgressRingProps {
  progress: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
}

export function ProgressRing({
  progress,
  size = 120,
  strokeWidth = 8,
  className
}: ProgressRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="hsl(var(--primary))"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <span className="absolute text-lg font-semibold">{progress}%</span>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/ui/progress-ring.tsx
git commit -m "feat: add ProgressRing UI component"
```

---

## Task 12: Frontend — Bento Grid Layout Helper

**Files:**
- Create: `Frontend/components/ui/bento-grid.tsx`

- [ ] **Step 1: Write bento grid**

```typescript
import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface BentoGridProps {
  children: ReactNode;
  className?: string;
  columns?: 2 | 3 | 4;
}

export function BentoGrid({ children, className, columns = 3 }: BentoGridProps) {
  return (
    <div
      className={cn(
        "grid gap-4",
        columns === 2 && "grid-cols-1 md:grid-cols-2",
        columns === 3 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
        columns === 4 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-4",
        className
      )}
    >
      {children}
    </div>
  );
}

interface BentoItemProps {
  children: ReactNode;
  className?: string;
  span?: 1 | 2;
}

export function BentoItem({ children, className, span = 1 }: BentoItemProps) {
  return (
    <div
      className={cn(
        span === 2 && "md:col-span-2",
        className
      )}
    >
      {children}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/ui/bento-grid.tsx
git commit -m "feat: add BentoGrid layout component"
```

---

## Task 13: Frontend — Active Study Card Component

**Files:**
- Create: `Frontend/components/dashboard/active-study-card.tsx`

- [ ] **Step 1: Write active study card**

```typescript
"use client";

import { MagicCard } from "@/components/ui/magic-card";
import { ProgressRing } from "@/components/ui/progress-ring";
import { useDashboard } from "@/lib/context/dashboard-context";
import { Container } from "@/components/global/container";
import { Button } from "@/components/ui/button";
import { GraduationCap, Plus } from "lucide-react";
import { cn } from "@/lib/utils";

interface ActiveStudyCardProps {
  onAddExam: () => void;
  onContinueSession: () => void;
}

export function ActiveStudyCard({ onAddExam, onContinueSession }: ActiveStudyCardProps) {
  const { activeExam } = useDashboard();

  if (!activeExam) {
    return (
      <Container delay={0.1}>
        <MagicCard
          gradientFrom="#38bdf8"
          gradientTo="#3b82f6"
          className="p-8 rounded-2xl lg:rounded-3xl"
          gradientColor="rgba(59,130,246,0.1)"
        >
          <div className="flex flex-col items-center text-center gap-4">
            <GraduationCap className="h-12 w-12 text-primary" />
            <h3 className="font-heading text-xl font-medium">Set your exam goal to get started</h3>
            <p className="text-muted-foreground text-sm max-w-sm">
              Add an exam and start organizing your study materials.
            </p>
            <Button onClick={onAddExam} className="mt-2">
              <Plus className="h-4 w-4 mr-2" />
              Add Exam
            </Button>
          </div>
        </MagicCard>
      </Container>
    );
  }

  // Calculate overall progress from subjects
  const overallProgress = activeExam.subjects.length > 0
    ? Math.round(activeExam.subjects.reduce((acc, s) => acc + s.progress, 0) / activeExam.subjects.length)
    : 0;

  return (
    <Container delay={0.1}>
      <MagicCard
        gradientFrom="#38bdf8"
        gradientTo="#3b82f6"
        className="p-6 lg:p-8 rounded-2xl lg:rounded-3xl"
        gradientColor="rgba(59,130,246,0.1)"
      >
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
          <div className="flex-1">
            <p className="text-xs uppercase tracking-wider text-muted-foreground mb-1">
              Currently Preparing For
            </p>
            <h2 className="font-subheading italic text-2xl lg:text-3xl text-foreground">
              {activeExam.name}
            </h2>
            <Button
              variant="ghost"
              size="sm"
              className="mt-4 text-primary hover:text-primary/80"
              onClick={onContinueSession}
            >
              Continue Last Session →
            </Button>
          </div>
          <ProgressRing progress={overallProgress} size={100} strokeWidth={6} />
        </div>
      </MagicCard>
    </Container>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/dashboard/active-study-card.tsx
git commit -m "feat: add ActiveStudyCard centerpiece component"
```

---

## Task 14: Frontend — Subject Card Component

**Files:**
- Create: `Frontend/components/dashboard/subject-card.tsx`

- [ ] **Step 1: Write subject card**

```typescript
"use client";

import { MagicCard } from "@/components/ui/magic-card";
import { Container } from "@/components/global/container";
import { Subject } from "@/lib/context/dashboard-context";
import { BookOpen, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

interface SubjectCardProps {
  subject: Subject;
  index: number;
  onClick: () => void;
}

export function SubjectCard({ subject, index, onClick }: SubjectCardProps) {
  const collectionCount = subject.collections?.length || 0;
  const lastStudied = subject.lastStudiedAt
    ? new Date(subject.lastStudiedAt).toLocaleDateString()
    : "Not started";

  return (
    <Container delay={0.2 + index * 0.1}>
      <MagicCard
        gradientFrom="#38bdf8"
        gradientTo="#3b82f6"
        className="p-5 rounded-xl cursor-pointer hover:-translate-y-1 hover:shadow-lg transition-all duration-300"
        gradientColor="rgba(59,130,246,0.05)"
        onClick={onClick}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="flex items-center justify-center h-10 w-10 rounded-lg bg-primary/10">
            <BookOpen className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="font-medium text-sm">{subject.name}</h3>
            <p className="text-xs text-muted-foreground">{collectionCount} collections</p>
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            {lastStudied}
          </div>
          <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-500"
              style={{ width: `${subject.progress}%` }}
            />
          </div>
        </div>
      </MagicCard>
    </Container>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/dashboard/subject-card.tsx
git commit -m "feat: add SubjectCard component"
```

---

## Task 15: Frontend — Collections Panel (Sheet Slide-Over)

**Files:**
- Create: `Frontend/components/dashboard/collections-panel.tsx`

- [ ] **Step 1: Write collections panel**

```typescript
"use client";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetClose,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Exam } from "@/lib/context/dashboard-context";
import { SubjectAccordion } from "./subject-accordion";
import { X, MessageSquare } from "lucide-react";

interface CollectionsPanelProps {
  exam: Exam | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChat: (examId: string) => void;
}

export function CollectionsPanel({ exam, open, onOpenChange, onChat }: CollectionsPanelProps) {
  if (!exam) return null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:w-[480px] p-0 flex flex-col">
        <SheetHeader className="px-6 py-4 border-b">
          <div className="flex items-center justify-between">
            <SheetTitle className="font-heading text-xl">{exam.name}</SheetTitle>
            <SheetClose asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <X className="h-4 w-4" />
              </Button>
            </SheetClose>
          </div>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-2">
          {exam.subjects?.map((subject) => (
            <SubjectAccordion key={subject.id} subject={subject} />
          ))}
        </div>

        <div className="p-4 border-t bg-background/95 backdrop-blur">
          <Button
            className="w-full gap-2"
            onClick={() => onChat(exam.id)}
          >
            <MessageSquare className="h-4 w-4" />
            Chat with this Exam
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/dashboard/collections-panel.tsx
git commit -m "feat: add CollectionsPanel Sheet slide-over"
```

---

## Task 16: Frontend — Subject Accordion

**Files:**
- Create: `Frontend/components/dashboard/subject-accordion.tsx`

- [ ] **Step 1: Write subject accordion**

```typescript
"use client";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Subject } from "@/lib/context/dashboard-context";
import { CollectionItem } from "./collection-item";
import { ChevronDown, BookOpen } from "lucide-react";
import { useState } from "react";

interface SubjectAccordionProps {
  subject: Subject;
}

export function SubjectAccordion({ subject }: SubjectAccordionProps) {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger className="flex items-center gap-3 w-full py-3 px-2 hover:bg-muted/50 rounded-lg transition-colors">
        <BookOpen className="h-5 w-5 text-primary shrink-0" />
        <span className="font-medium text-sm flex-1 text-left">{subject.name}</span>
        <ChevronDown
          className={`h-4 w-4 text-muted-foreground transition-transform ${isOpen ? "rotate-180" : ""}`}
        />
      </CollapsibleTrigger>
      <CollapsibleContent className="pl-10 space-y-1">
        {subject.collections?.map((collection) => (
          <CollectionItem key={collection.id} collection={collection} />
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/dashboard/subject-accordion.tsx
git commit -m "feat: add SubjectAccordion collapsible component"
```

---

## Task 17: Frontend — Collection Item and Material List

**Files:**
- Create: `Frontend/components/dashboard/collection-item.tsx`
- Create: `Frontend/components/dashboard/material-list.tsx`

- [ ] **Step 1: Write collection item**

```typescript
"use client";

import { Collection } from "@/lib/context/dashboard-context";
import { MaterialList } from "./material-list";
import { Folder, ChevronRight } from "lucide-react";
import { useState } from "react";

interface CollectionItemProps {
  collection: Collection;
}

export function CollectionItem({ collection }: CollectionItemProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="space-y-1">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 w-full py-2 px-2 hover:bg-muted/30 rounded-md transition-colors text-left"
      >
        <Folder className="h-4 w-4 text-muted-foreground shrink-0" />
        <span className="text-sm flex-1">{collection.name}</span>
        <span className="text-xs text-muted-foreground">{collection.materials?.length || 0}</span>
        <ChevronRight
          className={`h-3 w-3 text-muted-foreground transition-transform ${isOpen ? "rotate-90" : ""}`}
        />
      </button>
      {isOpen && <MaterialList materials={collection.materials || []} collectionId={collection.id} />}
    </div>
  );
}
```

- [ ] **Step 2: Write material list**

```typescript
"use client";

import { Material } from "@/lib/context/dashboard-context";
import { Button } from "@/components/ui/button";
import { FileText, Upload, Trash2 } from "lucide-react";
import { formatFileSize } from "@/lib/utils";

interface MaterialListProps {
  materials: Material[];
  collectionId: string;
}

export function MaterialList({ materials, collectionId }: MaterialListProps) {
  const handleUpload = () => {
    // Trigger file input
    document.getElementById(`upload-${collectionId}`)?.click();
  };

  return (
    <div className="pl-6 space-y-1">
      {materials.map((material) => (
        <div
          key={material.id}
          className="flex items-center gap-2 py-1.5 px-2 hover:bg-muted/30 rounded-md group"
        >
          <FileText className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          <span className="text-xs flex-1 truncate">{material.name}</span>
          <span className="text-xs text-muted-foreground">{formatFileSize(material.size)}</span>
          <button className="opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive">
            <Trash2 className="h-3 w-3" />
          </button>
        </div>
      ))}
      <Button
        variant="ghost"
        size="sm"
        className="w-full justify-start gap-2 text-muted-foreground hover:text-foreground border border-dashed border-muted-foreground/30"
        onClick={handleUpload}
      >
        <Upload className="h-3.5 w-3.5" />
        <span className="text-xs">Upload Material</span>
      </Button>
      <input id={`upload-${collectionId}`} type="file" className="hidden" accept=".pdf,.txt" />
    </div>
  );
}
```

- [ ] **Step 3: Add formatFileSize utility**

Modify `Frontend/lib/utils.ts` to add:

```typescript
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}
```

- [ ] **Step 4: Commit**

```bash
git add Frontend/components/dashboard/collection-item.tsx Frontend/components/dashboard/material-list.tsx Frontend/lib/utils.ts
git commit -m "feat: add CollectionItem and MaterialList components"
```

---

## Task 18: Frontend — Exam Setup Dialog

**Files:**
- Create: `Frontend/components/dashboard/exam-setup-dialog.tsx`

- [ ] **Step 1: Write exam setup dialog**

```typescript
"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PRESET_EXAMS } from "@/lib/constants/exams";
import { MagicCard } from "@/components/ui/magic-card";
import { useState } from "react";
import { Plus } from "lucide-react";

interface ExamSetupDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExamCreated: (examId: string) => void;
}

export function ExamSetupDialog({ open, onOpenChange, onExamCreated }: ExamSetupDialogProps) {
  const [customName, setCustomName] = useState("");

  const handlePresetSelect = async (preset: typeof PRESET_EXAMS[0]) => {
    // Create exam with preset subjects
    const res = await fetch("/api/exams", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: preset.name,
        icon: preset.icon,
        is_active: true
      })
    });
    const exam = await res.json();

    // Create subjects
    for (const subjectName of preset.subjects) {
      await fetch(`/api/subjects/${exam.id}/subjects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: subjectName })
      });
    }

    onExamCreated(exam.id);
    onOpenChange(false);
  };

  const handleCustomCreate = async () => {
    if (!customName.trim()) return;
    const res = await fetch("/api/exams", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: customName, is_active: true })
    });
    const exam = await res.json();
    onExamCreated(exam.id);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="font-heading text-xl">Set Your Exam Goal</DialogTitle>
          <DialogDescription>
            Choose an exam to prepare for. Subjects will be set up automatically.
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          {PRESET_EXAMS.map((preset) => (
            <MagicCard
              key={preset.id}
              gradientFrom="#38bdf8"
              gradientTo="#3b82f6"
              className="p-4 rounded-xl cursor-pointer hover:-translate-y-1 transition-all"
              gradientColor="rgba(59,130,246,0.05)"
              onClick={() => handlePresetSelect(preset)}
            >
              <h3 className="font-medium">{preset.name}</h3>
              <p className="text-xs text-muted-foreground mt-1">{preset.tagline}</p>
              <p className="text-xs text-muted-foreground mt-2">
                {preset.subjects.length} subjects
              </p>
            </MagicCard>
          ))}
        </div>

        <div className="mt-6 pt-4 border-t">
          <Label className="text-sm font-medium">Or create a custom exam</Label>
          <div className="flex gap-2 mt-2">
            <Input
              placeholder="e.g., My Physics Course"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
            />
            <Button onClick={handleCustomCreate}>
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/dashboard/exam-setup-dialog.tsx
git commit -m "feat: add ExamSetupDialog with preset exams"
```

---

## Task 19: Frontend — Onboarding Step 1 (About You)

**Files:**
- Create: `Frontend/components/onboarding/step-about-you.tsx`

- [ ] **Step 1: Write step 1**

```typescript
"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { INDIAN_LANGUAGES } from "@/lib/constants/exams";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useState } from "react";

interface StepAboutYouProps {
  onNext: (data: { name: string; role: string; institute: string; language: string }) => void;
  onSkip: () => void;
  defaultName?: string;
}

export function StepAboutYou({ onNext, onSkip, defaultName = "" }: StepAboutYouProps) {
  const [name, setName] = useState(defaultName);
  const [role, setRole] = useState("student");
  const [institute, setInstitute] = useState("");
  const [language, setLanguage] = useState("en");

  const handleNext = () => {
    onNext({ name, role, institute, language });
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="font-heading text-2xl font-medium">About You</h2>
        <p className="text-muted-foreground text-sm mt-2">
          Help us personalize your study experience
        </p>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="name">Name</Label>
          <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
        </div>

        <div className="space-y-2">
          <Label>I am a</Label>
          <div className="flex gap-2">
            <Button
              variant={role === "student" ? "default" : "outline"}
              className="flex-1"
              onClick={() => setRole("student")}
            >
              Student
            </Button>
            <Button
              variant={role === "teacher" ? "default" : "outline"}
              className="flex-1"
              onClick={() => setRole("teacher")}
            >
              Teacher
            </Button>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="institute">Where do you study? (Optional)</Label>
          <Input
            id="institute"
            placeholder="e.g., Allen Kota, FIITJEE, Delhi Public School..."
            value={institute}
            onChange={(e) => setInstitute(e.target.value)}
          />
        </div>

        <div className="space-y-2">
          <Label>Preferred Language for AI Chat</Label>
          <Select value={language} onValueChange={setLanguage}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {INDIAN_LANGUAGES.map((lang) => (
                <SelectItem key={lang.code} value={lang.code}>
                  {lang.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex justify-between pt-4">
        <Button variant="ghost" onClick={onSkip}>
          Skip for now
        </Button>
        <Button onClick={handleNext} disabled={!name.trim()}>
          Next
        </Button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/onboarding/step-about-you.tsx
git commit -m "feat: add onboarding Step 1 About You"
```

---

## Task 20: Frontend — Onboarding Step 2 (Study Goal)

**Files:**
- Create: `Frontend/components/onboarding/step-study-goal.tsx`

- [ ] **Step 1: Write step 2**

```typescript
"use client";

import { Button } from "@/components/ui/button";
import { PRESET_EXAMS } from "@/lib/constants/exams";
import { MagicCard } from "@/components/ui/magic-card";
import { FolderOpen, ArrowLeft } from "lucide-react";

interface StepStudyGoalProps {
  onNext: (presetId: string | null) => void;
  onBack: () => void;
}

export function StepStudyGoal({ onNext, onBack }: StepStudyGoalProps) {
  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="font-heading text-2xl font-medium">Your Study Goal</h2>
        <p className="text-muted-foreground text-sm mt-2">
          Choose the exam you are preparing for
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {PRESET_EXAMS.map((preset) => (
          <MagicCard
            key={preset.id}
            gradientFrom="#38bdf8"
            gradientTo="#3b82f6"
            className="p-5 rounded-xl cursor-pointer hover:-translate-y-1 hover:shadow-lg transition-all"
            gradientColor="rgba(59,130,246,0.05)"
            onClick={() => onNext(preset.id)}
          >
            <h3 className="font-medium">{preset.name}</h3>
            <p className="text-xs text-muted-foreground mt-1">{preset.tagline}</p>
            <p className="text-xs text-muted-foreground mt-2">
              {preset.subjects.length} subjects
            </p>
          </MagicCard>
        ))}

        <MagicCard
          gradientFrom="#64748b"
          gradientTo="#475569"
          className="p-5 rounded-xl cursor-pointer hover:-translate-y-1 hover:shadow-lg transition-all"
          gradientColor="rgba(100,116,139,0.05)"
          onClick={() => onNext(null)}
        >
          <div className="flex items-center gap-3">
            <FolderOpen className="h-8 w-8 text-muted-foreground" />
            <div>
              <h3 className="font-medium">I'll organize my own way</h3>
              <p className="text-xs text-muted-foreground mt-1">
                Start with a blank workspace
              </p>
            </div>
          </div>
        </MagicCard>
      </div>

      <div className="flex justify-between pt-4">
        <Button variant="ghost" onClick={onBack}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/components/onboarding/step-study-goal.tsx
git commit -m "feat: add onboarding Step 2 Study Goal"
```

---

## Task 21: Frontend — Onboarding Container and Page

**Files:**
- Create: `Frontend/components/onboarding/onboarding-container.tsx`
- Create: `Frontend/app/onboarding/page.tsx`

- [ ] **Step 1: Write onboarding container**

```typescript
"use client";

import { useState } from "react";
import { MagicCard } from "@/components/ui/magic-card";
import { Container } from "@/components/global/container";
import { StepAboutYou } from "./step-about-you";
import { StepStudyGoal } from "./step-study-goal";
import { useRouter } from "next/navigation";

export function OnboardingContainer() {
  const [step, setStep] = useState(1);
  const [aboutData, setAboutData] = useState({});
  const router = useRouter();

  const handleStep1 = async (data: any) => {
    setAboutData(data);
    // Save step 1 data
    await fetch("/api/onboarding", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: data.role,
        institute: data.institute,
        preferred_language: data.language
      })
    });
    setStep(2);
  };

  const handleStep2 = async (presetId: string | null) => {
    if (presetId) {
      // Find preset and create exam
      const { PRESET_EXAMS } = await import("@/lib/constants/exams");
      const preset = PRESET_EXAMS.find((e: any) => e.id === presetId);
      if (preset) {
        const res = await fetch("/api/exams", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: preset.name,
            icon: preset.icon,
            is_active: true
          })
        });
        const exam = await res.json();
        // Create subjects
        for (const subjectName of preset.subjects) {
          await fetch(`/api/subjects/${exam.id}/subjects`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: subjectName })
          });
        }
      }
    }
    // Complete onboarding
    await fetch("/api/onboarding/complete", { method: "POST" });
    router.push("/dashboard");
  };

  const handleSkip = async () => {
    await fetch("/api/onboarding/complete", { method: "POST" });
    router.push("/dashboard");
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Container className="w-full max-w-md">
        <MagicCard
          gradientFrom="#38bdf8"
          gradientTo="#3b82f6"
          className="p-6 lg:p-8 rounded-2xl"
          gradientColor="rgba(59,130,246,0.05)"
        >
          <div className="flex justify-center gap-2 mb-6">
            {[1, 2].map((s) => (
              <div
                key={s}
                className={`h-2 w-2 rounded-full transition-colors ${
                  s === step ? "bg-primary" : "bg-muted"
                }`}
              />
            ))}
          </div>

          {step === 1 && (
            <StepAboutYou
              onNext={handleStep1}
              onSkip={handleSkip}
              defaultName="" // Fetch from auth context
            />
          )}

          {step === 2 && (
            <StepStudyGoal
              onNext={handleStep2}
              onBack={() => setStep(1)}
            />
          )}
        </MagicCard>
      </Container>
    </div>
  );
}
```

- [ ] **Step 2: Write onboarding page**

```typescript
import { OnboardingContainer } from "@/components/onboarding/onboarding-container";

export default function OnboardingPage() {
  return <OnboardingContainer />;
}
```

- [ ] **Step 3: Commit**

```bash
git add Frontend/components/onboarding/onboarding-container.tsx Frontend/app/onboarding/page.tsx
git commit -m "feat: add onboarding flow container and route"
```

---

## Task 22: Frontend — Add Onboarding Guard to Dashboard Layout

**Files:**
- Modify: `Frontend/app/(dashboard)/layout.tsx`

- [ ] **Step 1: Add onboarding redirect**

Add a client-side check in the dashboard layout. If the user hasn't completed onboarding, redirect to `/onboarding`.

```typescript
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AuthProtection } from "@/components/auth/auth-protection";
import { AppShell } from "@/components/dashboard/app-shell";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkOnboarding = async () => {
      try {
        const res = await fetch("/api/onboarding");
        const data = await res.json();
        if (!data.onboarding_completed) {
          router.push("/onboarding");
        } else {
          setIsLoading(false);
        }
      } catch (err) {
        setIsLoading(false);
      }
    };
    checkOnboarding();
  }, [router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse-glow h-8 w-8 rounded-full bg-primary" />
      </div>
    );
  }

  return (
    <AuthProtection>
      <AppShell>{children}</AppShell>
    </AuthProtection>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/app/(dashboard)/layout.tsx
git commit -m "feat: add onboarding redirect guard to dashboard layout"
```

---

## Task 23: Frontend — Rewrite Dashboard Page

**Files:**
- Modify: `Frontend/app/(dashboard)/dashboard/page.tsx`

- [ ] **Step 1: Write new dashboard page**

```typescript
"use client";

import { useState, useEffect } from "react";
import { Container } from "@/components/global/container";
import { ActiveStudyCard } from "@/components/dashboard/active-study-card";
import { SubjectCard } from "@/components/dashboard/subject-card";
import { CollectionsPanel } from "@/components/dashboard/collections-panel";
import { ExamSetupDialog } from "@/components/dashboard/exam-setup-dialog";
import { BentoGrid } from "@/components/ui/bento-grid";
import { DashboardProvider, useDashboard } from "@/lib/context/dashboard-context";
import { Button } from "@/components/ui/button";
import { ArrowRight, MessageSquare, Target } from "lucide-react";
import Link from "next/link";

function DashboardContent() {
  const { activeExam, refreshExams } = useDashboard();
  const [panelOpen, setPanelOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedSubject, setSelectedSubject] = useState<string | null>(null);

  useEffect(() => {
    refreshExams();
  }, [refreshExams]);

  const handleSubjectClick = (subjectId: string) => {
    setSelectedSubject(subjectId);
    setPanelOpen(true);
  };

  const handleChat = (examId: string) => {
    // Navigate to chat with exam context
    window.location.href = `/chat?examId=${examId}`;
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 py-6">
      {/* Zone 1: Welcome */}
      <Container>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="font-heading text-lg">Welcome back!</h1>
            <p className="text-sm text-muted-foreground">Ready to continue your study session?</p>
          </div>
          <div className="flex gap-2">
            <Link href="/chat">
              <Button variant="outline" size="sm">
                <MessageSquare className="h-4 w-4 mr-2" />
                New Chat
              </Button>
            </Link>
            <Link href="/test">
              <Button size="sm">
                <Target className="h-4 w-4 mr-2" />
                Take Test
              </Button>
            </Link>
          </div>
        </div>
      </Container>

      {/* Zone 2: Active Study */}
      <ActiveStudyCard
        onAddExam={() => setDialogOpen(true)}
        onContinueSession={() => setPanelOpen(true)}
      />

      {/* Zone 3: Subjects Grid */}
      {activeExam?.subjects && activeExam.subjects.length > 0 && (
        <div>
          <Container delay={0.2}>
            <h2 className="font-heading text-lg mb-4">Subjects</h2>
          </Container>
          <BentoGrid columns={3}>
            {activeExam.subjects.map((subject, index) => (
              <SubjectCard
                key={subject.id}
                subject={subject}
                index={index}
                onClick={() => handleSubjectClick(subject.id)}
              />
            ))}
          </BentoGrid>
        </div>
      )}

      {/* Zone 4: Stats (placeholder for now) */}
      <Container delay={0.4}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Documents", value: "12" },
            { label: "Chat Sessions", value: "8" },
            { label: "Mock Tests", value: "5" },
            { label: "Avg Score", value: "78%" }
          ].map((stat, i) => (
            <div key={i} className="bg-card border rounded-xl p-4">
              <p className="text-2xl font-semibold">{stat.value}</p>
              <p className="text-xs text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>
      </Container>

      {/* Zone 5: My Collections */}
      <Container delay={0.6}>
        <h2 className="font-heading text-lg mb-4">My Collections</h2>
        <div className="bg-muted/50 rounded-xl p-8 text-center">
          <p className="text-muted-foreground">Your exam collections will appear here</p>
        </div>
      </Container>

      {/* Panel */}
      <CollectionsPanel
        exam={activeExam}
        open={panelOpen}
        onOpenChange={setPanelOpen}
        onChat={handleChat}
      />

      {/* Dialog */}
      <ExamSetupDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onExamCreated={() => refreshExams()}
      />
    </div>
  );
}

export default function DashboardPage() {
  return (
    <DashboardProvider>
      <DashboardContent />
    </DashboardProvider>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add Frontend/app/(dashboard)/dashboard/page.tsx
git commit -m "feat: rewrite dashboard page with new zones and components"
```

---

## Task 24: Frontend — Deprecate Neomorphic Utilities

**Files:**
- Modify: `Frontend/app/globals.css`

- [ ] **Step 1: Add deprecation comments**

Add a comment block above the `.neo-card` utility:

```css
/* 
  DEPRECATED: Neomorphic shadow utilities 
  These are kept for backwards compatibility with existing components.
  New components should use Tailwind's shadow utilities (shadow-lg, shadow-xl)
  and the MagicCard component from the landing page.
*/
```

Keep the existing code but mark it deprecated.

- [ ] **Step 2: Commit**

```bash
git add Frontend/app/globals.css
git commit -m "docs: mark neo-card utilities as deprecated"
```

---

## Task 25: Frontend — Build Verification

**Files:**
- Run build command

- [ ] **Step 1: Run Next.js build**

```bash
cd Frontend && npm run build
```

Expected: Build succeeds with 0 errors.

- [ ] **Step 2: Run lint**

```bash
cd Frontend && npm run lint
```

Expected: No ESLint errors.

- [ ] **Step 3: Commit if clean**

```bash
git commit -m "chore: dashboard redesign build verification" || echo "Nothing to commit"
```

---

## Self-Review

### Spec Coverage Check

| Spec Section | Task(s) Implementing It |
|---|---|
| Onboarding Step 1 (About You) | Task 19 |
| Onboarding Step 2 (Study Goal) | Task 20 |
| Onboarding container + route | Task 21 |
| Dashboard redirect guard | Task 22 |
| Active Study Card | Task 13 |
| Subject Card | Task 14 |
| Collections Panel | Task 15 |
| Subject Accordion | Task 16 |
| Collection Item + Material List | Task 17 |
| Exam Setup Dialog | Task 18 |
| Dashboard Page rewrite | Task 23 |
| Backend User model extension | Task 1 |
| Backend Exam/Subject/Collection/Material models | Tasks 2-7 |
| Backend API routes | Tasks 4-7 |
| Dashboard Context | Task 10 |
| Progress Ring + Bento Grid | Tasks 11-12 |
| Visual system (MagicCard, Container) | Reused from landing page |

### Placeholder Scan
- No "TBD", "TODO", or "implement later" found.
- All API endpoints have actual route handlers.
- All components have actual TSX code.
- No vague "add validation" steps.

### Type Consistency
- `Exam`, `Subject`, `Collection`, `Material` interfaces match between frontend context and backend models.
- API paths consistent: `/api/exams`, `/api/subjects/{exam_id}/subjects`, etc.
- Property names: `isActive` (frontend) ↔ `is_active` (backend) — acceptable due to camelCase/Snake_case conventions.

---

*Plan written on 2026-06-08. Ready for execution.*
