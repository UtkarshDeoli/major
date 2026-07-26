import axios from 'axios';

// API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

// Create axios instance with default config
// NOTE: Do NOT set a global Content-Type here. Let each request set its own.
const api = axios.create({
  baseURL: API_BASE_URL,
});

// Request interceptor to attach Bearer token
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Response interceptor to handle 401 and 402 globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error)) {
      const status = error.response?.status;
      if (status === 401) {
        // Token expired or invalid — clear auth
        localStorage.removeItem('token');
        // Only redirect if not already on the login page to prevent infinite loops
        if (typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
          window.location.replace('/login');
        }
      } else if (status === 402) {
        // Plan limit hit — emit a global upgrade event so UI can show a banner
        if (typeof window !== 'undefined') {
          const detail = (error.response?.data as any)?.detail || {};
          window.dispatchEvent(
            new CustomEvent('orbit:upgrade-required', {
              detail: {
                resource: detail.resource,
                used: detail.used,
                limit: detail.limit,
                plan: detail.plan,
                upgradeUrl: detail.upgrade_url || '/billing',
                message: typeof detail === 'string' ? detail : detail.message,
              },
            })
          );
        }
      }
    }
    return Promise.reject(error);
  }
);

// Authentication APIs
// NOTE: The AuthContext is the single owner of token + user state.
// These helpers perform the HTTP call only; callers are responsible for
// updating localStorage and context state.
export const authAPI = {
  login: async (email: string, password: string) => {
    const response = await api.post('/auth/login', new URLSearchParams({
      'username': email,
      'password': password,
    }), {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  },

  signup: async (email: string, password: string, name?: string) => {
    const response = await api.post('/auth/signup', { email, password, name });
    return response.data;
  },

  logout: () => {
    localStorage.removeItem('token');
  },

  isAuthenticated: () => {
    return !!localStorage.getItem('token');
  },

  getMe: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  }
};

// PDF APIs
export const pdfAPI = {
  uploadPDF: async (file: File, title?: string, description?: string, tags?: string[]) => {
    const formData = new FormData();
    formData.append('file', file);
    
    if (title) {
      formData.append('title', title);
    }
    
    if (description) {
      formData.append('description', description);
    }
    
    // Add tags as repeated form fields so FastAPI collects them into List[str]
    if (tags && tags.length > 0) {
      tags.forEach(tag => formData.append('tags', tag));
    }
    
    // IMPORTANT: Do NOT set Content-Type manually.
    // Axios will automatically set it to 'multipart/form-data' with the correct boundary.
    const response = await api.post('/pdfs/upload', formData);
    
    return response.data;
  },
  
  listPDFs: async () => {
    try {
      const response = await api.get('/pdfs/');
      return response.data.pdfs;
    } catch (error) {
      console.error('Error fetching PDFs:', error);
      throw error;
    }
  },
  
  getPDF: async (pdfId: string) => {
    const response = await api.get(`/pdfs/${pdfId}`);
    return response.data;
  },
  
  downloadPDF: async (pdfId: string) => {
    const response = await api.get(
      `/pdfs/${pdfId}/download`, 
      { responseType: 'blob' }
    );
    return response.data;
  }
};

// Helper function to process SSE chunks
const processSSEChunks = (text: string, onChunk?: (chunk: any) => void) => {
  if (!onChunk) return;
  
  // Split the text by double newlines (standard SSE format)
  const chunks = text.split('\n\n').filter(chunk => chunk.trim() !== '');
  
  // Process each chunk
  chunks.forEach(chunk => {
    // Get the data portion of the SSE event
    const dataMatch = chunk.match(/data: (.*)/);
    if (dataMatch && dataMatch[1]) {
      try {
        const parsedData = JSON.parse(dataMatch[1]);
        onChunk(parsedData);
      } catch (error) {
        console.error('Error parsing SSE data:', error);
      }
    }
  });
};

// Chat APIs
export const chatAPI = {
  // Ask a question without saving to history
  askQuestion: async (question: string, pdfId?: string, imageDataUrl?: string) => {
    const response = await api.post('/questions/ask', {
      question,
      pdf_id: pdfId,
      image_data_url: imageDataUrl,
    });
    return response.data;
  },

  // Stream a question response without saving to history
  askQuestionStream: async (question: string, pdfId?: string, onChunk?: (chunk: any) => void, imageDataUrl?: string) => {
    const response = await api.post('/questions/ask/stream', {
      question,
      pdf_id: pdfId,
      image_data_url: imageDataUrl,
    }, {
      responseType: 'text',
      onDownloadProgress: (progressEvent) => {
        if (!progressEvent.event.target) return;
        const text = progressEvent.event.target.responseText || '';
        processSSEChunks(text, onChunk);
      }
    });
    return response.data;
  },
  
  // Chat session management
  createChatSession: async (title: string, pdfId?: string, docIds?: string[]) => {
    const response = await api.post('/questions/sessions', {
      title,
      pdf_id: pdfId,
      doc_ids: docIds,
    });
    return response.data;
  },
  
  listChatSessions: async () => {
    const response = await api.get('/questions/sessions');
    return response.data.sessions;
  },
  
  getChatSession: async (sessionId: string) => {
    const response = await api.get(`/questions/sessions/${sessionId}`);
    return response.data;
  },
  
  // Add a message to a chat session
  addMessageToChat: async (sessionId: string, content: string, imageDataUrl?: string) => {
    const response = await api.post(`/questions/sessions/${sessionId}/messages`, {
      content,
      image_data_url: imageDataUrl,
    });
    return response.data;
  },

  // Add a message to a chat session with streaming response
  addMessageToChatStream: async (sessionId: string, content: string, onChunk?: (chunk: any) => void, imageDataUrl?: string) => {
    const response = await api.post(`/questions/sessions/${sessionId}/messages/stream`, {
      content,
      image_data_url: imageDataUrl,
    }, {
      responseType: 'text',
      onDownloadProgress: (progressEvent) => {
        if (!progressEvent.event.target) return;
        const text = progressEvent.event.target.responseText || '';
        processSSEChunks(text, onChunk);
      }
    });
    return response.data;
  }
};

// Analysis APIs
export const analysisAPI = {
  // Analyze question papers using syllabus and previous year papers
  analyzeQuestionPapers: async (syllabusId: string, questionPaperIds: string[]) => {
    const response = await api.post('/analysis/question-papers', {
      syllabus_pdf_id: syllabusId,
      question_paper_pdf_ids: questionPaperIds
    });
    return response.data;
  }
};

// Mock Test APIs
export const mockTestAPI = {
  // Generate a new mock test
  generateMockTest: async (
    syllabusId: string,
    questionPaperIds: string[],
    notesId?: string,
    numMcq: number = 15,
    numText: number = 5,
    totalMarks: number = 50,
    difficultyLevel: string = "medium",
    focusTopics?: string[],
    weakTopics?: string[],
    subject?: string,
    studentEmail?: string,
    gradingMode: "auto" | "teacher" = "auto",
    sourceMaterialIds?: string[],
    adaptive: boolean = false,
  ) => {
    const response = await api.post('/mock-tests/generate', {
      syllabus_pdf_id: syllabusId,
      question_paper_pdf_ids: questionPaperIds,
      notes_pdf_id: notesId,
      num_mcq: numMcq,
      num_text: numText,
      total_marks: totalMarks,
      difficulty_level: difficultyLevel,
      focus_topics: focusTopics,
      weak_topics: weakTopics,
      subject,
      student_email: studentEmail,
      grading_mode: gradingMode,
      source_material_ids: sourceMaterialIds,
      adaptive,
    });
    return response.data;
  },

  // List all mock tests for the user
  listMockTests: async () => {
    const response = await api.get('/mock-tests/');
    return response.data.tests;
  },

  // Get a specific mock test
  getMockTest: async (testId: string) => {
    const response = await api.get(`/mock-tests/${testId}`);
    return response.data;
  },

  // List all attempts for a test (student sees own; teacher sees assigned)
  listSubmissions: async (testId: string) => {
    const response = await api.get(`/mock-tests/${testId}/submissions`);
    return response.data.submissions;
  },

  // Submit a mock test and get analysis
  submitMockTest: async (testId: string, answers: Record<string, string>, timeTaken: number) => {
    const response = await api.post(`/mock-tests/${testId}/submit`, {
      test_id: testId,
      answers: answers,
      time_taken: timeTaken,
      submitted_at: new Date().toISOString()
    });
    return response.data;
  },

  // Teacher grades a pending-review submission
  gradeSubmission: async (submissionId: string, grades: { question_id: string; marks_awarded: number; feedback?: string }[]) => {
    const response = await api.post(`/mock-tests/submissions/${submissionId}/grade`, { grades });
    return response.data;
  },

  // Get analysis results by submission ID
  getAnalysisBySubmissionId: async (submissionId: string) => {
    const response = await api.get(`/mock-tests/submissions/${submissionId}/analysis`);
    return response.data;
  }
};

// Analytics APIs
export const analyticsAPI = {
  getStudentAnalytics: async () => {
    const response = await api.get('/analytics/student');
    return response.data;
  },

  getTeacherAnalytics: async () => {
    const response = await api.get('/analytics/teacher');
    return response.data;
  },

  getTeacherAlerts: async () => {
    const response = await api.get('/analytics/teacher/alerts');
    return response.data;
  },

  getTeacherInsights: async () => {
    const response = await api.get('/analytics/teacher/insights');
    return response.data;
  },
};

// Teacher APIs
export const teacherAPI = {
  manageStudent: async (studentEmail: string) => {
    const response = await api.post('/teachers/students/manage', { student_email: studentEmail });
    return response.data;
  },

  unmanageStudent: async (studentEmail: string) => {
    const response = await api.post('/teachers/students/unmanage', { student_email: studentEmail });
    return response.data;
  },

  listManagedStudents: async () => {
    const response = await api.get('/teachers/students');
    return response.data.students;
  },

  getAnalytics: async () => {
    const response = await api.get('/teachers/analytics');
    return response.data;
  },

  assignMockTest: async (testId: string, studentEmail: string) => {
    const response = await api.post(`/mock-tests/${encodeURIComponent(testId)}/assign`, null, {
      params: { student_email: studentEmail },
    });
    return response.data;
  },
};

// ─── Exams / Subjects ─────────────────────────────────────────────────────────
export const examAPI = {
  async listExams(): Promise<any> {
    const res = await api.get("/api/exams/");
    return res.data;
  },
  async createExam(payload: { name: string; icon?: string; is_active?: boolean }): Promise<any> {
    const res = await api.post("/api/exams/", payload);
    return res.data;
  },
  async setActiveExam(examId: string): Promise<any> {
    const res = await api.patch(`/api/exams/${examId}/active`);
    return res.data;
  },
  async createSubject(examId: string, name: string): Promise<any> {
    const res = await api.post(`/api/exams/${examId}/subjects`, { name });
    return res.data;
  },
};

// ─── Onboarding ───────────────────────────────────────────────────────────────
export const onboardingAPI = {
  async getStatus(): Promise<{ onboarding_completed: boolean }> {
    const res = await api.get("/api/onboarding/");
    return res.data;
  },
  async saveStep1(data: {
    name: string;
    role: string;
    institute: string;
    language: string;
  }): Promise<any> {
    const res = await api.post("/api/onboarding/", {
      name: data.name,
      role: data.role,
      institute: data.institute,
      preferred_language: data.language,
    });
    return res.data;
  },
  async complete(): Promise<any> {
    const res = await api.post("/api/onboarding/complete");
    return res.data;
  },
};

// ─── Socratic tutor ───────────────────────────────────────────────────────────
export const socraticAPI = {
  async explain(req: { question: string; concept?: string; doc_ids?: string[] }): Promise<any> {
    const res = await api.post('/socratic/explain', req);
    return res.data;
  },
  async feedback(req: { question: string; user_answer: string; correct_answer?: string }): Promise<any> {
    const res = await api.post('/socratic/feedback', req);
    return res.data;
  },
};

// ─── Focus & Study Plans ───────────────────────────────────────────────────────
export const studyAPI = {
  async startFocusSession(req: { task: string; duration_minutes?: number }): Promise<any> {
    const res = await api.post('/study/focus-sessions', req);
    return res.data;
  },

  async endFocusSession(sessionId: string, req: { completed?: boolean; notes?: string }): Promise<any> {
    const res = await api.patch(`/study/focus-sessions/${encodeURIComponent(sessionId)}`, req);
    return res.data;
  },

  async listFocusSessions(limit?: number): Promise<{ sessions: any[] }> {
    const res = await api.get('/study/focus-sessions', { params: { limit } });
    return res.data;
  },

  async getFocusStats(): Promise<any> {
    const res = await api.get('/study/focus-stats');
    return res.data;
  },

  async createStudyPlan(req: {
    title: string;
    exam_date?: string;
    subjects?: string[];
    weak_topics?: string[];
    hours_per_day?: number;
    weeks?: number;
  }): Promise<any> {
    const res = await api.post('/study/plans', req);
    return res.data;
  },

  async listStudyPlans(): Promise<{ plans: any[] }> {
    const res = await api.get('/study/plans');
    return res.data;
  },

  async deleteStudyPlan(planId: string): Promise<any> {
    const res = await api.delete(`/study/plans/${encodeURIComponent(planId)}`);
    return res.data;
  },

  async updatePlanProgress(planId: string, week: number, day: string, taskIndex: number, completed: boolean): Promise<any> {
    const res = await api.patch(`/study/plans/${encodeURIComponent(planId)}/progress`, {
      week,
      day,
      task_index: taskIndex,
      completed,
    });
    return res.data;
  },
};

// ─── Subjects / Collections / Materials (the "books" tree) ───────────────────
export const subjectAPI = {
  async listSubjects(examId: string): Promise<any[]> {
    const res = await api.get(`/api/exams/${examId}/subjects`);
    return res.data.subjects ?? res.data ?? [];
  },
};

export const collectionAPI = {
  async listCollections(subjectId: string): Promise<any[]> {
    const res = await api.get(`/api/subjects/${subjectId}/collections`);
    return res.data.collections ?? res.data ?? [];
  },
  async createCollection(subjectId: string, name: string, description?: string): Promise<any> {
    const res = await api.post(`/api/subjects/${subjectId}/collections`, { name, description });
    return res.data;
  },
};

export const materialAPI = {
  async listMaterials(collectionId: string): Promise<any[]> {
    const res = await api.get(`/api/collections/${collectionId}/materials`);
    return res.data.materials ?? res.data ?? [];
  },
  async uploadMaterial(collectionId: string, file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post(`/api/collections/${collectionId}/materials`, formData);
    return res.data;
  },
  async deleteMaterial(materialId: string): Promise<any> {
    const res = await api.delete(`/api/materials/${materialId}`);
    return res.data;
  },
};

// ─── Flashcards ───────────────────────────────────────────────────────────────
export const flashcardAPI = {
  async generate(req: { material_ids?: string[]; doc_ids?: string[]; subject?: string; title?: string; num_cards?: number }): Promise<{ deck_id: string; card_count: number }> {
    const res = await api.post('/flashcards/generate', {
      material_ids: req.material_ids ?? [],
      doc_ids: req.doc_ids ?? [],
      subject: req.subject,
      title: req.title,
      num_cards: req.num_cards ?? 15,
    });
    return res.data;
  },
  async listDecks(): Promise<any[]> {
    const res = await api.get('/flashcards/decks');
    return res.data.decks ?? [];
  },
  async getDeck(deckId: string): Promise<any> {
    const res = await api.get(`/flashcards/decks/${deckId}`);
    return res.data;
  },
  async reviewCard(cardId: string, grade: "again" | "hard" | "good" | "easy"): Promise<any> {
    const res = await api.post(`/flashcards/cards/${cardId}/review`, { grade });
    return res.data;
  },
  async deleteDeck(deckId: string): Promise<any> {
    const res = await api.delete(`/flashcards/decks/${deckId}`);
    return res.data;
  },
};

// ─── AI study materials (summaries — right sidebar) ─────────────────────────
export const aiMaterialAPI = {
  async summarize(req: { material_ids?: string[]; doc_ids?: string[]; subject?: string; title?: string; style?: "brief" | "detailed" | "bullet" }): Promise<any> {
    const res = await api.post('/ai-materials/summarize', {
      material_ids: req.material_ids ?? [],
      doc_ids: req.doc_ids ?? [],
      subject: req.subject,
      title: req.title,
      style: req.style ?? "detailed",
    });
    return res.data;
  },
  async list(): Promise<any[]> {
    const res = await api.get('/ai-materials/');
    return res.data.materials ?? [];
  },
  async get(id: string): Promise<any> {
    const res = await api.get(`/ai-materials/${id}`);
    return res.data;
  },
  async delete(id: string): Promise<any> {
    const res = await api.delete(`/ai-materials/${id}`);
    return res.data;
  },
};

// ─── Sample material (NCERT/PYQ for unenrolled students) ────────────────────
export const sampleMaterialAPI = {
  async seed(): Promise<any> {
    const res = await api.post("/api/sample-material/seed");
    return res.data;
  },
};

// ─── Teacher classes / batches ───────────────────────────────────────────────
export const classAPI = {
  async createClass(req: { name: string; description?: string; exam_preset?: string }): Promise<any> {
    const res = await api.post('/classes/', req);
    return res.data;
  },
  async listClasses(): Promise<any[]> {
    const res = await api.get('/classes/');
    return res.data.classes ?? [];
  },
  async getClass(classId: string): Promise<any> {
    const res = await api.get(`/classes/${classId}`);
    return res.data;
  },
  async previewEnroll(code: string): Promise<any> {
    const res = await api.get(`/classes/enroll/${encodeURIComponent(code)}`);
    return res.data;
  },
  async enroll(enrollCode: string): Promise<any> {
    const res = await api.post('/classes/enroll', { enroll_code: enrollCode });
    return res.data;
  },
  async removeStudent(classId: string, studentEmail: string): Promise<any> {
    const res = await api.delete(`/classes/${classId}/students/${encodeURIComponent(studentEmail)}`);
    return res.data;
  },
};

// ─── Subscriptions / Billing ────────────────────────────────────────────────
export const subscriptionAPI = {
  async getPlans(): Promise<{ plans: any[] }> {
    const res = await api.get('/subscriptions/plans');
    return res.data;
  },

  async getMe(): Promise<any> {
    const res = await api.get('/subscriptions/me');
    return res.data;
  },

  async checkout(plan: 'pro' | 'premium', billing_cycle: 'monthly' | 'yearly' = 'monthly'): Promise<any> {
    const res = await api.post('/subscriptions/checkout', { plan, billing_cycle });
    return res.data;
  },

  async verify(payload: {
    razorpay_payment_id: string;
    razorpay_subscription_id?: string;
    razorpay_signature: string;
  }): Promise<any> {
    const res = await api.post('/subscriptions/verify', payload);
    return res.data;
  },

  async cancel(): Promise<any> {
    const res = await api.post('/subscriptions/cancel');
    return res.data;
  },

  async getInvoices(): Promise<{ invoices: any[] }> {
    const res = await api.get('/subscriptions/invoices');
    return res.data;
  },
};

// ─── Organizations / Seat licenses ────────────────────────────────────────────
export const orgAPI = {
  async create(payload: {
    name: string;
    brand_name?: string;
    tier: 'pro' | 'premium';
    seats_total?: number;
    billing_cycle?: 'monthly' | 'yearly';
  }): Promise<any> {
    const res = await api.post('/orgs/', payload);
    return res.data;
  },

  async getMe(): Promise<{ org: any; members: any[]; seats: { used: number; total: number } }> {
    const res = await api.get('/orgs/me');
    return res.data;
  },

  async invite(payload: { member_role: 'teacher' | 'student'; email?: string }): Promise<any> {
    const res = await api.post('/orgs/invite', payload);
    return res.data;
  },

  async listMembers(): Promise<{ members: any[] }> {
    const res = await api.get('/orgs/members');
    return res.data;
  },

  async removeMember(memberEmail: string): Promise<any> {
    const res = await api.delete(`/orgs/members/${encodeURIComponent(memberEmail)}`);
    return res.data;
  },

  async addSeats(addSeats: number): Promise<any> {
    const res = await api.post('/orgs/seats', { add_seats: addSeats });
    return res.data;
  },

  async enroll(code: string): Promise<any> {
    const res = await api.post(`/orgs/enroll/${encodeURIComponent(code)}`);
    return res.data;
  },

  async previewEnroll(code: string): Promise<any> {
    const res = await api.get(`/orgs/enroll/${encodeURIComponent(code)}`);
    return res.data;
  },
};

// ─── Platform Admin ───────────────────────────────────────────────────────────
export const adminAPI = {
  async listUsers(params?: { role?: string; org_id?: string; limit?: number; skip?: number }): Promise<{ users: any[]; total: number }> {
    const res = await api.get('/admin/users', { params });
    return res.data;
  },

  async updateRole(email: string, role: string): Promise<any> {
    const res = await api.patch(`/admin/users/${encodeURIComponent(email)}/role`, { role });
    return res.data;
  },

  async updateStatus(email: string, active: boolean): Promise<any> {
    const res = await api.patch(`/admin/users/${encodeURIComponent(email)}/status`, { active });
    return res.data;
  },

  async listOrgs(): Promise<{ orgs: any[] }> {
    const res = await api.get('/admin/orgs');
    return res.data;
  },

  async updateOrg(orgId: string, payload: { status?: string; seats_total?: number; expires_at?: string }): Promise<any> {
    const res = await api.patch(`/admin/orgs/${encodeURIComponent(orgId)}`, payload);
    return res.data;
  },

  async listSubscriptions(): Promise<{ subscriptions: any[]; payments: any[] }> {
    const res = await api.get('/admin/subscriptions');
    return res.data;
  },

  async activateSubscription(userId: string, plan: string, days: number = 30): Promise<any> {
    const res = await api.post(`/admin/subscriptions/${encodeURIComponent(userId)}/activate`, { plan, days });
    return res.data;
  },

  async getAnalytics(): Promise<any> {
    const res = await api.get('/admin/analytics');
    return res.data;
  },
};

export default api;
