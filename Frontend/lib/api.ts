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

// Response interceptor to handle 401 globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error) && error.response?.status === 401) {
      // Token expired or invalid — clear auth
      localStorage.removeItem('token');
      // Only redirect if not already on the login page to prevent infinite loops
      if (typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
        window.location.replace('/login');
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
  askQuestion: async (question: string, pdfId?: string) => {
    const response = await api.post('/questions/ask', { 
      question, 
      pdf_id: pdfId 
    });
    return response.data;
  },
  
  // Stream a question response without saving to history
  askQuestionStream: async (question: string, pdfId?: string, onChunk?: (chunk: any) => void) => {
    const response = await api.post('/questions/ask/stream', { 
      question, 
      pdf_id: pdfId 
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
  addMessageToChat: async (sessionId: string, content: string) => {
    const response = await api.post(`/questions/sessions/${sessionId}/messages`, {
      content
    });
    return response.data;
  },
  
  // Add a message to a chat session with streaming response
  addMessageToChatStream: async (sessionId: string, content: string, onChunk?: (chunk: any) => void) => {
    const response = await api.post(`/questions/sessions/${sessionId}/messages/stream`, {
      content
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

export default api;
