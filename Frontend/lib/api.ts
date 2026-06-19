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
    
    localStorage.setItem('token', response.data.access_token);
    return response.data;
  },
  
  signup: async (email: string, password: string, name?: string) => {
    const response = await api.post('/auth/signup', { email, password, name });
    // Auto-login after signup if backend returns a token
    if (response.data?.access_token) {
      localStorage.setItem('token', response.data.access_token);
    }
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
  createChatSession: async (title: string, pdfId?: string) => {
    const response = await api.post('/questions/sessions', { 
      title, 
      pdf_id: pdfId 
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
    const res = await api.post("/api/onboarding/", data);
    return res.data;
  },
  async complete(): Promise<any> {
    const res = await api.post("/api/onboarding/complete");
    return res.data;
  },
};

export default api;
