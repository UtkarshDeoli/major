import axios from 'axios';

// API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to URL parameters for routes that need it
const addTokenToRequest = (url: string): string => {
  const token = localStorage.getItem('token');
  if (!token) return url;
  
  // Check if URL already has query parameters
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}authorization=${token}`;
};

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
  
  signup: async (email: string, password: string) => {
    const response = await api.post('/auth/signup', { email, password });
    return response.data;
  },
  
  logout: () => {
    localStorage.removeItem('token');
  },
  
  isAuthenticated: () => {
    return !!localStorage.getItem('token');
  }
};

// PDF APIs
export const pdfAPI = {
  uploadPDF: async (file: File, title?: string, description?: string, tags?: string[]) => {
    const token = localStorage.getItem('token');
    const formData = new FormData();
    formData.append('file', file);
    
    if (title) {
      formData.append('title', title);
    }
    
    if (description) {
      formData.append('description', description);
    }
    
    // Add tags as a JSON string in the form data
    if (tags && tags.length > 0) {
      formData.append('tags', JSON.stringify(tags));
    }
    
    const response = await api.post(
      addTokenToRequest('/pdfs/upload'), 
      formData, 
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    
    return response.data;
  },
  
  listPDFs: async () => {
    try {
      const response = await api.get(addTokenToRequest('/pdfs/'));
      return response.data.pdfs;
    } catch (error) {
      console.error('Error fetching PDFs:', error);
      throw error;
    }
  },
  
  getPDF: async (pdfId: string) => {
    const response = await api.get(addTokenToRequest(`/pdfs/${pdfId}`));
    return response.data;
  },
  
  downloadPDF: async (pdfId: string) => {
    const response = await api.get(
      addTokenToRequest(`/pdfs/${pdfId}/download`), 
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
    const response = await api.post(
      addTokenToRequest('/questions/ask'), 
      { 
        question, 
        pdf_id: pdfId 
      }
    );
    return response.data;
  },
  
  // Stream a question response without saving to history
  askQuestionStream: async (question: string, pdfId?: string, onChunk?: (chunk: any) => void) => {
    const response = await api.post(
      addTokenToRequest('/questions/ask/stream'), 
      { 
        question, 
        pdf_id: pdfId 
      }, 
      {
        responseType: 'text',
        onDownloadProgress: (progressEvent) => {
          if (!progressEvent.event.target) return;
          const text = progressEvent.event.target.responseText || '';
          processSSEChunks(text, onChunk);
        }
      }
    );
    return response.data;
  },
  
  // Chat session management
  createChatSession: async (title: string, pdfId?: string) => {
    const response = await api.post(
      addTokenToRequest('/questions/sessions'), 
      { 
        title, 
        pdf_id: pdfId 
      }
    );
    return response.data;
  },
  
  listChatSessions: async () => {
    const response = await api.get(addTokenToRequest('/questions/sessions'));
    return response.data.sessions;
  },
  
  getChatSession: async (sessionId: string) => {
    const response = await api.get(addTokenToRequest(`/questions/sessions/${sessionId}`));
    return response.data;
  },
  
  // Add a message to a chat session
  addMessageToChat: async (sessionId: string, content: string) => {
    const response = await api.post(
      addTokenToRequest(`/questions/sessions/${sessionId}/messages`), 
      {
        content
      }
    );
    console.log('Response from addMessageToChat:', response.data);
    return response.data;
    
  },
  
  // Add a message to a chat session with streaming response
  addMessageToChatStream: async (sessionId: string, content: string, onChunk?: (chunk: any) => void) => {
    const response = await api.post(
      addTokenToRequest(`/questions/sessions/${sessionId}/messages/stream`), 
      {
        content
      }, 
      {
        responseType: 'text',
        onDownloadProgress: (progressEvent) => {
          if (!progressEvent.event.target) return;
          const text = progressEvent.event.target.responseText || '';
          processSSEChunks(text, onChunk);
        }
      }
    );
    return response.data;
  }
};

// Analysis APIs
export const analysisAPI = {
  // Analyze question papers using syllabus and previous year papers
  analyzeQuestionPapers: async (syllabusId: string, questionPaperIds: string[]) => {
    const response = await api.post(
      addTokenToRequest('/analysis/question-papers'),
      {
        syllabus_pdf_id: syllabusId,
        question_paper_pdf_ids: questionPaperIds
      }
    );
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
    difficultyLevel: string = "medium"
  ) => {
    const response = await api.post(
      addTokenToRequest('/mock-tests/generate'),
      {
        syllabus_pdf_id: syllabusId,
        question_paper_pdf_ids: questionPaperIds,
        notes_pdf_id: notesId,
        num_mcq: numMcq,
        num_text: numText,
        total_marks: totalMarks,
        difficulty_level: difficultyLevel
      }
    );
    return response.data;
  },

  // List all mock tests for the user
  listMockTests: async () => {
    const response = await api.get(addTokenToRequest('/mock-tests/'));
    return response.data.tests;
  },

  // Get a specific mock test
  getMockTest: async (testId: string) => {
    const response = await api.get(addTokenToRequest(`/mock-tests/${testId}`));
    return response.data;
  },

  // Submit a mock test and get analysis
  submitMockTest: async (testId: string, answers: Record<string, string>, timeTaken: number) => {
    const response = await api.post(
      addTokenToRequest(`/mock-tests/${testId}/submit`),
      {
        test_id: testId,
        answers: answers,
        time_taken: timeTaken,
        submitted_at: new Date().toISOString()
      }
    );
    return response.data;
  },

  // Get analysis results by submission ID
  getAnalysisBySubmissionId: async (submissionId: string) => {
    const response = await api.get(addTokenToRequest(`/mock-tests/submissions/${submissionId}/analysis`));
    return response.data;
  }
};

// Teacher APIs
export const teacherAPI = {
  getAnalytics: async () => {
    const response = await api.get('/teachers/analytics');
    return response.data;
  },
};

export default api;