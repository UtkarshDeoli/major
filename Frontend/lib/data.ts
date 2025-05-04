import { nanoid } from './utils';

// User
export interface User {
  email: string;
}

// Document Types
export interface Document {
  id: string;
  title: string;
  filename: string;
  size: number;
  uploadedAt: string;
  file_path: string;
  processed: boolean;
  user_id: string;
  tags?: string[];
  page_count?: number;
  description?: string;
  vector_db_path?: string;
}

// Chat Message Types
export enum MessageRole {
  User = 'user',
  Assistant = 'assistant',
  System = 'system',
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  attachments?: string[];
  reactions?: Record<string, number>;
}

// Create a new message
export function createMessage(message: Message): Message {
  return {
    id: message.id || nanoid(),
    role: message.role,
    content: message.content,
    timestamp: message.timestamp || new Date().toISOString(),
    attachments: message.attachments || [],
    reactions: message.reactions || {},
  };
}

// Chat Session Types
export interface ChatSession {
  id: string;
  title: string;
  pdf_id?: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
  user_id: string;
  pdf_title?: string;
}

// Convert backend document to frontend document
export function convertApiDocumentToDocument(apiDocument: any): Document {
  return {
    id: apiDocument.id,
    title: apiDocument.title || apiDocument.filename,
    filename: apiDocument.filename,
    size: apiDocument.size,
    uploadedAt: apiDocument.upload_date,
    file_path: apiDocument.file_path,
    processed: apiDocument.processed,
    user_id: apiDocument.user_id,
    tags: apiDocument.tags || [],
    page_count: apiDocument.page_count,
    description: apiDocument.description,
    vector_db_path: apiDocument.vector_db_path,
  };
}

// Convert backend chat session to frontend chat session
export function convertApiSessionToSession(apiSession: any): ChatSession {
  return {
    id: apiSession.id,
    title: apiSession.title,
    pdf_id: apiSession.pdf_id,
    created_at: apiSession.created_at,
    updated_at: apiSession.updated_at,
    user_id: apiSession.user_id,
    messages: apiSession.messages ? apiSession.messages.map((msg: any) => ({
      id: msg.id || nanoid(),
      role: msg.role,
      content: msg.content,
      timestamp: msg.timestamp,
    })) : [],
  };
}

// Mock data for development - will be replaced with actual API calls
export const DEFAULT_DOCUMENTS: Document[] = [
  {
    id: '1',
    title: 'Physics 101 Textbook',
    filename: 'physics-101.pdf',
    size: 2500000,
    uploadedAt: '2023-01-15T12:30:00Z',
    tags: ['physics', 'textbook'],
    file_path: '/uploads/physics-101.pdf',
    processed: true,
    user_id: '123',
    page_count: 120
  },
  {
    id: '2',
    title: 'Introduction to Machine Learning',
    filename: 'intro-ml.pdf',
    size: 3500000,
    uploadedAt: '2023-02-10T15:45:00Z',
    tags: ['machine learning', 'computer science'],
    file_path: '/uploads/intro-ml.pdf',
    processed: true,
    user_id: '123',
    page_count: 87
  },
  {
    id: '3',
    title: 'Organic Chemistry Notes',
    filename: 'organic-chem-notes.pdf',
    size: 1200000,
    uploadedAt: '2023-03-05T09:15:00Z',
    tags: ['chemistry', 'notes'],
    file_path: '/uploads/organic-chem-notes.pdf',
    processed: true,
    user_id: '123',
    page_count: 45
  }
];

export const DEFAULT_MESSAGES: Message[] = [
  createMessage({
    id: '1',
    role: MessageRole.System,
    content: 'I am an AI tutor that can help you understand your course materials.',
    timestamp: new Date(Date.now() - 3600000).toISOString(),
  }),
  createMessage({
    id: '2',
    role: MessageRole.User,
    content: 'Can you explain the concept of force in physics?',
    timestamp: new Date(Date.now() - 3500000).toISOString(),
  }),
  createMessage({
    id: '3',
    role: MessageRole.Assistant,
    content: 'Force is a push or pull on an object that can cause it to accelerate. According to Newton\'s second law, force equals mass times acceleration (F = ma). This fundamental relationship explains how objects move when forces are applied to them.',
    timestamp: new Date(Date.now() - 3400000).toISOString(),
  }),
];

export const DEFAULT_CHAT_HISTORY: ChatSession[] = [
  {
    id: '1',
    title: 'Physics Force Concepts',
    pdf_id: '1',
    pdf_title: 'Physics 101 Textbook',
    created_at: new Date(Date.now() - 86400000).toISOString(),
    updated_at: new Date(Date.now() - 3400000).toISOString(),
    messages: DEFAULT_MESSAGES,
    user_id: '123'
  },
  {
    id: '2',
    title: 'Machine Learning Algorithms',
    pdf_id: '2',
    pdf_title: 'Introduction to Machine Learning',
    created_at: new Date(Date.now() - 172800000).toISOString(),
    updated_at: new Date(Date.now() - 86400000).toISOString(),
    messages: [],
    user_id: '123'
  },
];