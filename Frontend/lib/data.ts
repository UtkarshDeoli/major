export enum MessageRole {
  User = 'user',
  Assistant = 'assistant',
  System = 'system'
}

export interface Message {
  id: string
  role: MessageRole
  content: string
  timestamp: string
  attachments?: string[]
  reactions?: Record<string, number>
}

export interface MessageGroup {
  role: MessageRole
  messages: Message[]
}

export interface Document {
  id: string
  title: string
  fileName: string
  url: string
  type: 'pdf' | 'doc' | 'docx'
  uploadedAt: string
  tags: string[]
}

export interface ChatHistory {
  id: string;
  documentId: string;
  documentTitle: string;
  messages: Message[];
  lastUpdated: string;
  title: string;
}

// Sample documents
export const DEFAULT_DOCUMENTS: Document[] = [
  {
    id: 'doc1',
    title: 'Mathematics: Calculus Basics',
    fileName: 'calculus-basics.pdf',
    url: '/documents/calculus-basics.pdf',
    type: 'pdf',
    uploadedAt: '2025-03-12T10:30:00Z',
    tags: ['math', 'calculus']
  },
  {
    id: 'doc2',
    title: 'Physics: Laws of Motion',
    fileName: 'laws-of-motion.pdf',
    url: '/documents/laws-of-motion.pdf',
    type: 'pdf',
    uploadedAt: '2025-03-10T14:15:00Z',
    tags: ['physics', 'mechanics']
  },
  {
    id: 'doc3',
    title: 'Chemistry: Organic Compounds',
    fileName: 'organic-chemistry.pdf',
    url: '/documents/organic-chemistry.pdf',
    type: 'pdf',
    uploadedAt: '2025-03-05T09:45:00Z',
    tags: ['chemistry', 'organic']
  }
];

// Sample messages
export const DEFAULT_MESSAGES: Message[] = [
  {
    id: 'msg1',
    role: MessageRole.System,
    content: 'I am your personal learning assistant. How can I help you study today?',
    timestamp: '2025-03-15T09:00:00Z'
  },
  {
    id: 'msg2',
    role: MessageRole.User,
    content: 'Can you explain the concept of derivatives in calculus?',
    timestamp: '2025-03-15T09:01:00Z'
  },
  {
    id: 'msg3',
    role: MessageRole.Assistant,
    content: "A derivative measures the rate at which a function's output changes with respect to its input. It's written as f'(x) or df/dx.\n\nThe concept comes from calculating the slope of a tangent line at any point on a curve. Here's a simple explanation:\n\n1. The derivative of a constant is always 0\n2. For a function f(x) = x^n, the derivative is n*x^(n-1)\n3. There are special rules for composite functions (chain rule), products, and quotients\n\nWould you like me to go deeper into any specific aspect of derivatives?",
    timestamp: '2025-03-15T09:01:30Z',
    reactions: { '👍': 1 }
  },
  {
    id: 'msg4',
    role: MessageRole.User,
    content: 'Can you give me a few practice problems to solve?',
    timestamp: '2025-03-15T09:02:15Z'
  },
  {
    id: 'msg5',
    role: MessageRole.Assistant,
    content: "Here are some practice problems on derivatives:\n\n1. Find the derivative of f(x) = 3x² - 2x + 5\n2. Find f'(x) if f(x) = x³ + 7x - 9\n3. Calculate the derivative of g(x) = (2x + 1)(x - 3)\n4. Find the derivative of h(x) = sin(x) * cos(x)\n5. What is the derivative of j(x) = e^(2x) / x\n\nLet me know when you've tried them, and I'll provide the solutions!",
    timestamp: '2025-03-15T09:03:00Z'
  }
];

export const createMessage = (props: Partial<Message> & Pick<Message, 'id' | 'role'>): Message => {
  return {
    id: props.id,
    role: props.role,
    content: props.content || '',
    timestamp: props.timestamp || new Date().toISOString(),
    attachments: props.attachments || [],
    reactions: props.reactions || {}
  };
};

// Mock chat history data
export const DEFAULT_CHAT_HISTORY: ChatHistory[] = [
  {
    id: 'chat-1',
    documentId: 'doc-1',
    documentTitle: 'Advanced Mathematics Study Guide',
    messages: [
      createMessage({
        id: 'msg-1',
        role: MessageRole.User,
        content: 'Can you explain the concept of derivatives?',
        timestamp: '2025-05-01T10:30:00Z',
      }),
      createMessage({
        id: 'msg-2',
        role: MessageRole.Assistant,
        content: 'In calculus, a derivative measures the sensitivity to change of a function\'s output with respect to its input. It helps us understand the rate at which a quantity changes with respect to another quantity. For example, the derivative of position with respect to time is velocity.',
        timestamp: '2025-05-01T10:30:15Z',
      }),
    ],
    lastUpdated: '2025-05-01T10:30:15Z',
    title: 'Derivatives explanation',
  },
  {
    id: 'chat-2',
    documentId: 'doc-2',
    documentTitle: 'Physics Fundamentals',
    messages: [
      createMessage({
        id: 'msg-3',
        role: MessageRole.User,
        content: 'What are Newton\'s laws of motion?',
        timestamp: '2025-05-02T14:20:00Z',
      }),
      createMessage({
        id: 'msg-4',
        role: MessageRole.Assistant,
        content: 'Newton\'s three laws of motion are: 1) An object at rest stays at rest, and an object in motion stays in motion unless acted upon by an external force. 2) Force equals mass times acceleration (F=ma). 3) For every action, there is an equal and opposite reaction.',
        timestamp: '2025-05-02T14:20:30Z',
      }),
    ],
    lastUpdated: '2025-05-02T14:20:30Z',
    title: 'Newton\'s laws of motion',
  },
  {
    id: 'chat-3',
    documentId: 'doc-3',
    documentTitle: 'Chemistry Handbook',
    messages: [
      createMessage({
        id: 'msg-5',
        role: MessageRole.User,
        content: 'Explain the periodic table organization',
        timestamp: '2025-05-03T09:15:00Z',
      }),
      createMessage({
        id: 'msg-6',
        role: MessageRole.Assistant,
        content: 'The periodic table is organized with elements in order of increasing atomic number. Elements are arranged in rows (periods) and columns (groups). Elements in the same group have similar chemical properties because they have the same number of valence electrons.',
        timestamp: '2025-05-03T09:15:25Z',
      }),
    ],
    lastUpdated: '2025-05-03T09:15:25Z',
    title: 'Periodic table organization',
  }
];