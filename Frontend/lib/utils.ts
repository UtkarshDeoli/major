import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Message, MessageGroup, MessageRole } from '@/lib/data';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Generate a simple nanoid for use in components
export function nanoid(): string {
  return Math.random().toString(36).substring(2, 10);
}

// Format relative time strings
export function formatRelativeTime(timestamp: string): string {
  const now = new Date();
  const date = new Date(timestamp);
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  // Less than a minute
  if (seconds < 60) {
    return 'just now';
  }
  
  // Less than an hour
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  
  // Less than a day
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  
  // Less than a week
  const days = Math.floor(hours / 24);
  if (days < 7) {
    return `${days}d ago`;
  }
  
  // Format as date
  return date.toLocaleDateString();
}

// Group messages by role and time proximity
export function formatMessageGroups(messages: Message[]): MessageGroup[] {
  const groups: MessageGroup[] = [];
  
  messages.forEach((message, index) => {
    const previousMessage = index > 0 ? messages[index - 1] : null;
    
    // Start a new group if:
    // 1. This is the first message
    // 2. The role changes from the previous message
    // 3. Time gap is more than 5 minutes (300 seconds)
    if (
      !previousMessage || 
      previousMessage.role !== message.role ||
      Math.abs(new Date(message.timestamp).getTime() - new Date(previousMessage.timestamp).getTime()) > 300000
    ) {
      groups.push({
        role: message.role,
        messages: [message]
      });
    } else {
      // Add to the last group
      groups[groups.length - 1].messages.push(message);
    }
  });
  
  return groups;
}