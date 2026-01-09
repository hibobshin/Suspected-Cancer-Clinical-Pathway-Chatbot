/**
 * Type definitions for the Qualified Health frontend.
 * All types are explicit and documented for clarity.
 */

export type MessageRole = 'user' | 'assistant' | 'system';

export type ResponseType = 'answer' | 'clarification' | 'refusal' | 'error';

export interface Citation {
  statement_id: string;
  section: string;
  text?: string;
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  response_type?: ResponseType;
  citations?: Citation[];
  isTyping?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  context?: {
    conversation_id: string;
    messages: Array<{
      role: MessageRole;
      content: string;
      timestamp: string;
    }>;
  };
}

export interface ChatResponse {
  conversation_id: string;
  message: string;
  response_type: ResponseType;
  citations: Citation[];
  follow_up_questions: string[];
  processing_time_ms: number;
  timestamp: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  environment: string;
  timestamp: string;
  checks: Record<string, boolean>;
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  request_id?: string;
}
