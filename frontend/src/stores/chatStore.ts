/**
 * Chat state management using Zustand.
 * Manages conversations, messages, and UI state with full observability.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ChatMessage, Conversation, ResponseType, Citation } from '@/types';
import { generateId } from '@/lib/utils';
import { sendChatMessageStream, type StreamEvent } from '@/lib/api';

interface ChatState {
  // State
  conversations: Conversation[];
  activeConversationId: string | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  createConversation: () => string;
  setActiveConversation: (id: string | null) => void;
  deleteConversation: (id: string) => void;
  sendMessage: (content: string) => Promise<void>;
  stopGeneration: () => void;
  clearError: () => void;
}

// Store abort controller outside of state (not serializable)
let currentAbortController: AbortController | null = null;

/**
 * Selector to get the active conversation.
 * Use this in components for reactive updates.
 */
export const selectActiveConversation = (state: ChatState): Conversation | null => {
  return state.conversations.find(c => c.id === state.activeConversationId) ?? null;
};

/**
 * Create a new message object.
 */
function createMessage(
  role: 'user' | 'assistant',
  content: string,
  extras?: {
    response_type?: ResponseType;
    citations?: Citation[];
    isTyping?: boolean;
  }
): ChatMessage {
  return {
    id: generateId(),
    role,
    content,
    timestamp: new Date(),
    ...extras,
  };
}

/**
 * Generate a title from the first message.
 */
function generateTitle(message: string): string {
  const cleaned = message.trim();
  if (cleaned.length <= 50) return cleaned;
  return cleaned.slice(0, 47) + '...';
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // Initial state
      conversations: [],
      activeConversationId: null,
      isLoading: false,
      error: null,
      
      // Create a new conversation
      createConversation: () => {
        const id = generateId();
        const now = new Date();
        
        const newConversation: Conversation = {
          id,
          title: 'New conversation',
          messages: [],
          createdAt: now,
          updatedAt: now,
        };
        
        set(state => ({
          conversations: [newConversation, ...state.conversations],
          activeConversationId: id,
        }));
        
        console.log('[Chat] Created new conversation:', id);
        return id;
      },
      
      // Set the active conversation
      setActiveConversation: (id: string | null) => {
        set({ activeConversationId: id, error: null });
        console.log('[Chat] Active conversation:', id);
      },
      
      // Delete a conversation
      deleteConversation: (id: string) => {
        set(state => {
          const newConversations = state.conversations.filter(c => c.id !== id);
          const newActiveId = state.activeConversationId === id
            ? (newConversations[0]?.id ?? null)
            : state.activeConversationId;
          
          return {
            conversations: newConversations,
            activeConversationId: newActiveId,
          };
        });
        console.log('[Chat] Deleted conversation:', id);
      },
      
      // Send a message with streaming
      sendMessage: async (content: string) => {
        const state = get();
        let conversationId = state.activeConversationId;
        
        // Create conversation if needed
        if (!conversationId) {
          conversationId = get().createConversation();
        }
        
        const conversation = get().conversations.find(c => c.id === conversationId);
        if (!conversation) {
          console.error('[Chat] Conversation not found:', conversationId);
          return;
        }
        
        // Add user message and empty assistant message for streaming
        const userMessage = createMessage('user', content);
        const assistantMessageId = generateId();
        const streamingMessage: ChatMessage = {
          id: assistantMessageId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isTyping: true,
        };
        
        set(state => ({
          isLoading: true,
          error: null,
          conversations: state.conversations.map(c => {
            if (c.id !== conversationId) return c;
            
            const isFirstMessage = c.messages.length === 0;
            return {
              ...c,
              title: isFirstMessage ? generateTitle(content) : c.title,
              messages: [...c.messages, userMessage, streamingMessage],
              updatedAt: new Date(),
            };
          }),
        }));
        
        console.log('[Chat] Sending streaming message:', content.slice(0, 50));
        
        // Create abort controller for this request
        currentAbortController = new AbortController();
        
        // Build context from previous messages
        const previousMessages = conversation.messages
          .filter(m => !m.isTyping)
          .slice(-10)
          .map(m => ({
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: m.timestamp.toISOString(),
          }));
        
        let fullContent = '';
        
        await sendChatMessageStream(
          {
            message: content,
            conversation_id: conversationId,
            context: previousMessages.length > 0
              ? {
                  conversation_id: conversationId,
                  messages: previousMessages,
                }
              : undefined,
          },
          // onChunk - update the streaming message with new content
          (chunk: string) => {
            fullContent += chunk;
            set(state => ({
              conversations: state.conversations.map(c => {
                if (c.id !== conversationId) return c;
                return {
                  ...c,
                  messages: c.messages.map(m => 
                    m.id === assistantMessageId
                      ? { ...m, content: fullContent }
                      : m
                  ),
                };
              }),
            }));
          },
          // onComplete - finalize the message with metadata
          (event: StreamEvent) => {
            console.log('[Chat] Stream completed:', {
              type: event.response_type,
              citations: event.citations?.length ?? 0,
              time_ms: event.processing_time_ms,
            });
            
            set(state => ({
              isLoading: false,
              conversations: state.conversations.map(c => {
                if (c.id !== conversationId) return c;
                return {
                  ...c,
                  messages: c.messages.map(m => 
                    m.id === assistantMessageId
                      ? {
                          ...m,
                          content: fullContent,
                          isTyping: false,
                          response_type: event.response_type as ResponseType,
                          citations: event.citations,
                        }
                      : m
                  ),
                  updatedAt: new Date(),
                };
              }),
            }));
          },
          // onError - handle errors
          (errorMsg: string) => {
            console.error('[Chat] Stream error:', errorMsg);
            
            set(state => ({
              isLoading: false,
              error: errorMsg,
              conversations: state.conversations.map(c => {
                if (c.id !== conversationId) return c;
                return {
                  ...c,
                  messages: c.messages.map(m => 
                    m.id === assistantMessageId
                      ? {
                          ...m,
                          content: fullContent || `I apologize, but I encountered an issue: ${errorMsg}\n\nPlease try again.`,
                          isTyping: false,
                          response_type: 'error' as ResponseType,
                        }
                      : m
                  ),
                  updatedAt: new Date(),
                };
              }),
            }));
          },
          currentAbortController.signal,
        );
        
        // Clear abort controller
        currentAbortController = null;
      },
      
      // Stop the current generation
      stopGeneration: () => {
        if (currentAbortController) {
          console.log('[Chat] Stopping generation');
          currentAbortController.abort();
          currentAbortController = null;
          set({ isLoading: false });
        }
      },
      
      // Clear error
      clearError: () => {
        set({ error: null });
      },
    }),
    {
      name: 'qualified-health-chat',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        conversations: state.conversations.map(c => ({
          ...c,
          messages: c.messages.filter(m => !m.isTyping),
        })),
        activeConversationId: state.activeConversationId,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          // Convert date strings back to Date objects
          state.conversations = state.conversations.map(c => ({
            ...c,
            createdAt: new Date(c.createdAt),
            updatedAt: new Date(c.updatedAt),
            messages: c.messages.map(m => ({
              ...m,
              timestamp: new Date(m.timestamp),
            })),
          }));
          console.log('[Chat] Rehydrated', state.conversations.length, 'conversations');
        }
      },
    }
  )
);
