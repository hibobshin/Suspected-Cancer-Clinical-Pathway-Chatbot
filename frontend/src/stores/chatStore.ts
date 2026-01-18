/**
 * Chat state management using Zustand.
 * Manages conversations, messages, and UI state with full observability.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { Artifact, ChatMessage, Conversation, ResponseType, Citation, PathwayRouteType } from '@/types';
import { generateId } from '@/lib/utils';
import { sendChatMessageStream, type StreamEvent } from '@/lib/api';

// Solution modes - selected when creating a new conversation
export type SolutionMode = 'graphrag' | 'rag' | 'custom';

interface ChatState {
  // State
  conversations: Conversation[];
  activeConversationId: string | null;
  isLoading: boolean;
  error: string | null;
  solutionMode: SolutionMode;
  routeType: PathwayRouteType;
  showArtifacts: boolean;
  
  // Actions
  createConversation: () => string;
  setActiveConversation: (id: string | null) => void;
  deleteConversation: (id: string) => void;
  sendMessage: (content: string) => Promise<void>;
  stopGeneration: () => void;
  clearError: () => void;
  setSolutionMode: (mode: SolutionMode) => void;
  setRouteType: (type: PathwayRouteType) => void;
  setShowArtifacts: (show: boolean) => void;
  addPathwayResponse: (message: ChatMessage) => void;
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
    artifacts?: Artifact[];
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
      solutionMode: 'rag' as SolutionMode, // Default to RAG
      routeType: 'cancer_recognition' as PathwayRouteType, // Default custom route
      showArtifacts: true, // Show artifacts by default
      
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
          error: null, // Clear any previous errors
          isLoading: false, // Reset loading state
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
        // Filter out typing messages and messages with empty content (validation requires min_length=1)
        const previousMessages = conversation.messages
          .filter(m => !m.isTyping && m.content && m.content.trim().length > 0)
          .slice(-10)
          .map(m => ({
            role: m.role as 'user' | 'assistant',
            content: m.content.trim(),
            timestamp: m.timestamp.toISOString(),
          }));
        
        let fullContent = '';
        
        // Determine endpoint based on solution mode
        let { solutionMode, routeType } = get();
        
        // Map solution mode to endpoint
        let endpoint: 'rag' | 'graphrag' | 'custom';
        let effectiveRouteType = routeType;
        
        if (solutionMode === 'graphrag') {
          endpoint = 'graphrag';
          effectiveRouteType = 'graph_rag';
        } else if (solutionMode === 'rag') {
          endpoint = 'rag';
          // Keep the current routeType (cancer_recognition, symptom_triage, etc.)
        } else { // custom
          endpoint = 'custom';
          effectiveRouteType = 'custom';
        }
        
        console.log('[Chat] Sending message with route:', {
          solutionMode,
          routeType,
          effectiveRouteType,
          endpoint,
        });
        
        await sendChatMessageStream(
          {
            message: content,
            route_type: effectiveRouteType,
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
              artifacts: event.artifacts?.length ?? 0,
              artifacts_data: event.artifacts,
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
                          artifacts: event.artifacts,
                          pathway_available: event.pathway_available,
                          pathway_spec: event.pathway_spec,
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
          endpoint,
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
      
      // Set solution mode (vendor vs custom)
      setSolutionMode: (mode: SolutionMode) => {
        console.log('[Chat] Solution mode:', mode);
        // When switching to custom, ensure routeType is a custom route
        const currentRouteType = get().routeType;
        if (mode === 'custom' && currentRouteType === 'graph_rag') {
          set({ solutionMode: mode, routeType: 'cancer_recognition' });
        } else {
          set({ solutionMode: mode });
        }
      },
      
      // Set route type for custom mode
      setRouteType: (type: PathwayRouteType) => {
        console.log('[Chat] Route type:', type);
        set({ routeType: type });
      },
      
      // Toggle artifact visibility
      setShowArtifacts: (show: boolean) => {
        set({ showArtifacts: show });
      },
      
      // Add pathway response as a new message
      addPathwayResponse: (message: ChatMessage) => {
        const { activeConversationId, conversations } = get();
        if (!activeConversationId) return;
        
        set({
          conversations: conversations.map(conv => {
            if (conv.id !== activeConversationId) return conv;
            return {
              ...conv,
              messages: [...conv.messages, message],
              updatedAt: new Date(),
            };
          }),
        });
        
        console.log('[Chat] Added pathway response');
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
        solutionMode: state.solutionMode,
        routeType: state.routeType,
        showArtifacts: state.showArtifacts,
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
