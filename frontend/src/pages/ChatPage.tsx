import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ChatWindow } from '@/components/ChatWindow';
import { useChatStore } from '@/stores/chatStore';

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const { setActiveConversation, activeConversationId, conversations } = useChatStore();
  
  // Single effect to handle URL <-> state synchronization
  useEffect(() => {
    // If there's a conversationId in the URL
    if (conversationId) {
      // Check if this conversation exists
      const exists = conversations.some(c => c.id === conversationId);
      if (exists) {
        // Only update state if it's different
        if (activeConversationId !== conversationId) {
          setActiveConversation(conversationId);
        }
      } else {
        // Conversation not found, redirect to chat root
        navigate('/chat', { replace: true });
      }
    } else {
      // No conversationId in URL, but we have an active conversation
      if (activeConversationId) {
        navigate(`/chat/${activeConversationId}`, { replace: true });
      }
    }
  }, [conversationId, activeConversationId, conversations, setActiveConversation, navigate]);
  
  return (
    <div className="h-screen flex bg-surface-50">
      <ChatSidebar />
      <ChatWindow />
    </div>
  );
}
