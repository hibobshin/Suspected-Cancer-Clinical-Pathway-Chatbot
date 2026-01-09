import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ChatSidebar } from '@/components/ChatSidebar';
import { ChatWindow } from '@/components/ChatWindow';
import { useChatStore } from '@/stores/chatStore';

export function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const { setActiveConversation, activeConversationId, conversations } = useChatStore();
  
  // Sync URL with active conversation
  useEffect(() => {
    if (conversationId) {
      const exists = conversations.some(c => c.id === conversationId);
      if (exists) {
        setActiveConversation(conversationId);
      } else {
        // Conversation not found, redirect to chat root
        navigate('/chat', { replace: true });
      }
    }
  }, [conversationId, conversations, setActiveConversation, navigate]);
  
  // Update URL when active conversation changes
  useEffect(() => {
    if (activeConversationId && activeConversationId !== conversationId) {
      navigate(`/chat/${activeConversationId}`, { replace: true });
    } else if (!activeConversationId && conversationId) {
      navigate('/chat', { replace: true });
    }
  }, [activeConversationId, conversationId, navigate]);
  
  return (
    <div className="h-screen flex bg-surface-50">
      <ChatSidebar />
      <ChatWindow />
    </div>
  );
}
