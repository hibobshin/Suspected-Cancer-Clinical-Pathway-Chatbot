import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  MessageSquare,
  Trash2,
  Stethoscope,
  Home,
  Settings,
  Database,
  Sparkles,
  Brain,
  X,
} from 'lucide-react';
import { useChatStore, type SolutionMode } from '@/stores/chatStore';
import { formatDate, cn, truncate } from '@/lib/utils';

export function ChatSidebar() {
  const {
    conversations,
    activeConversationId,
    createConversation,
    setActiveConversation,
    deleteConversation,
    setSolutionMode,
  } = useChatStore();
  const navigate = useNavigate();
  const [showModeDialog, setShowModeDialog] = useState(false);
  
  const handleNewConversation = () => {
    setShowModeDialog(true);
  };
  
  const handleModeSelect = (mode: SolutionMode) => {
    setSolutionMode(mode);
    const newId = createConversation();
    setShowModeDialog(false);
    // Navigate immediately to the new conversation to prevent race conditions
    navigate(`/chat/${newId}`, { replace: true });
  };
  
  return (
    <aside className="w-72 h-full flex flex-col bg-surface-900 text-white">
      {/* Header */}
      <div className="p-4 border-b border-surface-800">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25 group-hover:scale-105 transition-transform">
            <Stethoscope className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-display font-bold text-lg">Qualified Health</h1>
            <p className="text-xs text-surface-400">Clinical Decision Support</p>
          </div>
        </Link>
      </div>
      
      {/* New Chat Button */}
      <div className="p-3">
        <button
          onClick={handleNewConversation}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-xl bg-surface-800 hover:bg-surface-700 border border-surface-700 hover:border-surface-600 transition-all group"
        >
          <Plus className="w-5 h-5 text-surface-400 group-hover:text-white transition-colors" />
          <span className="font-medium text-surface-300 group-hover:text-white transition-colors">
            New conversation
          </span>
        </button>
      </div>
      
      {/* Mode Selection Dialog */}
      <AnimatePresence>
        {showModeDialog && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setShowModeDialog(false)}
              className="fixed inset-0 bg-black/50 z-40"
            />
            {/* Dialog */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="fixed inset-0 z-50 flex items-center justify-center p-4"
              onClick={() => setShowModeDialog(false)}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-6"
              >
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-display font-bold text-surface-900">
                    Choose Conversation Mode
                  </h3>
                  <button
                    onClick={() => setShowModeDialog(false)}
                    className="p-2 hover:bg-surface-100 rounded-lg transition-colors"
                  >
                    <X className="w-5 h-5 text-surface-500" />
                  </button>
                </div>
                
                <div className="space-y-3">
                  {/* GraphRAG Option */}
                  <button
                    onClick={() => handleModeSelect('graphrag')}
                    className="w-full text-left p-4 rounded-xl border-2 border-surface-200 hover:border-violet-300 hover:bg-violet-50 transition-all group"
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-violet-600 flex items-center justify-center flex-shrink-0">
                        <Database className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-surface-900 mb-1">GraphRAG</h4>
                        <p className="text-sm text-surface-600">
                          Knowledge graph queries powered by ArangoDB GraphRAG
                        </p>
                      </div>
                    </div>
                  </button>
                  
                  {/* RAG Option */}
                  <button
                    onClick={() => handleModeSelect('rag')}
                    className="w-full text-left p-4 rounded-xl border-2 border-surface-200 hover:border-primary-300 hover:bg-primary-50 transition-all group"
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center flex-shrink-0">
                        <Sparkles className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-surface-900 mb-1">RAG</h4>
                        <p className="text-sm text-surface-600">
                          Classic RAG with NICE NG12 guidelines and traceable artifacts
                        </p>
                      </div>
                    </div>
                  </button>
                  
                  {/* Custom Option */}
                  <button
                    onClick={() => handleModeSelect('custom')}
                    className="w-full text-left p-4 rounded-xl border-2 border-surface-200 hover:border-blue-300 hover:bg-blue-50 transition-all group"
                  >
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center flex-shrink-0">
                        <Brain className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-surface-900 mb-1">Custom</h4>
                        <p className="text-sm text-surface-600">
                          Flexible custom implementation for specialized use cases
                        </p>
                      </div>
                    </div>
                  </button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
      
      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1 no-scrollbar">
        <AnimatePresence mode="popLayout">
          {conversations.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-8"
            >
              <MessageSquare className="w-8 h-8 text-surface-600 mx-auto mb-3" />
              <p className="text-sm text-surface-500">No conversations yet</p>
              <p className="text-xs text-surface-600 mt-1">
                Start a new conversation to ask about cancer referral pathways
              </p>
            </motion.div>
          ) : (
            conversations.map((conversation) => (
              <motion.div
                key={conversation.id}
                layout
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="group relative"
              >
                <button
                  onClick={() => setActiveConversation(conversation.id)}
                  className={cn(
                    'w-full text-left px-4 py-3 rounded-xl transition-all',
                    activeConversationId === conversation.id
                      ? 'bg-surface-700 border border-surface-600'
                      : 'hover:bg-surface-800 border border-transparent'
                  )}
                >
                  <div className="flex items-start gap-3">
                    <MessageSquare className={cn(
                      'w-4 h-4 mt-0.5 flex-shrink-0',
                      activeConversationId === conversation.id
                        ? 'text-primary-400'
                        : 'text-surface-500'
                    )} />
                    <div className="flex-1 min-w-0">
                      <p className={cn(
                        'text-sm font-medium truncate',
                        activeConversationId === conversation.id
                          ? 'text-white'
                          : 'text-surface-300'
                      )}>
                        {truncate(conversation.title, 30)}
                      </p>
                      <p className="text-xs text-surface-500 mt-0.5">
                        {formatDate(conversation.updatedAt)}
                      </p>
                    </div>
                  </div>
                </button>
                
                {/* Delete button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(conversation.id);
                  }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-lg opacity-0 group-hover:opacity-100 hover:bg-surface-600 transition-all"
                  title="Delete conversation"
                >
                  <Trash2 className="w-4 h-4 text-surface-400 hover:text-red-400" />
                </button>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
      
      {/* Footer */}
      <div className="p-3 border-t border-surface-800 space-y-1">
        <Link
          to="/"
          className="flex items-center gap-3 px-4 py-2.5 rounded-lg text-surface-400 hover:text-white hover:bg-surface-800 transition-all"
        >
          <Home className="w-4 h-4" />
          <span className="text-sm">Back to Home</span>
        </Link>
        <button
          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-surface-400 hover:text-white hover:bg-surface-800 transition-all"
          disabled
          title="Settings coming soon"
        >
          <Settings className="w-4 h-4" />
          <span className="text-sm">Settings</span>
        </button>
      </div>
    </aside>
  );
}
