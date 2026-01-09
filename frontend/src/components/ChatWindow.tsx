import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Square,
  Sparkles,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  XCircle,
  Stethoscope,
} from 'lucide-react';
import { useChatStore, selectActiveConversation } from '@/stores/chatStore';
import { cn, parseCitations } from '@/lib/utils';
import type { ChatMessage, ResponseType } from '@/types';

const EXAMPLE_PROMPTS = [
  {
    text: "45-year-old with visible haematuria. What's the referral pathway?",
    category: "Urological",
  },
  {
    text: "When should I order a FIT test for bowel symptoms?",
    category: "Colorectal",
  },
  {
    text: "Persistent hoarseness in a 50-year-old smoker. Next steps?",
    category: "Head & Neck",
  },
  {
    text: "Unexplained breast lump in a 35-year-old woman.",
    category: "Breast",
  },
];

export function ChatWindow() {
  const activeConversation = useChatStore(selectActiveConversation);
  const { sendMessage, isLoading, createConversation, stopGeneration } = useChatStore();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  const messages = activeConversation?.messages ?? [];
  
  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    
    const message = input.trim();
    setInput('');
    await sendMessage(message);
  };
  
  const handleExampleClick = (prompt: string) => {
    setInput(prompt);
    inputRef.current?.focus();
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };
  
  return (
    <main className="flex-1 flex flex-col h-full bg-white">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState onPromptClick={handleExampleClick} onCreate={createConversation} />
        ) : (
          <div className="max-w-3xl mx-auto py-8 px-4 space-y-6">
            <AnimatePresence mode="popLayout">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {/* Input Area */}
      <div className="border-t border-surface-200 bg-surface-50 p-4">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
          <div className="relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about cancer recognition and referral pathways..."
              className="w-full px-4 py-3 pr-14 rounded-2xl border border-surface-200 bg-white text-surface-900 placeholder:text-surface-400 resize-none focus:border-primary-500 focus:ring-4 focus:ring-primary-500/10 transition-all"
              rows={1}
              disabled={isLoading}
            />
            {isLoading ? (
              <button
                type="button"
                onClick={stopGeneration}
                className="absolute right-2 bottom-2 p-2.5 rounded-xl bg-red-500 text-white hover:bg-red-600 shadow-lg shadow-red-500/25 transition-all"
                title="Stop generating"
              >
                <Square className="w-5 h-5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim()}
                className={cn(
                  'absolute right-2 bottom-2 p-2.5 rounded-xl transition-all',
                  input.trim()
                    ? 'bg-primary-600 text-white hover:bg-primary-700 shadow-lg shadow-primary-500/25'
                    : 'bg-surface-100 text-surface-400'
                )}
              >
                <Send className="w-5 h-5" />
              </button>
            )}
          </div>
          <p className="text-xs text-surface-400 text-center mt-3">
            Decision support based on NICE NG12. Not a substitute for clinical judgment.
          </p>
        </form>
      </div>
    </main>
  );
}

function EmptyState({
  onPromptClick,
  onCreate,
}: {
  onPromptClick: (prompt: string) => void;
  onCreate: () => void;
}) {
  return (
    <div className="h-full flex flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center max-w-2xl"
      >
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center mx-auto mb-6 shadow-xl shadow-primary-500/25">
          <Stethoscope className="w-8 h-8 text-white" />
        </div>
        
        <h2 className="font-display font-bold text-2xl text-surface-900 mb-3">
          Clinical Decision Support
        </h2>
        <p className="text-surface-600 mb-8 max-w-md mx-auto">
          Ask questions about suspected cancer recognition and referral pathways 
          based on NICE NG12 guideline.
        </p>
        
        <div className="grid sm:grid-cols-2 gap-3 max-w-xl mx-auto">
          {EXAMPLE_PROMPTS.map((prompt, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => {
                onCreate();
                onPromptClick(prompt.text);
              }}
              className="text-left p-4 rounded-xl border border-surface-200 bg-white hover:border-primary-300 hover:bg-primary-50 transition-all group"
            >
              <span className="inline-flex items-center gap-1.5 text-xs font-medium text-primary-600 mb-2">
                <Sparkles className="w-3 h-3" />
                {prompt.category}
              </span>
              <p className="text-sm text-surface-700 group-hover:text-surface-900">
                {prompt.text}
              </p>
            </motion.button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  const isTyping = message.isTyping;
  
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={cn('flex gap-4', isUser && 'flex-row-reverse')}
    >
      {/* Avatar */}
      <div
        className={cn(
          'w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0',
          isUser
            ? 'bg-surface-200'
            : 'bg-gradient-to-br from-primary-500 to-primary-600 shadow-lg shadow-primary-500/25'
        )}
      >
        {isUser ? (
          <span className="text-lg">👤</span>
        ) : (
          <Stethoscope className="w-5 h-5 text-white" />
        )}
      </div>
      
      {/* Message Content */}
      <div className={cn('flex-1 max-w-[80%]', isUser && 'flex flex-col items-end')}>
        {isTyping && !message.content ? (
          // Initial typing indicator (dots)
          <div className="bg-surface-100 rounded-2xl rounded-tl-md px-4 py-3">
            <div className="flex gap-1.5">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        ) : isTyping && message.content ? (
          // Streaming content with cursor
          <div className="bg-surface-100 rounded-2xl rounded-tl-md px-4 py-3 text-surface-800">
            <div
              className="prose prose-sm max-w-none prose-surface"
              dangerouslySetInnerHTML={{
                __html: formatMessage(message.content),
              }}
            />
            <span className="inline-block w-2 h-4 bg-primary-500 animate-pulse ml-0.5" />
          </div>
        ) : (
          <>
            <div
              className={cn(
                'rounded-2xl px-4 py-3',
                isUser
                  ? 'bg-primary-600 text-white rounded-tr-md'
                  : 'bg-surface-100 text-surface-800 rounded-tl-md'
              )}
            >
              {isUser ? (
                <p className="whitespace-pre-wrap">{message.content}</p>
              ) : (
                <div
                  className="prose prose-sm max-w-none prose-surface"
                  dangerouslySetInnerHTML={{
                    __html: formatMessage(message.content),
                  }}
                />
              )}
            </div>
            
            {/* Response type indicator */}
            {!isUser && message.response_type && (
              <ResponseTypeIndicator type={message.response_type} />
            )}
            
            {/* Citations */}
            {!isUser && message.citations && message.citations.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-2">
                {message.citations.map((citation, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-md bg-primary-100 text-primary-700 text-xs font-medium"
                  >
                    {citation.statement_id}: {citation.section}
                  </span>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}

function ResponseTypeIndicator({ type }: { type: ResponseType }) {
  const config = {
    answer: {
      icon: CheckCircle2,
      label: 'Grounded Answer',
      color: 'text-accent-600',
      bg: 'bg-accent-50',
    },
    clarification: {
      icon: HelpCircle,
      label: 'Needs More Info',
      color: 'text-amber-600',
      bg: 'bg-amber-50',
    },
    refusal: {
      icon: AlertCircle,
      label: 'Out of Scope',
      color: 'text-trust-600',
      bg: 'bg-trust-50',
    },
    error: {
      icon: XCircle,
      label: 'Error',
      color: 'text-red-600',
      bg: 'bg-red-50',
    },
  };
  
  const { icon: Icon, label, color, bg } = config[type] || config.answer;
  
  return (
    <div className={cn('inline-flex items-center gap-1.5 mt-2 px-2 py-1 rounded-md text-xs font-medium', color, bg)}>
      <Icon className="w-3.5 h-3.5" />
      {label}
    </div>
  );
}

/**
 * Format message content with markdown-like styling.
 */
function formatMessage(content: string): string {
  let formatted = content;
  
  // Convert citations
  formatted = parseCitations(formatted);
  
  // Convert **bold**
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  
  // Convert bullet points
  formatted = formatted.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
  formatted = formatted.replace(/(<li>.*<\/li>\n?)+/g, '<ul class="list-disc list-inside space-y-1 my-2">$&</ul>');
  
  // Convert numbered lists
  formatted = formatted.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
  
  // Convert line breaks
  formatted = formatted.replace(/\n\n/g, '</p><p class="mt-3">');
  formatted = formatted.replace(/\n/g, '<br/>');
  
  // Wrap in paragraph
  if (!formatted.startsWith('<')) {
    formatted = `<p>${formatted}</p>`;
  }
  
  return formatted;
}
