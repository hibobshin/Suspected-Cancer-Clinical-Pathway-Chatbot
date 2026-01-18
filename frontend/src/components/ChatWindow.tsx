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
  Database,
  Brain,
  FileText,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Eye,
  EyeOff,
} from 'lucide-react';
import { useChatStore, selectActiveConversation, type SolutionMode } from '@/stores/chatStore';
import { cn, parseCitations } from '@/lib/utils';
import type { Artifact, ChatMessage, ResponseType, CompileResponse } from '@/types';
import { DocumentViewer } from './DocumentViewer';
import { PathwayTool } from './PathwayTool';

const EXAMPLE_PROMPTS = {
  graphrag: [
    {
      text: "What entities relate to colorectal cancer referral?",
      category: "Knowledge Graph",
    },
    {
      text: "Show me connections between FIT testing and diagnosis",
      category: "Graph Query",
    },
    {
      text: "What does the graph say about 2WW pathways?",
      category: "Relationships",
    },
    {
      text: "Find cancer symptoms that trigger urgent referral",
      category: "Entity Search",
    },
  ],
  rag: [
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
  ],
  custom: [
    {
      text: "Explain the general approach to suspected cancer referrals",
      category: "General",
    },
    {
      text: "What are the key principles of cancer pathway management?",
      category: "Principles",
    },
  ],
};

export function ChatWindow() {
  const activeConversation = useChatStore(selectActiveConversation);
  const { 
    sendMessage, 
    isLoading, 
    createConversation, 
    stopGeneration,
    solutionMode,
    showArtifacts,
    setShowArtifacts,
  } = useChatStore();
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [documentViewerOpen, setDocumentViewerOpen] = useState(false);
  const [selectedRuleId, setSelectedRuleId] = useState<string | undefined>();
  const [selectedSectionPath, setSelectedSectionPath] = useState<string | undefined>();
  // Used for hover effects on artifact buttons (optional visual feedback)
  const [, setHoveredSection] = useState<string | null>(null);
  
  const messages = activeConversation?.messages ?? [];
  
  // Auto-open document viewer for custom mode when artifacts are present
  useEffect(() => {
    if (solutionMode === 'custom' && messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'assistant' && lastMessage.artifacts && lastMessage.artifacts.length > 0 && !lastMessage.isTyping) {
        setDocumentViewerOpen(true);
      }
    }
  }, [messages, solutionMode]);
  
  // Track scroll request with a counter to force re-triggers
  const [scrollRequest, setScrollRequest] = useState<{ target: string; id: number } | null>(null);
  
  const handleArtifactClick = (artifact: Artifact) => {
    // Only open document viewer in custom mode
    if (solutionMode === 'custom') {
      const sectionIdentifier = artifact.rule_id || artifact.section;
      
      // Set URL parameters for scrolling
      if (artifact.rule_id) {
        setSelectedRuleId(artifact.rule_id);
        setSelectedSectionPath(undefined);
      } else if (artifact.section) {
        setSelectedRuleId(undefined);
        setSelectedSectionPath(artifact.section);
      }
      
      // Open document viewer
      setDocumentViewerOpen(true);
      
      // Set scroll request with unique ID to force re-trigger even for same section
      if (sectionIdentifier) {
        setScrollRequest({ target: sectionIdentifier, id: Date.now() });
      }
    }
  };
  
  // Handle pathway compilation result
  const handlePathwaySubmit = (response: CompileResponse) => {
    // Add the compiled recommendation as a new assistant message
    if (activeConversation) {
      const newMessage: ChatMessage = {
        id: `pathway-${Date.now()}`,
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        response_type: response.meets_criteria ? 'answer' : 'clarification',
        artifacts: response.artifacts,
      };
      
      // Use the store's addMessage if available, or just update locally
      useChatStore.getState().addPathwayResponse(newMessage);
    }
  };
  
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
    <main 
      className="flex-1 flex flex-col h-full bg-white transition-all duration-300"
      style={{
        marginRight: solutionMode === 'custom' && documentViewerOpen ? '600px' : '0'
      }}
    >
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState 
            onPromptClick={handleExampleClick} 
            onCreate={createConversation}
            solutionMode={solutionMode}
          />
        ) : (
          <div className="max-w-3xl mx-auto py-8 px-4 space-y-6">
            <AnimatePresence mode="popLayout">
              {messages.map((message) => (
                <MessageBubble 
                  key={message.id} 
                  message={message} 
                  showArtifacts={showArtifacts}
                  onArtifactClick={handleArtifactClick}
                  onArtifactHover={setHoveredSection}
                  solutionMode={solutionMode}
                  onPathwaySubmit={handlePathwaySubmit}
                />
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {/* Input Area */}
      <div className="border-t border-surface-200 bg-surface-50 p-4">
        <div className="max-w-3xl mx-auto">
          {/* Artifacts Toggle - Show for RAG and Custom modes */}
          {(solutionMode === 'rag' || solutionMode === 'custom') && (
            <div className="flex items-center justify-center gap-2 mb-3">
              <span className="text-xs text-surface-500">Artifacts:</span>
              <button
                type="button"
                onClick={() => setShowArtifacts(!showArtifacts)}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                  showArtifacts
                    ? 'bg-primary-100 text-primary-700 hover:bg-primary-200'
                    : 'bg-surface-200 text-surface-600 hover:bg-surface-300'
                )}
                title={showArtifacts ? 'Hide source artifacts' : 'Show source artifacts'}
              >
                {showArtifacts ? (
                  <>
                    <Eye className="w-3.5 h-3.5" />
                    <span>Show</span>
                  </>
                ) : (
                  <>
                    <EyeOff className="w-3.5 h-3.5" />
                    <span>Hide</span>
                  </>
                )}
              </button>
            </div>
          )}
          
          <form onSubmit={handleSubmit}>
            <div className="relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  solutionMode === 'graphrag'
                    ? 'Query the knowledge graph...'
                    : solutionMode === 'rag'
                    ? 'Ask about cancer pathways...'
                    : 'Ask a question...'
                }
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
              {solutionMode === 'graphrag' 
                ? 'Using ArangoDB GraphRAG retriever for context-aware responses.'
                : solutionMode === 'rag'
                ? 'Decision support based on NICE NG12 with RAG. Not a substitute for clinical judgment.'
                : 'Custom implementation. Not a substitute for clinical judgment.'}
            </p>
          </form>
        </div>
      </div>
      
      {/* Document Viewer Side Panel - Only for custom mode */}
      {solutionMode === 'custom' && (
        <DocumentViewer
          isOpen={documentViewerOpen}
          onClose={() => {
            setDocumentViewerOpen(false);
            setSelectedRuleId(undefined);
            setSelectedSectionPath(undefined);
            setScrollRequest(null);
          }}
          ruleId={selectedRuleId}
          sectionPath={selectedSectionPath}
          solutionMode={solutionMode}
          scrollRequest={scrollRequest}
        />
      )}
    </main>
  );
}

function EmptyState({
  onPromptClick,
  onCreate,
  solutionMode,
}: {
  onPromptClick: (prompt: string) => void;
  onCreate: () => void;
  solutionMode: SolutionMode;
}) {
  const prompts = EXAMPLE_PROMPTS[solutionMode] || EXAMPLE_PROMPTS.rag;
  const isGraphRAG = solutionMode === 'graphrag';
  const isRAG = solutionMode === 'rag';
  
  return (
    <div className="h-full flex flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center max-w-2xl"
      >
        <div className={cn(
          "w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl",
          isGraphRAG 
            ? "bg-gradient-to-br from-violet-500 to-violet-600 shadow-violet-500/25"
            : isRAG
            ? "bg-gradient-to-br from-primary-500 to-primary-600 shadow-primary-500/25"
            : "bg-gradient-to-br from-blue-500 to-blue-600 shadow-blue-500/25"
        )}>
          {isGraphRAG ? (
            <Database className="w-8 h-8 text-white" />
          ) : isRAG ? (
            <Stethoscope className="w-8 h-8 text-white" />
          ) : (
            <Brain className="w-8 h-8 text-white" />
          )}
        </div>
        
        <h2 className="font-display font-bold text-2xl text-surface-900 mb-3">
          {isGraphRAG ? 'Knowledge Graph Assistant' : isRAG ? 'RAG Clinical Decision Support' : 'Custom Assistant'}
        </h2>
        <p className="text-surface-600 mb-8 max-w-md mx-auto">
          {isGraphRAG 
            ? 'Query the clinical knowledge graph for context-rich answers powered by ArangoDB GraphRAG.'
            : isRAG
            ? 'Ask questions about suspected cancer recognition and referral pathways based on NICE NG12 guideline with RAG retrieval.'
            : 'Custom implementation for flexible healthcare assistance.'}
        </p>
        
        <div className="grid sm:grid-cols-2 gap-3 max-w-xl mx-auto">
          {prompts.map((prompt, index) => (
            <motion.button
              key={`${solutionMode}-${index}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => {
                onCreate();
                onPromptClick(prompt.text);
              }}
              className="text-left p-4 rounded-xl border border-surface-200 bg-white hover:border-primary-300 hover:bg-primary-50 transition-all group"
            >
              <span className={cn(
                "inline-flex items-center gap-1.5 text-xs font-medium mb-2",
                isGraphRAG ? "text-violet-600" : isRAG ? "text-primary-600" : "text-blue-600"
              )}>
                {isGraphRAG ? <Database className="w-3 h-3" /> : isRAG ? <Sparkles className="w-3 h-3" /> : <Brain className="w-3 h-3" />}
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

function MessageBubble({ 
  message, 
  showArtifacts,
  onArtifactClick,
  onArtifactHover,
  solutionMode,
  onPathwaySubmit,
}: { 
  message: ChatMessage; 
  showArtifacts: boolean;
  onArtifactClick?: (artifact: Artifact) => void;
  onArtifactHover?: (section: string | null) => void;
  solutionMode?: 'graphrag' | 'rag' | 'custom';
  onPathwaySubmit?: (response: CompileResponse) => void;
}) {
  const [showPathwayTool, setShowPathwayTool] = useState(false);
  
  const handleCitationClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    
    // Handle citation links
    if (target.classList.contains('citation-link')) {
      const ruleId = target.getAttribute('data-rule-id');
      const section = target.getAttribute('data-section');
      if (ruleId && onArtifactClick) {
        onArtifactClick({
          section: section || `NG12 > ${ruleId}`,
          text: '',
          source: 'NICE NG12',
          source_url: 'https://www.nice.org.uk/guidance/ng12',
          relevance_score: 0,
          rule_id: ruleId,
        });
      }
    }
    
    // Handle NG12 badge clicks
    if (target.classList.contains('ng12-badge')) {
      const ruleId = target.getAttribute('data-rule-id');
      if (ruleId && onArtifactClick) {
        onArtifactClick({
          section: `NG12 ${ruleId}`,
          text: '',
          source: 'NICE NG12',
          source_url: 'https://www.nice.org.uk/guidance/ng12',
          relevance_score: 0,
          rule_id: ruleId,
        });
      }
    }
  };
  
  const handlePathwaySubmit = (response: CompileResponse) => {
    setShowPathwayTool(false);
    if (onPathwaySubmit) {
      onPathwaySubmit(response);
    }
  };
  
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
              onClick={handleCitationClick}
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
                  onClick={handleCitationClick}
                />
              )}
            </div>
            
            {/* Response type indicator */}
            {!isUser && message.response_type && (
              <ResponseTypeIndicator type={message.response_type} />
            )}
            
            {/* Inline Chunks for Custom Mode - Show after response indicator */}
            {!isUser && !isTyping && solutionMode === 'custom' && message.artifacts && message.artifacts.length > 0 && onArtifactClick && (
              <div className="mt-2 flex flex-wrap gap-2 items-center">
                <span className="text-xs text-surface-500 font-medium">Evidence:</span>
                {message.artifacts.map((artifact, index) => (
                  <button
                    key={`inline-artifact-${index}`}
                    onClick={() => {
                      onArtifactClick(artifact);
                      onArtifactHover?.(artifact.section || artifact.rule_id || null);
                    }}
                    onMouseEnter={() => onArtifactHover?.(artifact.section || artifact.rule_id || null)}
                    onMouseLeave={() => onArtifactHover?.(null)}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-primary-50 hover:bg-primary-100 border border-primary-200 hover:border-primary-300 text-primary-700 text-xs font-medium transition-all hover:shadow-sm cursor-pointer"
                    title={`${artifact.section}${artifact.rule_id ? ` [${artifact.rule_id}]` : ''} - Click to view in document`}
                  >
                    <FileText className="w-3 h-3" />
                    <span className="max-w-[200px] truncate">{artifact.section}</span>
                    {artifact.rule_id && (
                      <span className="text-[10px] px-1 py-0.5 rounded bg-primary-100 text-primary-600">
                        {artifact.rule_id}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
            
            {/* Citations - only show if no inline artifacts are shown (to avoid duplicates) */}
            {!isUser && message.citations && message.citations.length > 0 && 
             !(solutionMode === 'custom' && message.artifacts && message.artifacts.length > 0) && (
              <div className="mt-2 flex flex-wrap gap-2">
                {message.citations.map((citation, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      if (onArtifactClick) {
                        // Create artifact-like object for the click handler
                        onArtifactClick({
                          section: citation.section,
                          text: citation.text,
                          source: 'NICE NG12',
                          source_url: 'https://www.nice.org.uk/guidance/ng12',
                          relevance_score: 1.0,
                          rule_id: citation.statement_id,
                        } as Artifact);
                      }
                    }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-primary-50 hover:bg-primary-100 border border-primary-200 hover:border-primary-300 text-primary-700 text-xs font-medium transition-all hover:shadow-sm cursor-pointer"
                    title={`View ${citation.statement_id} in document`}
                  >
                    <FileText className="w-3 h-3" />
                    {citation.statement_id}: {citation.section}
                  </button>
                ))}
              </div>
            )}
            
            {/* Artifacts - Show when toggle is on (for RAG and Custom modes) */}
            <AnimatePresence mode="wait">
              {!isUser && showArtifacts && message.artifacts && message.artifacts.length > 0 && (
                <motion.div
                  key="artifacts"
                  initial={{ opacity: 0, height: 0, marginTop: 0 }}
                  animate={{ opacity: 1, height: 'auto', marginTop: 16 }}
                  exit={{ opacity: 0, height: 0, marginTop: 0 }}
                  transition={{ duration: 0.3, ease: 'easeInOut' }}
                  style={{ overflow: 'hidden' }}
                >
                  <ArtifactsDisplay 
                    artifacts={message.artifacts} 
                    onArtifactClick={onArtifactClick}
                    onArtifactHover={onArtifactHover}
                  />
                </motion.div>
              )}
            </AnimatePresence>
            
            {/* Pathway Button - Show when pathway is available */}
            {!isUser && !isTyping && message.pathway_available && message.pathway_spec && (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-3"
              >
                {!showPathwayTool ? (
                  <button
                    onClick={() => setShowPathwayTool(true)}
                    className="inline-flex items-center gap-2 px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors shadow-sm"
                  >
                    <Stethoscope className="w-4 h-4" />
                    Check a patient against these criteria
                  </button>
                ) : (
                  <PathwayTool
                    spec={message.pathway_spec}
                    onSubmit={handlePathwaySubmit}
                    onCancel={() => setShowPathwayTool(false)}
                  />
                )}
              </motion.div>
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
 * Handles HTML tags (<u>), markdown headers (###), and preserves structure.
 */
function formatMessage(content: string): string {
  let formatted = content;
  
  // First, unescape any HTML entities (in case content was escaped)
  formatted = formatted.replace(/&lt;/g, '<');
  formatted = formatted.replace(/&gt;/g, '>');
  formatted = formatted.replace(/&amp;/g, '&');
  
  // Pre-process: Combine lines where text is followed by orphaned NG12 reference
  // e.g., "Refer for stomach cancer\nNG12 1.2.7" → "Refer for stomach cancer NG12 1.2.7"
  formatted = formatted.replace(/^([^\n*•-][^\n]+[a-zA-Z])\n(NG12\s+[\d.]+)\s*$/gm, '$1 $2');
  
  // Convert citations
  formatted = parseCitations(formatted);
  
  // Handle <u> tags - convert to proper underline spans
  formatted = formatted.replace(/<u>/gi, '<span class="underline">');
  formatted = formatted.replace(/<\/u>/gi, '</span>');
  
  // Convert horizontal rules FIRST (before headers consume ---)
  formatted = formatted.replace(/^[-*]{3,}\s*$/gm, '<hr class="my-4 border-surface-300" />');
  
  // Handle "For assessment of X" followed by NG12 reference on next line
  // Combine them into a single styled header
  formatted = formatted.replace(
    /^(For (?:assessment of |evaluation of )?[^\n]+)\n(NG12\s+[\d.]+)\s*$/gm,
    '<div class="flex items-center gap-2 mt-5 mb-3 pb-2 border-b border-surface-200"><h3 class="font-bold text-surface-900 text-base">$1</h3><span class="px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700">$2</span></div>'
  );
  
  // Handle "Summary for X:" as a styled header
  formatted = formatted.replace(
    /^(Summary for [^:\n]+):?\s*$/gm,
    '<div class="font-bold text-surface-900 mt-5 mb-2 pb-1 border-b border-surface-200">$1</div>'
  );
  
  // Convert markdown headers (### Header) to styled headers
  formatted = formatted.replace(/^###\s+(\d+\.)\s+(.+?)\s+(NG12\s+[\d.]+)?\s*$/gm, 
    (_match, num, title, ref) => {
      const refSpan = ref ? `<span class="ml-2 text-xs font-medium text-primary-600 bg-primary-50 px-2 py-0.5 rounded">${ref}</span>` : '';
      return `<h3 class="font-bold text-surface-900 text-base mt-5 mb-2 pb-1 border-b border-surface-200"><span class="text-primary-600 mr-1">${num}</span>${title}${refSpan}</h3>`;
    });
  formatted = formatted.replace(/^###\s+(.+)$/gm, 
    '<h3 class="font-bold text-surface-900 text-base mt-5 mb-2 pb-1 border-b border-surface-200">$1</h3>');
  formatted = formatted.replace(/^##\s+(.+)$/gm, 
    '<h2 class="font-bold text-surface-900 mt-5 mb-3 text-lg">$1</h2>');
  
  // Convert NG12 references on their own line to clickable badges (cleanup orphaned refs)
  formatted = formatted.replace(/^\s*NG12\s+([\d.]+)\s*$/gm, 
    '<span class="ng12-badge inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700 mb-2 cursor-pointer hover:bg-primary-200 transition-colors" data-rule-id="$1">NG12 $1</span>');
  
  // Handle field labels on their own lines (Recommendation:, Criteria:, Action:)
  // and convert following indented/short lines as nested bullets
  formatted = formatted.replace(
    /^(Recommendation|Criteria|Action):\s*([^\n]*(?:with:|following:)?)\n((?:[^\n*•-][^\n]{0,60}\n?)+)/gm,
    (_match, label, intro, items) => {
      // Split items by newline and convert to nested list
      const itemLines = items.trim().split('\n').filter((line: string) => line.trim());
      if (itemLines.length > 0 && intro.trim().endsWith(':')) {
        // These are nested items
        const nestedItems = itemLines.map((item: string) => 
          `<li class="ml-6 text-surface-600">${item.trim()}</li>`
        ).join('');
        return `<div class="mb-2"><span class="font-semibold text-surface-800">${label}:</span> ${intro}<ul class="list-disc list-outside ml-8 mt-1 space-y-0.5">${nestedItems}</ul></div>`;
      } else {
        // Not nested, just format normally
        return `<div class="mb-2"><span class="font-semibold text-surface-800">${label}:</span> ${intro} ${items.trim().replace(/\n/g, ' ')}</div>`;
      }
    }
  );
  
  // Handle standalone field labels (Recommendation:, Criteria:, Action:) 
  formatted = formatted.replace(
    /^(Recommendation|Criteria|Action):\s*(.+)$/gm,
    '<div class="mb-2"><span class="font-semibold text-surface-800">$1:</span> $2</div>'
  );
  
  // Handle inline field labels within bullet points
  formatted = formatted.replace(
    /^(\s*)[-•*]\s+(Recommendation|Criteria|Action):\s*(.+)$/gm,
    '$1<li class="ml-4 text-surface-700"><span class="font-semibold text-surface-800">$2:</span> $3</li>'
  );
  
  // Handle regular bullet points
  formatted = formatted.replace(/^(\s*)[-•*]\s+(.+)$/gm, '$1<li class="ml-4 text-surface-700">$2</li>');
  
  // Wrap consecutive li elements in ul
  formatted = formatted.replace(/(<li[^>]*>.*?<\/li>\s*)+/gs, (match) => {
    return `<ul class="list-disc list-outside ml-4 space-y-1.5 my-2">${match}</ul>`;
  });
  
  // Convert **bold** text
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-surface-900">$1</strong>');
  
  // Convert *italic* (but not **bold**)
  formatted = formatted.replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, '<em class="text-surface-600">$1</em>');
  
  // Convert remaining NG12 references inline to clickable badges
  formatted = formatted.replace(/\bNG12\s+(\d+\.\d+(?:\.\d+)?)\b/g, 
    '<span class="ng12-badge inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700 cursor-pointer hover:bg-primary-200 transition-colors" data-rule-id="$1">NG12 $1</span>');
  
  // Process paragraphs - split by double newlines
  const sections = formatted.split(/\n\n+/);
  const formattedSections = sections.map((section) => {
    section = section.trim();
    if (!section) return '';
    
    // Skip if already wrapped in a block element
    if (/^<(h[1-6]|div|hr|ul|ol|li|span)/.test(section)) {
      return section;
    }
    
    // Convert single newlines to <br/> within sections
    let sectionContent = section.replace(/\n/g, '<br/>');
    
    // Wrap in paragraph
    return `<p class="mb-3 text-surface-800 leading-relaxed">${sectionContent}</p>`;
  });
  
  formatted = formattedSections.filter(s => s).join('');
  
  // Wrap in container
  formatted = `<div class="prose-sm max-w-none">${formatted}</div>`;
  
  return formatted;
}

/**
 * Format artifact text to properly render HTML tags like <u>.
 * Converts raw HTML tags to styled spans.
 */
function formatArtifactText(text: string): string {
  let formatted = text;
  
  // Unescape HTML entities first
  formatted = formatted.replace(/&lt;/g, '<');
  formatted = formatted.replace(/&gt;/g, '>');
  formatted = formatted.replace(/&amp;/g, '&');
  
  // Convert <u> tags to underline spans
  formatted = formatted.replace(/<u>/gi, '<span class="underline decoration-primary-500">');
  formatted = formatted.replace(/<\/u>/gi, '</span>');
  
  // Convert **bold** markdown
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold">$1</strong>');
  
  // Convert *italic* (but not **bold**)
  formatted = formatted.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, '<em>$1</em>');
  
  // Convert bullet points
  formatted = formatted.replace(/^\s*[-•*]\s+(.+)$/gm, '<span class="block ml-3">• $1</span>');
  
  // Convert NG12 references to clickable badges
  formatted = formatted.replace(/\bNG12\s+(\d+\.\d+(?:\.\d+)?)\b/g, 
    '<span class="ng12-badge inline-flex px-1 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700 cursor-pointer hover:bg-primary-200 transition-colors" data-rule-id="$1">NG12 $1</span>');
  
  return formatted;
}

function ArtifactsDisplay({ 
  artifacts,
  onArtifactClick,
  onArtifactHover,
}: { 
  artifacts: Artifact[];
  onArtifactClick?: (artifact: Artifact) => void;
  onArtifactHover?: (section: string | null) => void;
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  
  const handleArtifactHover = (artifact: Artifact | null) => {
    if (artifact) {
      // Use section title for hover highlighting
      const sectionId = artifact.section || artifact.rule_id;
      onArtifactHover?.(sectionId || null);
    } else {
      onArtifactHover?.(null);
    }
  };
  
  return (
    <motion.div 
      className="space-y-2"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2, delay: 0.1 }}
    >
      <motion.div 
        className="flex items-center gap-2 text-xs font-medium text-surface-600 mb-2"
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.2 }}
      >
        <FileText className="w-4 h-4" />
        <span>Source Artifacts ({artifacts.length})</span>
      </motion.div>
      
      <AnimatePresence>
        {artifacts.map((artifact, index) => {
        const isExpanded = expandedIndex === index;
        const previewLength = 150;
        const needsTruncation = artifact.text.length > previewLength;
        const displayText = isExpanded 
          ? artifact.text 
          : (needsTruncation ? artifact.text.slice(0, previewLength) + '...' : artifact.text);
        
        return (
          <motion.div
            key={`artifact-${index}`}
            initial={{ opacity: 0, y: 10, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.98 }}
            transition={{ 
              duration: 0.3, 
              delay: index * 0.05,
              ease: 'easeOut'
            }}
            layout
            className="border border-surface-200 rounded-lg bg-surface-50 overflow-hidden hover:border-primary-300 hover:shadow-md transition-all duration-200"
            onMouseEnter={() => handleArtifactHover(artifact)}
            onMouseLeave={() => handleArtifactHover(null)}
          >
            <div className="p-3">
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <button
                      onClick={() => onArtifactClick?.(artifact)}
                      className="text-sm font-semibold text-surface-900 hover:text-primary-600 transition-colors cursor-pointer text-left flex items-center gap-1"
                      title={artifact.rule_id ? `View section ${artifact.rule_id} in document` : 'View section in document'}
                    >
                      {artifact.section}
                      {artifact.rule_id && (
                        <span className="text-xs text-primary-600 font-normal">
                          ({artifact.rule_id})
                        </span>
                      )}
                      <FileText className="w-3.5 h-3.5 text-primary-600" />
                    </button>
                    <a
                      href={artifact.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary-600 hover:text-primary-700"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-surface-500">
                    <span>{artifact.source}</span>
                    {artifact.relevance_score > 0 && (
                      <span className="px-1.5 py-0.5 rounded bg-primary-100 text-primary-700">
                        Relevance: {artifact.relevance_score.toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>
                {needsTruncation && (
                  <button
                    onClick={() => setExpandedIndex(isExpanded ? null : index)}
                    className="flex-shrink-0 p-1 rounded hover:bg-surface-200 transition-colors"
                    aria-label={isExpanded ? 'Collapse' : 'Expand'}
                  >
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4 text-surface-600" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-surface-600" />
                    )}
                  </button>
                )}
              </div>
              
                <AnimatePresence mode="wait">
                <motion.div
                  key={isExpanded ? 'expanded' : 'collapsed'}
                  initial={{ opacity: 0, y: -5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 5 }}
                  transition={{ 
                    duration: 0.2, 
                    ease: 'easeInOut',
                  }}
                  className="text-sm text-surface-700 whitespace-pre-wrap leading-relaxed"
                  dangerouslySetInnerHTML={{ __html: formatArtifactText(displayText) }}
                />
              </AnimatePresence>
            </div>
          </motion.div>
        );
      })}
      </AnimatePresence>
    </motion.div>
  );
}
