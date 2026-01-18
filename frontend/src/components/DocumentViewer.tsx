import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, FileText, Loader2, ChevronRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { cn } from '@/lib/utils';

interface DocumentViewerProps {
  isOpen: boolean;
  onClose: () => void;
  ruleId?: string;
  sectionPath?: string;
  solutionMode?: 'graphrag' | 'rag' | 'custom';
  scrollRequest?: { target: string; id: number } | null;
}

interface SectionMatch {
  id: string;
  title: string;
  startIndex: number;
  endIndex: number;
  level: number;
  content: string;
}

interface RuleMatch {
  id: string;
  content: string;
  sectionId: string;
}

export function DocumentViewer({ 
  isOpen, 
  onClose, 
  ruleId, 
  sectionPath,
  solutionMode = 'custom',
  scrollRequest,
}: DocumentViewerProps) {
  const [documentContent, setDocumentContent] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sections, setSections] = useState<SectionMatch[]>([]);
  const [, setRules] = useState<RuleMatch[]>([]);
  const [activeSection, setActiveSection] = useState<string | null>(null);
  const [activeRuleId, setActiveRuleId] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const sectionRefs = useRef<Map<string, HTMLElement>>(new Map());
  const ruleRefs = useRef<Map<string, HTMLElement>>(new Map());

  // Load document when opened
  useEffect(() => {
    if (isOpen && solutionMode === 'custom') {
      fetchDocument();
    }
  }, [isOpen, solutionMode]);

  // Parse sections and rules from markdown
  useEffect(() => {
    if (documentContent) {
      const parsedSections = parseMarkdownSections(documentContent);
      setSections(parsedSections);
      
      // Parse rule IDs from content (e.g., "1.1.1 Refer people...")
      const parsedRules = parseRuleIds(documentContent);
      setRules(parsedRules);
    }
  }, [documentContent]);
  
  const parseRuleIds = (markdown: string): RuleMatch[] => {
    const rules: RuleMatch[] = [];
    // Match patterns like "1.1.1 Refer..." or "1.3.5 Consider..."
    const rulePattern = /^(\d+\.\d+(?:\.\d+)?)\s+(.+)/gm;
    let match;
    
    while ((match = rulePattern.exec(markdown)) !== null) {
      rules.push({
        id: match[1],
        content: match[2].substring(0, 100),
        sectionId: '',
      });
    }
    
    return rules;
  };

  // Track the last processed scroll request ID
  const lastScrollIdRef = useRef<number>(0);
  const pendingScrollRef = useRef<string | null>(null);

  // Handle scroll requests - the id changes even for same target, ensuring re-trigger
  useEffect(() => {
    if (!scrollRequest || scrollRequest.id === lastScrollIdRef.current) {
      return;
    }
    
    lastScrollIdRef.current = scrollRequest.id;
    const target = scrollRequest.target;
    
    if (sections.length > 0) {
      // Document loaded, scroll immediately
      scrollToSection(target);
    } else {
      // Document not loaded yet, queue the scroll
      pendingScrollRef.current = target;
    }
  }, [scrollRequest, sections.length]);

  // Handle pending scroll after sections load
  useEffect(() => {
    if (sections.length > 0 && pendingScrollRef.current) {
      const target = pendingScrollRef.current;
      pendingScrollRef.current = null;
      // Small delay to ensure refs are registered after render
      setTimeout(() => scrollToSection(target), 100);
    }
  }, [sections.length]);

  // Auto-scroll to section based on ruleId or sectionPath
  useEffect(() => {
    if (isOpen && sections.length > 0 && (ruleId || sectionPath)) {
      const targetSection = ruleId || sectionPath;
      if (targetSection) {
        setTimeout(() => scrollToSection(targetSection), 300);
      }
    }
  }, [isOpen, sections, ruleId, sectionPath]);

  const fetchDocument = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch final.md from the backend
      const response = await fetch('/api/v1/document/final');
      if (!response.ok) {
        throw new Error('Failed to fetch document');
      }
      
      const text = await response.text();
      setDocumentContent(text);
    } catch (err) {
      console.error('Error fetching document:', err);
      setError(err instanceof Error ? err.message : 'Failed to load document');
    } finally {
      setLoading(false);
    }
  };

  const parseMarkdownSections = (markdown: string): SectionMatch[] => {
    const sections: SectionMatch[] = [];
    const lines = markdown.split('\n');
    let currentIndex = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
      
      if (headingMatch) {
        const level = headingMatch[1].length;
        const title = headingMatch[2].trim();
        const id = generateSectionId(title);
        
        // Find the end of this section (next heading of same or higher level)
        let endIndex = currentIndex + line.length + 1;
        for (let j = i + 1; j < lines.length; j++) {
          const nextLine = lines[j];
          const nextHeadingMatch = nextLine.match(/^(#{1,6})\s+/);
          if (nextHeadingMatch && nextHeadingMatch[1].length <= level) {
            break;
          }
          endIndex += nextLine.length + 1;
        }
        
        sections.push({
          id,
          title,
          level,
          startIndex: currentIndex,
          endIndex,
          content: markdown.slice(currentIndex, endIndex),
        });
      }
      
      currentIndex += line.length + 1;
    }
    
    return sections;
  };

  const generateSectionId = (title: string): string => {
    return title
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/--+/g, '-')
      .trim();
  };

  const scrollToSection = (sectionIdentifier: string, retryCount = 0) => {
    // First, check if this is a rule ID (e.g., "1.1.1")
    const isRuleId = /^\d+\.\d+(\.\d+)?$/.test(sectionIdentifier);
    
    if (isRuleId) {
      // Try to find and scroll to the rule
      const ruleElement = ruleRefs.current.get(sectionIdentifier);
      if (ruleElement) {
        setActiveRuleId(sectionIdentifier);
        setActiveSection(null);
        
        // Scroll immediately
        ruleElement.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center',
        });
        
        // Highlight persists until next click (no auto-clear)
        return;
      } else if (retryCount < 5) {
        // Element not yet rendered, retry after delay
        setTimeout(() => scrollToSection(sectionIdentifier, retryCount + 1), 150);
        return;
      }
    }
    
    // Try to find section by ID, title, or partial match
    const section = sections.find(s => 
      s.id === sectionIdentifier ||
      s.id === sectionIdentifier.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-') ||
      s.title.toLowerCase().includes(sectionIdentifier.toLowerCase()) ||
      sectionIdentifier.toLowerCase().includes(s.title.toLowerCase())
    );

    if (section) {
      const element = sectionRefs.current.get(section.id);
      if (element) {
        setActiveSection(section.id);
        setActiveRuleId(null);
        
        // Scroll immediately
        element.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center',
        });
        
        // Highlight persists until next click (no auto-clear)
      }
    }
  };

  const registerSectionRef = (id: string, element: HTMLElement | null) => {
    if (element) {
      sectionRefs.current.set(id, element);
    } else {
      sectionRefs.current.delete(id);
    }
  };
  
  const registerRuleRef = (ruleId: string, element: HTMLElement | null) => {
    if (element) {
      ruleRefs.current.set(ruleId, element);
    } else {
      ruleRefs.current.delete(ruleId);
    }
  };
  
  // Extract rule ID from text if present (e.g., "1.1.1 Refer people..." -> "1.1.1")
  const extractRuleId = (text: string): string | null => {
    const match = text.match(/^(\d+\.\d+(?:\.\d+)?)\s/);
    return match ? match[1] : null;
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      {isOpen && (
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 30, stiffness: 300 }}
          className="fixed right-0 top-0 h-full w-[600px] bg-white shadow-2xl border-l border-surface-200 z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-surface-200 bg-gradient-to-r from-surface-50 to-white">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
                  <FileText className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="text-lg font-display font-bold text-surface-900">
                    NICE NG12 Guideline
                  </h2>
                    <p className="text-xs text-surface-500">
                    Suspected cancer: recognition and referral
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-surface-100 transition-colors group text-sm font-medium text-surface-700 hover:text-surface-900"
                aria-label="Close document viewer"
              >
                <span>Close</span>
                <X className="w-4 h-4" />
              </button>
            </div>
            
            {/* Content */}
            <div 
              ref={contentRef}
              className="flex-1 overflow-y-auto px-6 py-4 scroll-smooth"
            >
              {loading && (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <Loader2 className="w-8 h-8 text-primary-600 animate-spin mx-auto mb-3" />
                    <p className="text-sm text-surface-600 font-medium">Loading document...</p>
                  </div>
                </div>
              )}
              
              {error && (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center max-w-md">
                    <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-3">
                      <X className="w-6 h-6 text-red-600" />
                    </div>
                    <p className="text-sm text-red-600 font-medium">{error}</p>
                    <button
                      onClick={fetchDocument}
                      className="mt-4 px-4 py-2 rounded-lg bg-primary-600 text-white text-sm font-medium hover:bg-primary-700 transition-colors"
                    >
                      Retry
                    </button>
                  </div>
                </div>
              )}
              
              {!loading && !error && documentContent && (
                <div className="w-full">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw]}
                    components={{
                      h1: ({ ...props }) => {
                        const id = generateSectionId(String(props.children));
                        const isHighlighted = activeSection === id;
                        return (
                          <h1
                            ref={(el) => registerSectionRef(id, el)}
                            id={id}
                            className={cn(
                              'text-2xl font-display font-bold text-surface-900 mt-6 mb-3 pb-2 border-b-2 border-surface-200 transition-all duration-300',
                              isHighlighted && 'bg-primary-100/50 -mx-3 px-3 rounded-lg border-primary-300'
                            )}
                            {...props}
                          />
                        );
                      },
                      h2: ({ ...props }) => {
                        const id = generateSectionId(String(props.children));
                        const isHighlighted = activeSection === id;
                        return (
                          <h2
                            ref={(el) => registerSectionRef(id, el)}
                            id={id}
                            className={cn(
                              'text-xl font-display font-bold text-surface-900 mt-5 mb-2 transition-all duration-300',
                              isHighlighted && 'bg-primary-100/50 -mx-3 px-3 py-2 rounded-lg'
                            )}
                            {...props}
                          />
                        );
                      },
                      h3: ({ ...props }) => {
                        const id = generateSectionId(String(props.children));
                        const isHighlighted = activeSection === id;
                        return (
                          <h3
                            ref={(el) => registerSectionRef(id, el)}
                            id={id}
                            className={cn(
                              'text-lg font-display font-semibold text-surface-800 mt-4 mb-2 transition-all duration-300',
                              isHighlighted && 'bg-primary-100/50 -mx-3 px-3 py-2 rounded-lg'
                            )}
                            {...props}
                          />
                        );
                      },
                      h4: ({ ...props }) => {
                        const id = generateSectionId(String(props.children));
                        const isHighlighted = activeSection === id;
                        return (
                          <h4
                            ref={(el) => registerSectionRef(id, el)}
                            id={id}
                            className={cn(
                              'text-base font-semibold text-surface-800 mt-3 mb-2 transition-all duration-300',
                              isHighlighted && 'bg-primary-100/50 -mx-3 px-3 py-1.5 rounded-lg'
                            )}
                            {...props}
                          />
                        );
                      },
                      h5: ({ ...props }) => {
                        const id = generateSectionId(String(props.children));
                        const isHighlighted = activeSection === id;
                        return (
                          <h5
                            ref={(el) => registerSectionRef(id, el)}
                            id={id}
                            className={cn(
                              'text-sm font-semibold text-surface-700 mt-3 mb-1.5 transition-all duration-300',
                              isHighlighted && 'bg-primary-100/50 -mx-3 px-3 py-1.5 rounded-lg'
                            )}
                            {...props}
                          />
                        );
                      },
                      h6: ({ ...props }) => {
                        const id = generateSectionId(String(props.children));
                        const isHighlighted = activeSection === id;
                        return (
                          <h6
                            ref={(el) => registerSectionRef(id, el)}
                            id={id}
                            className={cn(
                              'text-xs font-semibold text-surface-700 mt-2 mb-1.5 transition-all duration-300',
                              isHighlighted && 'bg-primary-100/50 -mx-3 px-3 py-1.5 rounded-lg'
                            )}
                            {...props}
                          />
                        );
                      },
                      p: ({ children, ...props }) => {
                        const text = String(children);
                        const ruleId = extractRuleId(text);
                        const isHighlighted = ruleId && activeRuleId === ruleId;
                        
                        if (ruleId) {
                          return (
                            <p 
                              ref={(el) => registerRuleRef(ruleId, el)}
                              data-rule-id={ruleId}
                              className={cn(
                                'text-sm text-surface-700 leading-relaxed mb-3 transition-all duration-300',
                                isHighlighted && 'bg-yellow-100 -mx-3 px-3 py-2 rounded-lg border-l-4 border-primary-500 shadow-sm'
                              )}
                              {...props}
                            >
                              {children}
                            </p>
                          );
                        }
                        
                        return (
                          <p className="text-sm text-surface-700 leading-relaxed mb-3" {...props} />
                        );
                      },
                      ul: ({ ...props }) => (
                        <ul className="list-disc list-inside space-y-1 mb-3 text-sm text-surface-700" {...props} />
                      ),
                      ol: ({ ...props }) => (
                        <ol className="list-decimal list-inside space-y-1 mb-3 text-sm text-surface-700" {...props} />
                      ),
                      li: ({ ...props }) => (
                        <li className="ml-3" {...props} />
                      ),
                      blockquote: ({ ...props }) => (
                        <blockquote className="border-l-4 border-primary-300 pl-4 py-2 my-4 italic text-surface-600 bg-surface-50 rounded-r-lg" {...props} />
                      ),
                      code: ({ inline, ...props }: any) => 
                        inline ? (
                          <code className="px-1.5 py-0.5 rounded bg-surface-100 text-primary-600 text-sm font-mono" {...props} />
                        ) : (
                          <code className="block px-4 py-3 rounded-lg bg-surface-900 text-surface-100 text-sm font-mono overflow-x-auto mb-4" {...props} />
                        ),
                      table: ({ ...props }) => (
                        <div className="overflow-x-auto my-4">
                          <table className="min-w-full divide-y divide-surface-300 border border-surface-300 rounded-lg text-xs" {...props} />
                        </div>
                      ),
                      thead: ({ ...props }) => (
                        <thead className="bg-surface-100" {...props} />
                      ),
                      tbody: ({ ...props }) => (
                        <tbody className="divide-y divide-surface-200 bg-white" {...props} />
                      ),
                      tr: ({ ...props }) => (
                        <tr className="hover:bg-surface-50 transition-colors" {...props} />
                      ),
                      th: ({ ...props }) => (
                        <th className="px-3 py-2 text-left text-xs font-semibold text-surface-900" {...props} />
                      ),
                      td: ({ ...props }) => (
                        <td className="px-3 py-2 text-xs text-surface-700" {...props} />
                      ),
                      a: ({ ...props }) => (
                        <a className="text-primary-600 hover:text-primary-700 underline transition-colors" target="_blank" rel="noopener noreferrer" {...props} />
                      ),
                      hr: ({ ...props }) => (
                        <hr className="my-8 border-surface-300" {...props} />
                      ),
                      strong: ({ ...props }) => (
                        <strong className="font-semibold text-surface-900" {...props} />
                      ),
                      em: ({ ...props }) => (
                        <em className="italic text-surface-800" {...props} />
                      ),
                    }}
                  >
                    {documentContent}
                  </ReactMarkdown>
                </div>
              )}
            </div>

            {/* Footer with breadcrumb if section or rule is active */}
            {(activeSection || activeRuleId) && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 20 }}
                className="px-6 py-3 border-t border-surface-200 bg-primary-50"
              >
                <div className="flex items-center gap-2 text-sm text-primary-700">
                  <ChevronRight className="w-4 h-4" />
                  <span className="font-medium">
                    {activeRuleId 
                      ? `Viewing Rule: ${activeRuleId}`
                      : `Viewing: ${sections.find(s => s.id === activeSection)?.title}`
                    }
                  </span>
                </div>
              </motion.div>
            )}
          </motion.div>
      )}
    </AnimatePresence>
  );
}
