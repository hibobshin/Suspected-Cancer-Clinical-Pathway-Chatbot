import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, FileText } from 'lucide-react';

interface DocumentViewerProps {
  isOpen: boolean;
  onClose: () => void;
  ruleId?: string;
  sectionPath?: string;
}

interface DocumentData {
  document: string;
  highlight_rule_id: string | null;
  highlight_section: string | null;
  highlight_start: number | null;
  highlight_end: number | null;
}

export function DocumentViewer({ isOpen, onClose, ruleId, sectionPath }: DocumentViewerProps) {
  const [documentData, setDocumentData] = useState<DocumentData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const highlightRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && (ruleId || sectionPath)) {
      fetchDocument();
    }
  }, [isOpen, ruleId, sectionPath]);

  const fetchDocument = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      if (ruleId) params.append('rule_id', ruleId);
      if (sectionPath) params.append('section_path', sectionPath);
      
      const response = await fetch(`/api/v1/document/section?${params.toString()}`);
      if (!response.ok) {
        throw new Error('Failed to fetch document');
      }
      
      const data: DocumentData = await response.json();
      setDocumentData(data);
      
      // Scroll to highlight after a brief delay
      setTimeout(() => {
        if (highlightRef.current) {
          highlightRef.current.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
        }
      }, 300);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load document');
    } finally {
      setLoading(false);
    }
  };

  const renderDocument = () => {
    if (!documentData) return null;
    
    const { document, highlight_start, highlight_end } = documentData;
    
    if (highlight_start !== null && highlight_end !== null) {
      // Split document into parts: before, highlighted, after
      const before = document.slice(0, highlight_start);
      const highlighted = document.slice(highlight_start, highlight_end);
      const after = document.slice(highlight_end);
      
      return (
        <div className="prose prose-sm max-w-none">
          <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
            {before}
            <mark
              ref={highlightRef}
              className="bg-yellow-200 text-yellow-900 px-1 rounded font-semibold"
            >
              {highlighted}
            </mark>
            {after}
          </pre>
        </div>
      );
    }
    
    // No highlighting - show full document
    return (
      <div className="prose prose-sm max-w-none">
        <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
          {document}
        </pre>
      </div>
    );
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/20 z-40"
          />
          
          {/* Side Panel */}
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 h-full w-full max-w-2xl bg-white shadow-2xl z-50 flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-surface-200 bg-surface-50">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary-600" />
                <div>
                  <h2 className="text-lg font-semibold text-surface-900">
                    NICE NG12 Guideline
                  </h2>
                  {ruleId && (
                    <p className="text-xs text-surface-500">
                      Section: {ruleId}
                    </p>
                  )}
                  {sectionPath && !ruleId && (
                    <p className="text-xs text-surface-500">
                      {sectionPath}
                    </p>
                  )}
                </div>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-surface-200 transition-colors"
                aria-label="Close document viewer"
              >
                <X className="w-5 h-5 text-surface-600" />
              </button>
            </div>
            
            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {loading && (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
                    <p className="text-sm text-surface-500">Loading document...</p>
                  </div>
                </div>
              )}
              
              {error && (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center text-red-600">
                    <p className="text-sm">{error}</p>
                  </div>
                </div>
              )}
              
              {!loading && !error && documentData && (
                <div className="max-w-4xl mx-auto">
                  {renderDocument()}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
