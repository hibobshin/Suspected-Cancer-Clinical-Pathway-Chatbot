/**
 * PathwayTool Component
 * 
 * Interactive criteria selection UI for checking a patient against NG12 recommendations.
 * Displays the verbatim NG12 text and allows clinicians to input patient criteria.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  User,
  Stethoscope,
  CheckCircle2,
  AlertCircle,
  Loader2,
  ClipboardCheck,
  ChevronRight,
  Sparkles,
} from 'lucide-react';
import { compileRecommendation } from '@/lib/api';
import type { PathwaySpec, PatientCriteria, Criterion, CompileResponse } from '@/types';
import { cn } from '@/lib/utils';

/**
 * Format verbatim text to render HTML tags like <u> properly.
 */
function formatVerbatimText(text: string): string {
  let formatted = text;
  formatted = formatted.replace(/&lt;/g, '<');
  formatted = formatted.replace(/&gt;/g, '>');
  formatted = formatted.replace(/&amp;/g, '&');
  formatted = formatted.replace(/<u>/gi, '<span class="underline decoration-primary-400 decoration-2">');
  formatted = formatted.replace(/<\/u>/gi, '</span>');
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-surface-100">$1</strong>');
  formatted = formatted.replace(/^\s*[-•*]\s+(.+)$/gm, '<li class="ml-4">$1</li>');
  return formatted;
}

/**
 * Clean symptom text by removing HTML tags and trailing conjunctions for display.
 */
function cleanSymptomText(text: string): string {
  let cleaned = text;
  cleaned = cleaned.replace(/&lt;/g, '<');
  cleaned = cleaned.replace(/&gt;/g, '>');
  cleaned = cleaned.replace(/<\/?u>/gi, '');
  cleaned = cleaned.replace(/\s+(or|and|who|with|that)\s*:?\s*$/i, '');
  cleaned = cleaned.replace(/[.,;:]+$/, '');
  cleaned = cleaned.trim();
  return cleaned;
}

interface PathwayToolProps {
  spec: PathwaySpec;
  onSubmit: (response: CompileResponse) => void;
  onCancel: () => void;
}

const COMMON_SYMPTOMS = [
  'cough',
  'fatigue',
  'shortness of breath',
  'chest pain',
  'weight loss',
  'appetite loss',
  'haemoptysis',
  'hoarseness',
];

export function PathwayTool({ spec, onSubmit, onCancel }: PathwayToolProps) {
  const [criteria, setCriteria] = useState<PatientCriteria>({
    age: undefined,
    smoking: false,
    symptoms: [],
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fields = new Set<string>();
  spec.criteria_groups.forEach((group) => {
    group.criteria.forEach((c: Criterion) => {
      fields.add(c.field);
    });
  });

  const getAvailableSymptoms = (): string[] => {
    for (const group of spec.criteria_groups) {
      for (const c of group.criteria) {
        if (c.field === 'symptoms' && Array.isArray(c.value)) {
          return c.value as string[];
        }
      }
    }
    return COMMON_SYMPTOMS;
  };

  const availableSymptoms = getAvailableSymptoms();

  const handleSymptomToggle = (symptom: string) => {
    const currentSymptoms = criteria.symptoms || [];
    if (currentSymptoms.includes(symptom)) {
      setCriteria({
        ...criteria,
        symptoms: currentSymptoms.filter((s) => s !== symptom),
      });
    } else {
      setCriteria({
        ...criteria,
        symptoms: [...currentSymptoms, symptom],
      });
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await compileRecommendation({
        recommendation_id: spec.recommendation_id,
        patient_criteria: criteria,
      });
      onSubmit(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check criteria');
    } finally {
      setIsLoading(false);
    }
  };

  const hasRequiredFields = () => {
    if (fields.has('age') && !criteria.age) return false;
    return true;
  };

  const recIds = spec.recommendation_id.split(',').map(r => r.trim());
  const isMultiRec = recIds.length > 1;
  const verbatimLimit = isMultiRec ? 600 : 400;
  const selectedCount = (criteria.symptoms || []).length;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98, y: 8 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className="relative overflow-hidden bg-gradient-to-br from-surface-50 to-surface-100 border border-surface-200 rounded-2xl shadow-lg mt-4"
    >
      {/* Decorative gradient accent */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary-500 via-primary-400 to-primary-600" />
      
      {/* Header */}
      <div className="px-5 pt-5 pb-4 border-b border-surface-200 bg-white/50">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-primary-100 rounded-xl">
              <ClipboardCheck className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-surface-900">
                Check Patient Criteria
              </h3>
              <p className="text-sm text-surface-500 mt-0.5">
                Verify against NG12 recommendations
              </p>
            </div>
          </div>
          <button
            onClick={onCancel}
            className="p-2 hover:bg-surface-100 rounded-lg transition-colors"
          >
            <X className="w-4 h-4 text-surface-400" />
          </button>
        </div>
        
        {/* Recommendation badges */}
        <div className="flex flex-wrap gap-2 mt-4">
          {recIds.map((id) => (
            <span
              key={id}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary-50 border border-primary-200 rounded-full text-xs font-medium text-primary-700"
            >
              <Sparkles className="w-3 h-3" />
              NG12 {id}
            </span>
          ))}
        </div>
      </div>

      {/* Verbatim NG12 Text */}
      <div className="px-5 py-4 bg-surface-800 text-surface-200">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-1 h-4 bg-primary-500 rounded-full" />
          <span className="text-xs font-medium text-surface-400 uppercase tracking-wide">
            Guideline Text
          </span>
        </div>
        <div 
          className="text-sm leading-relaxed max-h-40 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-surface-600 scrollbar-track-surface-700"
          dangerouslySetInnerHTML={{ 
            __html: formatVerbatimText(spec.verbatim_text.slice(0, verbatimLimit) + (spec.verbatim_text.length > verbatimLimit ? '...' : ''))
          }}
        />
      </div>

      {/* Form Section */}
      <div className="px-5 py-5 space-y-5">
        {/* Age Field */}
        {fields.has('age') && (
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-surface-700">
              <User className="w-4 h-4 text-surface-400" />
              Patient Age
            </label>
            <div className="relative">
              <input
                type="number"
                min={0}
                max={120}
                value={criteria.age || ''}
                onChange={(e) =>
                  setCriteria({
                    ...criteria,
                    age: e.target.value ? parseInt(e.target.value) : undefined,
                  })
                }
                className="w-full bg-white border border-surface-300 rounded-xl px-4 py-3 text-surface-900 placeholder-surface-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-shadow"
                placeholder="Enter age in years"
              />
              {criteria.age && (
                <div className="absolute right-3 top-1/2 -translate-y-1/2">
                  <CheckCircle2 className="w-5 h-5 text-green-500" />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Symptoms */}
        {fields.has('symptoms') && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 text-sm font-medium text-surface-700">
                <Stethoscope className="w-4 h-4 text-surface-400" />
                Presenting Symptoms
              </label>
              {selectedCount > 0 && (
                <span className="text-xs font-medium text-primary-600 bg-primary-50 px-2.5 py-1 rounded-full">
                  {selectedCount} selected
                </span>
              )}
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {availableSymptoms.map((symptom, index) => {
                const isSelected = (criteria.symptoms || []).includes(symptom);
                const cleanedText = cleanSymptomText(symptom);
                
                return (
                  <motion.button
                    key={symptom}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03 }}
                    onClick={() => handleSymptomToggle(symptom)}
                    className={cn(
                      'group relative flex items-center gap-3 px-4 py-3 rounded-xl border text-left transition-all duration-200',
                      isSelected
                        ? 'bg-primary-50 border-primary-300 shadow-sm'
                        : 'bg-white border-surface-200 hover:border-surface-300 hover:shadow-sm'
                    )}
                  >
                    <div className={cn(
                      'flex-shrink-0 w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors',
                      isSelected
                        ? 'bg-primary-500 border-primary-500'
                        : 'border-surface-300 group-hover:border-surface-400'
                    )}>
                      {isSelected && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        >
                          <CheckCircle2 className="w-3 h-3 text-white" />
                        </motion.div>
                      )}
                    </div>
                    <span className={cn(
                      'text-sm leading-tight',
                      isSelected ? 'text-primary-800 font-medium' : 'text-surface-700'
                    )}>
                      {cleanedText}
                    </span>
                  </motion.button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Error Message */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mx-5 mb-4 p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700 flex items-center gap-3"
          >
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
            <span>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Actions */}
      <div className="px-5 pb-5 pt-2 flex gap-3">
        <button
          onClick={onCancel}
          className="flex-1 px-4 py-3 text-sm font-medium bg-surface-100 hover:bg-surface-200 text-surface-700 rounded-xl transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={isLoading || !hasRequiredFields()}
          className={cn(
            'flex-1 px-4 py-3 text-sm font-medium rounded-xl transition-all flex items-center justify-center gap-2',
            isLoading || !hasRequiredFields()
              ? 'bg-surface-200 text-surface-400 cursor-not-allowed'
              : 'bg-primary-600 hover:bg-primary-700 text-white shadow-md hover:shadow-lg'
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Checking...
            </>
          ) : (
            <>
              Check Criteria
              <ChevronRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </motion.div>
  );
}
