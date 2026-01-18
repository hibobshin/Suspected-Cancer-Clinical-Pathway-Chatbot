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
  Cigarette,
  Stethoscope,
  CheckCircle2,
  AlertCircle,
  Loader2,
  FileText,
} from 'lucide-react';
import { compileRecommendation } from '@/lib/api';
import type { PathwaySpec, PatientCriteria, Criterion, CompileResponse } from '@/types';
import { cn } from '@/lib/utils';

/**
 * Format verbatim text to render HTML tags like <u> properly.
 */
function formatVerbatimText(text: string): string {
  let formatted = text;
  // Unescape HTML entities
  formatted = formatted.replace(/&lt;/g, '<');
  formatted = formatted.replace(/&gt;/g, '>');
  formatted = formatted.replace(/&amp;/g, '&');
  // Convert <u> tags to underline spans
  formatted = formatted.replace(/<u>/gi, '<span class="underline decoration-blue-400">');
  formatted = formatted.replace(/<\/u>/gi, '</span>');
  // Convert **bold** markdown
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-slate-100">$1</strong>');
  // Convert bullet points
  formatted = formatted.replace(/^\s*[-•*]\s+(.+)$/gm, '<span class="block ml-2">• $1</span>');
  return formatted;
}

/**
 * Clean symptom text by removing HTML tags and trailing conjunctions for display.
 */
function cleanSymptomText(text: string): string {
  let cleaned = text;
  // Unescape HTML entities
  cleaned = cleaned.replace(/&lt;/g, '<');
  cleaned = cleaned.replace(/&gt;/g, '>');
  // Remove <u> tags but keep the content
  cleaned = cleaned.replace(/<\/?u>/gi, '');
  // Remove trailing "or", "and", "who:", "and:", etc.
  cleaned = cleaned.replace(/\s+(or|and|who|with|that)\s*:?\s*$/i, '');
  // Remove trailing punctuation
  cleaned = cleaned.replace(/[.,;:]+$/, '');
  // Trim whitespace
  cleaned = cleaned.trim();
  return cleaned;
}

interface PathwayToolProps {
  spec: PathwaySpec;
  onSubmit: (response: CompileResponse) => void;
  onCancel: () => void;
}

// Common symptoms from NG12 recommendations
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

  // Extract unique fields from criteria groups
  const fields = new Set<string>();
  spec.criteria_groups.forEach((group) => {
    group.criteria.forEach((c: Criterion) => {
      fields.add(c.field);
    });
  });

  // Get available symptoms from criteria
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

  // Check if this is a multi-recommendation check
  const recIds = spec.recommendation_id.split(',').map(r => r.trim());
  const isMultiRec = recIds.length > 1;
  
  // For multi-rec, show more verbatim text
  const verbatimLimit = isMultiRec ? 600 : 300;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 10 }}
      className="bg-slate-800/50 border border-slate-700 rounded-lg p-4 mt-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-400" />
          <h3 className="text-sm font-semibold text-slate-200">{spec.title}</h3>
        </div>
        <button
          onClick={onCancel}
          className="p-1 hover:bg-slate-700 rounded transition-colors"
        >
          <X className="w-4 h-4 text-slate-400" />
        </button>
      </div>
      
      {/* Multiple Recommendations Badge */}
      {isMultiRec && (
        <div className="mb-3 flex flex-wrap gap-2">
          <span className="text-xs text-slate-400">Checking recommendations:</span>
          {recIds.map((id) => (
            <span
              key={id}
              className="px-2 py-0.5 bg-blue-600/20 border border-blue-500/30 rounded text-xs text-blue-300 font-medium"
            >
              NG12 {id}
            </span>
          ))}
        </div>
      )}

      {/* Verbatim NG12 Text */}
      <div 
        className="mb-4 p-3 bg-slate-900/50 border-l-2 border-blue-500 rounded text-sm text-slate-300 italic max-h-48 overflow-y-auto"
        dangerouslySetInnerHTML={{ 
          __html: formatVerbatimText(spec.verbatim_text.slice(0, verbatimLimit) + (spec.verbatim_text.length > verbatimLimit ? '...' : ''))
        }}
      />

      {/* Criteria Form */}
      <div className="space-y-4">
        {/* Age Field */}
        {fields.has('age') && (
          <div>
            <label className="flex items-center gap-2 text-sm text-slate-300 mb-2">
              <User className="w-4 h-4" />
              Patient Age
            </label>
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
              className="w-full bg-slate-900/50 border border-slate-600 rounded px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
              placeholder="Enter patient age"
            />
          </div>
        )}

        {/* Smoking Status */}
        {fields.has('smoking') && (
          <div>
            <label className="flex items-center gap-2 text-sm text-slate-300 mb-2">
              <Cigarette className="w-4 h-4" />
              Smoking History
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setCriteria({ ...criteria, smoking: true })}
                className={cn(
                  'flex-1 px-3 py-2 text-sm rounded border transition-colors',
                  criteria.smoking
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-slate-900/50 border-slate-600 text-slate-300 hover:border-slate-500'
                )}
              >
                Ever smoked
              </button>
              <button
                onClick={() => setCriteria({ ...criteria, smoking: false })}
                className={cn(
                  'flex-1 px-3 py-2 text-sm rounded border transition-colors',
                  !criteria.smoking
                    ? 'bg-slate-700 border-slate-600 text-white'
                    : 'bg-slate-900/50 border-slate-600 text-slate-300 hover:border-slate-500'
                )}
              >
                Never smoked
              </button>
            </div>
          </div>
        )}

        {/* Symptoms */}
        {fields.has('symptoms') && (
          <div>
            <label className="flex items-center gap-2 text-sm text-slate-300 mb-2">
              <Stethoscope className="w-4 h-4" />
              Unexplained Symptoms
            </label>
            <div className="flex flex-wrap gap-2">
              {availableSymptoms.map((symptom) => (
                <button
                  key={symptom}
                  onClick={() => handleSymptomToggle(symptom)}
                  className={cn(
                    'px-3 py-1.5 text-xs rounded-full border transition-colors',
                    (criteria.symptoms || []).includes(symptom)
                      ? 'bg-green-600/20 border-green-500 text-green-300'
                      : 'bg-slate-900/50 border-slate-600 text-slate-400 hover:border-slate-500'
                  )}
                >
                  {cleanSymptomText(symptom)}
                </button>
              ))}
            </div>
            {(criteria.symptoms || []).length > 0 && (
              <p className="mt-2 text-xs text-slate-400">
                Selected: {(criteria.symptoms || []).map(s => cleanSymptomText(s)).join(', ')}
              </p>
            )}
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
            className="mt-4 p-3 bg-red-900/20 border border-red-700 rounded text-sm text-red-300 flex items-center gap-2"
          >
            <AlertCircle className="w-4 h-4" />
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Actions */}
      <div className="flex gap-3 mt-6">
        <button
          onClick={onCancel}
          className="flex-1 px-4 py-2 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={isLoading || !hasRequiredFields()}
          className={cn(
            'flex-1 px-4 py-2 text-sm rounded transition-colors flex items-center justify-center gap-2',
            isLoading || !hasRequiredFields()
              ? 'bg-slate-600 text-slate-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Checking...
            </>
          ) : (
            <>
              <CheckCircle2 className="w-4 h-4" />
              Check Against NG12
            </>
          )}
        </button>
      </div>
    </motion.div>
  );
}
