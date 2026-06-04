/**
 * Shared frontend utilities.
 *
 * Pure helpers with no I/O. Anything touching the network lives in `./api`.
 */

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { formatDistanceToNow, isToday, isYesterday, format } from 'date-fns';
import { v4 as uuidv4 } from 'uuid';

/**
 * Merge Tailwind class names safely. Later classes win over earlier ones
 * even when they resolve to the same Tailwind property.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Generate a stable, collision-resistant identifier for client-side entities
 * (conversations, messages, etc.). Uses RFC 4122 v4 UUIDs.
 */
export function generateId(): string {
  return uuidv4();
}

/**
 * Truncate a string to `maxLength` characters, appending an ellipsis when
 * truncation happens. Short inputs are returned unchanged.
 */
export function truncate(text: string, maxLength: number): string {
  if (maxLength <= 0) return '';
  if (text.length <= maxLength) return text;
  return text.slice(0, Math.max(0, maxLength - 1)).trimEnd() + '…';
}

/**
 * Format a timestamp for display in conversation lists.
 *
 * - Today  -> "3:14 PM"
 * - Yesterday -> "Yesterday"
 * - Within 7 days -> "3 days ago"
 * - Older -> "Jan 5, 2026"
 */
export function formatDate(date: Date | string | number): string {
  const d = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(d.getTime())) return '';

  if (isToday(d)) return format(d, 'h:mm a');
  if (isYesterday(d)) return 'Yesterday';

  const sevenDaysMs = 7 * 24 * 60 * 60 * 1000;
  if (Date.now() - d.getTime() < sevenDaysMs) {
    return formatDistanceToNow(d, { addSuffix: true });
  }

  return format(d, 'MMM d, yyyy');
}

/**
 * Wrap inline NICE NG12 citation references in styled spans so that the
 * markdown renderer (with `rehype-raw`) surfaces them as visual chips.
 *
 * Recognizes:
 *   - Bracketed: `[NG12 1.2.7]`, `[NG12: 1.2.7]`
 *   - Parenthesized: `(NG12 1.2.7)`
 *   - Bare references at end of clauses: `... NG12 1.2.7`
 *
 * The matcher is intentionally conservative to avoid mangling code blocks
 * or unrelated numeric tokens. Existing HTML / markdown structure is
 * preserved; only the citation tokens themselves are wrapped.
 */
export function parseCitations(text: string): string {
  if (!text) return text;

  // Tag class kept inline so callers don't need to import CSS-in-JS.
  const citationClass =
    'inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-xs font-medium ' +
    'bg-blue-50 text-blue-700 border border-blue-200';

  // Pattern matches NG12 followed by a dotted recommendation id like 1.2.7
  // Captures optional surrounding brackets / parens so we can replace them.
  const pattern = /([\[(])?\s*NG12[:\s]+(\d+(?:\.\d+){1,3})\s*([\])])?/g;

  return text.replace(pattern, (match, open: string | undefined, ref: string, close: string | undefined) => {
    // Only treat as a citation if brackets balance, or if it's a bare reference
    // (no opener at all). This avoids consuming partial syntax.
    const hasOpen = Boolean(open);
    const hasClose = Boolean(close);
    if (hasOpen !== hasClose) return match;

    return `<span class="${citationClass}" data-citation="NG12 ${ref}">NG12 ${ref}</span>`;
  });
}
