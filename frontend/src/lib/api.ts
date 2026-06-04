/**
 * Frontend HTTP/SSE client for the Qualified Health backend.
 *
 * All requests target the FastAPI server proxied by Vite at `/api/v1/...`.
 * Network/parse failures surface as `ApiClientError` so callers can render
 * deterministic messages.
 */

import type {
  Artifact,
  Citation,
  ChatRequest,
  CompileRequest,
  CompileResponse,
  PathwaySpec,
  ResponseType,
} from '@/types';

const API_BASE = '/api/v1';

type ChatEndpoint = 'rag' | 'graphrag' | 'custom';

/**
 * Finalized response metadata emitted on the SSE `done` event.
 *
 * Shape mirrors the backend payload in `services/*_chat_service.py`.
 */
export interface StreamEvent {
  response_type: ResponseType;
  citations?: Citation[];
  artifacts?: Artifact[];
  processing_time_ms?: number;
  pathway_available?: boolean;
  pathway_spec?: PathwaySpec;
}

/**
 * Typed error raised by this module. Carries an HTTP status when available
 * so callers can branch on transport vs. application failures.
 */
export class ApiClientError extends Error {
  readonly status: number | undefined;

  constructor(message: string, status?: number) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
  }
}

/** Backend SSE envelope variants. */
type SseEnvelope =
  | { type: 'start'; conversation_id: string }
  | { type: 'chunk'; content: string }
  | { type: 'done'; response_type: ResponseType; citations?: Citation[]; artifacts?: Artifact[]; processing_time_ms?: number; pathway_available?: boolean; pathway_spec?: PathwaySpec }
  | { type: 'error'; message: string };

/**
 * POST a chat message and stream tokens back via Server-Sent Events.
 *
 * The backend emits four envelope types:
 *   - `start` — handshake; ignored by the UI
 *   - `chunk` — incremental token; forwarded to `onChunk`
 *   - `done`  — terminal metadata (citations, artifacts, timing)
 *   - `error` — terminal failure
 *
 * The promise resolves once the stream terminates (cleanly or otherwise).
 * Abort via the provided signal stops the underlying fetch and resolves
 * without invoking `onComplete` or `onError`.
 */
export async function sendChatMessageStream(
  request: ChatRequest,
  onChunk: (content: string) => void,
  onComplete: (event: StreamEvent) => void,
  onError: (message: string) => void,
  signal: AbortSignal,
  endpoint: ChatEndpoint,
): Promise<void> {
  const url = `${API_BASE}/chat/${endpoint}/stream`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(request),
      signal,
    });
  } catch (err) {
    if (signal.aborted) return;
    onError(networkErrorMessage(err));
    return;
  }

  if (!response.ok) {
    onError(await readHttpErrorMessage(response));
    return;
  }
  if (!response.body) {
    onError('Streaming response had no body.');
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line ("\n\n").
      let separatorIdx = buffer.indexOf('\n\n');
      while (separatorIdx !== -1) {
        const rawEvent = buffer.slice(0, separatorIdx);
        buffer = buffer.slice(separatorIdx + 2);

        const envelope = parseSseEvent(rawEvent);
        if (envelope) {
          completed = handleEnvelope(envelope, onChunk, onComplete, onError) || completed;
        }

        separatorIdx = buffer.indexOf('\n\n');
      }
    }

    // Flush any trailing event that wasn't followed by a blank line.
    const tail = buffer.trim();
    if (tail) {
      const envelope = parseSseEvent(tail);
      if (envelope) {
        completed = handleEnvelope(envelope, onChunk, onComplete, onError) || completed;
      }
    }

    if (!completed) {
      onError('Stream ended before completion.');
    }
  } catch (err) {
    if (signal.aborted) return;
    onError(networkErrorMessage(err));
  } finally {
    reader.releaseLock();
  }
}

/**
 * Compile an NG12 recommendation against patient criteria.
 *
 * Used by the pathway tool after a clinician fills out the criteria form.
 */
export async function compileRecommendation(
  request: CompileRequest,
): Promise<CompileResponse> {
  const response = await fetch(`${API_BASE}/chat/custom/compile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new ApiClientError(await readHttpErrorMessage(response), response.status);
  }

  return (await response.json()) as CompileResponse;
}

/**
 * Parse a single SSE event block into a typed envelope.
 * Returns `null` for unknown or malformed events; callers should skip them.
 */
function parseSseEvent(rawEvent: string): SseEnvelope | null {
  // SSE allows multiple `data:` lines per event; concatenate them.
  const dataLines: string[] = [];
  for (const line of rawEvent.split('\n')) {
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0) return null;

  const payload = dataLines.join('\n');
  try {
    const parsed = JSON.parse(payload) as unknown;
    if (!parsed || typeof parsed !== 'object' || !('type' in parsed)) return null;
    return parsed as SseEnvelope;
  } catch {
    return null;
  }
}

/**
 * Dispatch an envelope to the appropriate callback.
 * Returns true when the envelope represents stream completion (done|error).
 */
function handleEnvelope(
  envelope: SseEnvelope,
  onChunk: (content: string) => void,
  onComplete: (event: StreamEvent) => void,
  onError: (message: string) => void,
): boolean {
  switch (envelope.type) {
    case 'chunk':
      if (typeof envelope.content === 'string') onChunk(envelope.content);
      return false;
    case 'done':
      onComplete({
        response_type: envelope.response_type,
        citations: envelope.citations,
        artifacts: envelope.artifacts,
        processing_time_ms: envelope.processing_time_ms,
        pathway_available: envelope.pathway_available,
        pathway_spec: envelope.pathway_spec,
      });
      return true;
    case 'error':
      onError(envelope.message || 'Unknown server error.');
      return true;
    case 'start':
      return false;
    default:
      return false;
  }
}

/** Convert a non-2xx response into a human-readable error string. */
async function readHttpErrorMessage(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { message?: string; error?: string; detail?: string };
    return data.message ?? data.error ?? data.detail ?? `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status} ${response.statusText})`;
  }
}

/** Normalize fetch/stream exceptions into a user-facing message. */
function networkErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return 'Network error.';
}
