/**
 * Type definitions for the Qualified Health frontend.
 * All types are explicit and documented for clarity.
 */

export type MessageRole = 'user' | 'assistant' | 'system';

export type ResponseType = 'answer' | 'clarification' | 'refusal' | 'error';

export type PathwayRouteType = 
  | 'cancer_recognition'
  | 'symptom_triage' 
  | 'referral_guidance'
  | 'graph_rag'
  | 'custom';

export interface PathwayRoute {
  route_type: PathwayRouteType;
  name: string;
  description: string;
  welcome_message: string;
  example_prompts: string[];
}

export interface Citation {
  statement_id: string;
  section: string;
  text?: string;
}

export interface Artifact {
  section: string;
  text: string;
  source: string;
  source_url: string;
  relevance_score: number;
  chunk_id?: string;
  rule_id?: string;
  start_line?: number;
  end_line?: number;
}

export interface Criterion {
  field: string;
  operator: string;
  value: unknown;
  label: string;
}

export interface CriteriaGroup {
  operator: 'AND' | 'OR';
  criteria: Criterion[];
}

export interface PathwaySpec {
  recommendation_id: string;
  title: string;
  verbatim_text: string;
  criteria_groups: CriteriaGroup[];
  action_if_met: string;
}

export interface PatientCriteria {
  age?: number;
  sex?: 'male' | 'female';
  smoking?: boolean;
  symptoms?: string[];
  [key: string]: unknown;
}

export interface CompileRequest {
  recommendation_id: string;
  patient_criteria: PatientCriteria;
}

export interface CompileResponse {
  response: string;
  meets_criteria: boolean;
  matched_recommendation: string;
  artifacts: Artifact[];
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  response_type?: ResponseType;
  citations?: Citation[];
  artifacts?: Artifact[];
  isTyping?: boolean;
  pathway_available?: boolean;
  pathway_spec?: PathwaySpec;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ChatRequest {
  message: string;
  route_type?: PathwayRouteType;
  conversation_id?: string;
  context?: {
    conversation_id: string;
    messages: Array<{
      role: MessageRole;
      content: string;
      timestamp: string;
    }>;
  };
}

export interface ChatResponse {
  conversation_id: string;
  message: string;
  response_type: ResponseType;
  citations: Citation[];
  artifacts?: Artifact[];
  follow_up_questions: string[];
  processing_time_ms: number;
  timestamp: string;
  pathway_available?: boolean;
  pathway_spec?: PathwaySpec;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  environment: string;
  timestamp: string;
  checks: Record<string, boolean>;
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
  request_id?: string;
}
