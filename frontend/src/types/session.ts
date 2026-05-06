// src/types/session.ts

export type SessionStatus =
  | 'idle'
  | 'starting'
  | 'active'
  | 'ending'
  | 'ended'
  | 'error';

export interface TranscriptEntry {
  id: string;
  text: string;
  committed: boolean;
  timestamp: number;
}

export interface TherapistMessage {
  id: string;
  text: string;
  timestamp: number;
}

export interface BackendMessage {
  type: 'therapist_response' | 'error';
  text?: string;
  message?: string;
}
