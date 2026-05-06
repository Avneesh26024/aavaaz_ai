// src/components/DebugPanel.tsx
// Collapsible debug overlay — shows connection states, message rates, errors.
// Only rendered when VITE_DEBUG=true in .env

import { useState } from 'react';
import styles from './DebugPanel.module.css';

export interface DebugStats {
  backendWsState: string;
  scribeState: string;
  framesPerMin: number;
  audioChunksPerMin: number;
  transcriptsCommitted: number;
  ttsChunksReceived: number;
  lastError: string | null;
  lastTranscript: string | null;
  lastResponsePreview: string | null;
}

interface Props {
  stats: DebugStats;
}

export function DebugPanel({ stats }: Props) {
  const [open, setOpen] = useState(true);

  if (import.meta.env.VITE_DEBUG !== 'true') return null;

  const row = (label: string, value: string | number | null, ok?: boolean) => (
    <div className={styles.row} key={label}>
      <span className={styles.label}>{label}</span>
      <span className={`${styles.value} ${ok === false ? styles.bad : ok === true ? styles.good : ''}`}>
        {value ?? '—'}
      </span>
    </div>
  );

  return (
    <div className={styles.panel}>
      <button className={styles.toggle} onClick={() => setOpen((o) => !o)}>
        🛠 Debug {open ? '▲' : '▼'}
      </button>
      {open && (
        <div className={styles.body}>
          <div className={styles.section}>Connections</div>
          {row('Backend WS', stats.backendWsState, stats.backendWsState === 'OPEN')}
          {row('Scribe WS', stats.scribeState, stats.scribeState === 'OPEN')}

          <div className={styles.section}>Throughput</div>
          {row('Video frames/min', stats.framesPerMin)}
          {row('Audio chunks/min', stats.audioChunksPerMin)}
          {row('Transcripts committed', stats.transcriptsCommitted)}
          {row('TTS chunks rx', stats.ttsChunksReceived)}

          <div className={styles.section}>Last Activity</div>
          {row('Last transcript', stats.lastTranscript ? `"${stats.lastTranscript.slice(0, 40)}…"` : null)}
          {row('Last response', stats.lastResponsePreview ? `"${stats.lastResponsePreview.slice(0, 40)}…"` : null)}

          {stats.lastError && (
            <>
              <div className={styles.section}>Error</div>
              {row('Last error', stats.lastError, false)}
            </>
          )}
        </div>
      )}
    </div>
  );
}
