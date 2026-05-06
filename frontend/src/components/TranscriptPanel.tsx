// src/components/TranscriptPanel.tsx
import { useEffect, useRef } from 'react';
import type { TranscriptEntry } from '../types/session';
import styles from './TranscriptPanel.module.css';

interface Props {
  entries: TranscriptEntry[];
  partialText: string;
}

export function TranscriptPanel({ entries, partialText }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries, partialText]);

  const hasContent = entries.length > 0 || partialText.trim().length > 0;

  return (
    <div className={styles.panel}>
      <span className={styles.label}>Live Transcript</span>
      <div className={styles.scrollArea}>
        {!hasContent && (
          <span className={styles.empty}>Transcript will appear here…</span>
        )}
        {entries.map((entry) => (
          <div
            key={entry.id}
            className={`${styles.entry} ${entry.committed ? styles.committed : styles.partial}`}
          >
            {entry.text}
          </div>
        ))}
        {partialText && (
          <div className={`${styles.entry} ${styles.partial}`}>{partialText}</div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
