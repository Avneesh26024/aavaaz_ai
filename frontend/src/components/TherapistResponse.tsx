// src/components/TherapistResponse.tsx
import styles from './TherapistResponse.module.css';

interface Props {
  text: string | null;
  transcriptText: string | null;
  transcriptDraft: string | null;
  isSpeaking: boolean;
  errorMessage: string | null;
  canReplay: boolean;
  onReplay: () => void;
}

export function TherapistResponse({
  text,
  transcriptText,
  transcriptDraft,
  isSpeaking,
  errorMessage,
  canReplay,
  onReplay,
}: Props) {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.label}>Therapist</span>
        <button
          className={styles.replayButton}
          type="button"
          onClick={onReplay}
          disabled={!canReplay || isSpeaking}
          aria-label="Replay last response"
        >
          Replay
        </button>
      </div>
      <div className={`${styles.card} ${isSpeaking ? styles.speaking : ''}`}>
        <div className={styles.transcriptBlock}>
          <span className={styles.subLabel}>User Transcript</span>
          {transcriptText ? (
            <p className={styles.transcriptText}>{transcriptText}</p>
          ) : (
            <p className={styles.transcriptEmpty}>Awaiting transcript…</p>
          )}
          {transcriptDraft && (
            <p className={styles.transcriptDraft}>{transcriptDraft}</p>
          )}
        </div>

        {isSpeaking && (
          <div className={styles.speakingIndicator}>
            <div className={styles.waveform}>
              {[0, 1, 2, 3, 4].map((i) => (
                <div key={i} className={styles.bar} />
              ))}
            </div>
            <span className={styles.speakingLabel}>Speaking</span>
          </div>
        )}

        {text ? (
          <p className={styles.responseText}>{text}</p>
        ) : (
          <p className={styles.placeholder}>
            {isSpeaking ? '' : 'Therapist response will appear here…'}
          </p>
        )}

        {errorMessage && (
          <div className={styles.errorText}>⚠ {errorMessage}</div>
        )}
      </div>
    </div>
  );
}
