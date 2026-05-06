import { useEffect, useRef } from 'react';
import styles from './ConversationFeed.module.css';

type Role = 'user' | 'assistant';

export interface ConversationTurn {
  id: string;
  role: Role;
  text: string;
  timestamp: number;
}

interface Props {
  turns: ConversationTurn[];
  partialText: string;
  speakingId: string | null;
  canReplay: (utteranceId: string) => boolean;
  onReplay: (utteranceId: string) => void;
}

export function ConversationFeed({ turns, partialText, speakingId, canReplay, onReplay }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [turns, partialText, speakingId]);

  const hasContent = turns.length > 0 || partialText.trim().length > 0;

  return (
    <div className={styles.panel}>
      <span className={styles.label}>Conversation</span>
      <div className={styles.scrollArea}>
        {!hasContent && (
          <div className={styles.floatingHint}>
            Say hello to start the conversation
          </div>
        )}

        {turns.map((turn) => {
          const isAssistant = turn.role === 'assistant';
          const isSpeaking = speakingId === turn.id;
          return (
            <div key={turn.id} className={`${styles.turn} ${styles[turn.role]}`}>
              <div className={styles.bubbleRow}>
                <div className={styles.bubble}>
                  {isAssistant && (
                    <div className={styles.bubbleHeader}>
                      <span className={styles.roleLabel}>Therapist</span>
                      <button
                        className={styles.replayButton}
                        type="button"
                        onClick={() => onReplay(turn.id)}
                        disabled={!canReplay(turn.id) || isSpeaking}
                      >
                        Replay
                      </button>
                    </div>
                  )}
                  {!isAssistant && (
                    <span className={styles.roleLabel}>You</span>
                  )}
                  <p className={styles.text}>{turn.text}</p>
                  {isAssistant && isSpeaking && (
                    <div className={styles.speakingRow}>
                      <div className={styles.waveform}>
                        {[0, 1, 2, 3, 4].map((i) => (
                          <div key={i} className={styles.bar} />
                        ))}
                      </div>
                      <span className={styles.speakingLabel}>Speaking</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}

        {partialText.trim().length > 0 && (
          <div className={`${styles.turn} ${styles.user}`}>
            <div className={styles.bubbleRow}>
              <div className={`${styles.bubble} ${styles.partial}`}>
                <span className={styles.roleLabel}>You</span>
                <p className={styles.text}>{partialText}</p>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
