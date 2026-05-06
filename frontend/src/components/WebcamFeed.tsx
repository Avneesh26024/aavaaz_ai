// src/components/WebcamFeed.tsx
import { useEffect, useRef, useState } from 'react';
import type { SessionStatus } from '../types/session';
import styles from './WebcamFeed.module.css';

interface Props {
  status: SessionStatus;
  onVideoReady: (videoEl: HTMLVideoElement) => void;
  onStartSession: () => void;
  onEndSession: () => void;
}

const STATUS_LABELS: Record<SessionStatus, string> = {
  idle: 'Ready',
  starting: 'Connecting…',
  active: 'Session Active',
  ending: 'Ending…',
  ended: 'Session Ended',
  error: 'Error',
};

export function WebcamFeed({ status, onVideoReady, onStartSession, onEndSession }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);

  useEffect(() => {
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            setCameraReady(true);
            if (videoRef.current) onVideoReady(videoRef.current);
          };
        }
      } catch (err) {
        const msg = err instanceof DOMException && err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access and reload.'
          : 'Could not access camera. Check that no other app is using it.';
        setPermissionError(msg);
      }
    })();

    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const canStart = (['idle', 'ended', 'error'] as SessionStatus[]).includes(status);
  const canEnd = status === 'active';

  return (
    <div className={styles.container}>
      <div className={styles.videoWrapper}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className={styles.video}
        />
        {!cameraReady && !permissionError && (
          <div className={styles.permissionOverlay}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M15 10l4.553-2.069A1 1 0 0121 8.82v6.36a1 1 0 01-1.447.889L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
            </svg>
            Requesting camera…
          </div>
        )}
        {permissionError && (
          <div className={styles.permissionOverlay}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 8v4m0 4h.01" />
            </svg>
            <span className={styles.permissionError}>{permissionError}</span>
          </div>
        )}
      </div>

      <div className={styles.statusRow}>
        <span className={`${styles.statusDot} ${styles[status]}`} />
        <span className={styles.statusText}>{STATUS_LABELS[status]}</span>
      </div>

      {canStart && (
        <button
          className={`${styles.button} ${styles.startButton}`}
          onClick={onStartSession}
          disabled={!cameraReady || !!permissionError || status === 'starting'}
        >
          Start Session
        </button>
      )}
      {canEnd && (
        <button
          className={`${styles.button} ${styles.endButton}`}
          onClick={onEndSession}
          disabled={status !== 'active'}
        >
          End Session
        </button>
      )}
    </div>
  );
}
