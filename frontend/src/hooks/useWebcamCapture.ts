// src/hooks/useWebcamCapture.ts
import { useCallback, useRef } from 'react';

export interface WebcamCaptureApi {
  startFrameCapture: (videoEl: HTMLVideoElement, onFrame: (base64: string) => void) => void;
  stopFrameCapture: () => void;
}

const FRAME_INTERVAL_MS = 2500;
const JPEG_QUALITY = 0.7;

export function useWebcamCapture(): WebcamCaptureApi {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const startFrameCapture = useCallback(
    (videoEl: HTMLVideoElement, onFrame: (base64: string) => void) => {
      if (intervalRef.current) return;

      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
      }

      intervalRef.current = setInterval(() => {
        if (videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

        const canvas = canvasRef.current!;
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);
        // Strip the data URL prefix to get pure base64
        const base64 = dataUrl.replace(/^data:image\/jpeg;base64,/, '');
        onFrame(base64);
      }, FRAME_INTERVAL_MS);
    },
    [],
  );

  const stopFrameCapture = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  return { startFrameCapture, stopFrameCapture };
}
