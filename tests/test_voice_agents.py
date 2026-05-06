import asyncio
import os
import sys
import wave
from typing import Optional


def _ensure_backend_on_path() -> None:
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	backend_root = os.path.join(repo_root, "backend")
	if backend_root not in sys.path:
		sys.path.insert(0, backend_root)


async def _print_emotions(engine, interval_seconds: int, stop_event: asyncio.Event) -> None:
	while not stop_event.is_set():
		await asyncio.sleep(interval_seconds)
		result = await engine.handle_transcript_commit("live mic")
		print("\n[Emotion window]")
		print(result)


async def _stream_wav(engine, wav_path: str, frames_per_buffer: int, rate: int) -> None:
	with wave.open(wav_path, "rb") as wav_file:
		wav_rate = wav_file.getframerate()
		wav_channels = wav_file.getnchannels()
		wav_width = wav_file.getsampwidth()

		if wav_rate != rate or wav_channels != 2 or wav_width != 2:
			print(
				"Warning: wav format does not match engine defaults "
				f"(rate={wav_rate}, channels={wav_channels}, width={wav_width})."
			)

		while True:
			chunk = wav_file.readframes(frames_per_buffer)
			if not chunk:
				break
			await engine.append_audio(chunk)
			await asyncio.sleep(frames_per_buffer / rate)


async def _stream_mic(engine, frames_per_buffer: int, rate: int, channels: int) -> None:
	try:
		import sounddevice as sd
	except ImportError:
		print("sounddevice is not installed. Try: pip install sounddevice")
		return

	queue: asyncio.Queue[bytes] = asyncio.Queue()
	stream: Optional[sd.RawInputStream] = None

	try:
		loop = asyncio.get_running_loop()

		def _on_audio(indata, frames, time_info, status) -> None:
			if status:
				print(f"Audio status: {status}")
			loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))

		stream = sd.RawInputStream(
			samplerate=rate,
			channels=channels,
			dtype="int16",
			blocksize=frames_per_buffer,
			callback=_on_audio,
		)
		stream.start()
		print("Listening... Press Ctrl+C to stop.")

		while True:
			chunk = await queue.get()
			await engine.append_audio(chunk)
	except KeyboardInterrupt:
		print("\nStopping...")
	finally:
		if stream is not None:
			stream.stop()
			stream.close()


async def main() -> None:
	_ensure_backend_on_path()
	from engines.voice.ser_engine import SenseVoiceSEREngine

	engine = SenseVoiceSEREngine()
	await engine.start()

	stop_event = asyncio.Event()
	print_task = asyncio.create_task(_print_emotions(engine, 4, stop_event))

	rate = 44100
	channels = 2
	frames_per_buffer = int(rate * 0.2)
	wav_path = sys.argv[1] if len(sys.argv) > 1 else ""

	try:
		if wav_path:
			print(f"Streaming wav: {wav_path}")
			await _stream_wav(engine, wav_path, frames_per_buffer, rate)
		else:
			await _stream_mic(engine, frames_per_buffer, rate, channels)
	except KeyboardInterrupt:
		print("\nStopping...")
	finally:
		stop_event.set()
		print_task.cancel()
		try:
			await print_task
		except asyncio.CancelledError:
			pass
		await engine.stop()


if __name__ == "__main__":
	asyncio.run(main())
