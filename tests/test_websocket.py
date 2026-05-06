import asyncio
import json

import websockets


async def run_test() -> None:
    uri = "ws://localhost:8000/ws/session/"
    messages = [
        {"type": "video_frame", "data": ""},
        {"type": "audio_chunk", "data": ""},
        {
            "type": "transcript_committed",
            "transcript": "I have been feeling okay I guess, work has been fine.",
        },
    ]

    try:
        async with websockets.connect(uri) as websocket:
            for index, message in enumerate(messages, start=1):
                await websocket.send(json.dumps(message))
                await asyncio.sleep(2)

                if index == len(messages):
                    response = await websocket.recv()
                    print(response)
    except Exception as exc:
        print(f"WebSocket connection failed: {exc}")


if __name__ == "__main__":
    asyncio.run(run_test())
