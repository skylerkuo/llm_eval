"""
AI PC = client"""

from __future__ import annotations

import asyncio
import json
import websockets
from datetime import datetime
from rich.console import Console

console = Console()

URI = "ws://140.113.149.93:8765"  # replace with actual IP


async def client(msg: str, img: bool):
    async with websockets.connect(URI) as websocket:
        console.print("[green]Connected to server[/green]")

        # ✅ Add control flag
        payload = {
            "message": msg,
            "timestamp": str(datetime.now()),
            "request_image": img,  # <-- NEW
            "topic": "/eye_in_hand_cam_rgb"  # <-- NEW

        }

        await websocket.send(json.dumps(payload))
        console.print("[cyan]Sent JSON[/cyan]")

        # ✅ Only wait if we expect an image
        if payload["request_image"]:
            response = await websocket.recv()

            if isinstance(response, bytes):
                with open("received.jpg", "wb") as f:
                    f.write(response)

                console.print("[yellow]Image saved as received.jpg[/yellow]")
            else:
                console.print("[red]Expected binary but got text[/red]")


# if __name__ == "__main__":
#     asyncio.run(main())