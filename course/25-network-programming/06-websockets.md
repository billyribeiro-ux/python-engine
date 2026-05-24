# WebSockets in depth

Module 5 chapter 5 covered consuming WebSocket feeds (a trading use case). This chapter is the protocol-level toolkit: building servers, the wire format, frames, ping/pong, broadcast patterns.

## The `websockets` library

```python
import asyncio
import websockets


async def handler(ws):
    async for msg in ws:
        print(f"got: {msg}")
        await ws.send(f"echo: {msg}")


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()         # run forever


asyncio.run(main())
```

That's a complete WebSocket server. Each client connection runs `handler`. `async for msg in ws` consumes messages until the client disconnects.

Client:

```python
import asyncio
import websockets


async def client():
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send("hello")
        print(await ws.recv())


asyncio.run(client())
```

## Frame types

WebSocket has text and binary frames:

```python
async for msg in ws:
    if isinstance(msg, str):
        print(f"text: {msg}")
    elif isinstance(msg, bytes):
        print(f"binary: {msg!r}")
```

For JSON, send text:

```python
await ws.send(json.dumps({"type": "subscribe", "topic": "ticks"}))
```

For binary protocols (protobuf, MessagePack), send bytes:

```python
await ws.send(my_proto.SerializeToString())
```

## Broadcast — fan-out to many subscribers

```python
import asyncio
import websockets


CONNECTIONS = set()


async def register(ws):
    CONNECTIONS.add(ws)
    try:
        async for msg in ws:
            pass
    finally:
        CONNECTIONS.remove(ws)


async def broadcast(message):
    if CONNECTIONS:
        await asyncio.gather(
            *(ws.send(message) for ws in CONNECTIONS),
            return_exceptions=True,
        )


async def producer():
    """Emit a heartbeat every second."""
    counter = 0
    while True:
        await asyncio.sleep(1)
        await broadcast(f"tick {counter}")
        counter += 1


async def main():
    async with websockets.serve(register, "localhost", 8765):
        await producer()


asyncio.run(main())
```

`return_exceptions=True` is important — one slow client shouldn't crash the broadcast.

For thousands of connections, `websockets.broadcast(connections, message)` is the optimised version that doesn't await individual sends.

## Keep-alive

WebSocket has built-in ping/pong frames:

```python
async with websockets.serve(handler, "localhost", 8765,
                              ping_interval=20, ping_timeout=20):
    ...
```

Every 20s the server pings the client; if no pong in 20s, the connection closes. Catches network failures invisibly.

For the client:

```python
async with websockets.connect("ws://...", ping_interval=20, ping_timeout=20) as ws:
    ...
```

Always set ping_interval/ping_timeout. Without them, broken connections sit forever consuming resources.

## Subprotocols

For protocol versioning:

```python
async def handler(ws):
    print(f"subprotocol: {ws.subprotocol}")
    ...


# Server advertising subprotocols
async with websockets.serve(handler, "localhost", 8765,
                              subprotocols=["v1.protocol", "v2.protocol"]):
    ...


# Client requesting a specific one
async with websockets.connect("ws://localhost:8765",
                               subprotocols=["v2.protocol"]) as ws:
    ...
```

The negotiation happens in the WebSocket handshake. Both sides must agree on a subprotocol — clean way to support multiple versions.

## Headers and authentication

```python
# Client sends auth via headers
async with websockets.connect(
    "wss://api.example.com/ws",
    additional_headers={"Authorization": f"Bearer {token}"},
) as ws:
    ...


# Server validates
async def handler(ws):
    auth = ws.request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        await ws.close(code=4401, reason="unauthorised")
        return
    # ... verify token ...
```

Close codes:

- **1000** normal closure
- **1001** going away (e.g., server shutdown)
- **1008** policy violation
- **4000-4999** application-defined

## TLS / WSS

```python
import ssl


ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ctx.load_cert_chain("server.crt", "server.key")

async with websockets.serve(handler, "0.0.0.0", 443, ssl=ctx):
    ...
```

WSS (secure WebSocket) is just WebSocket-over-TLS. Same handler code; pass `ssl=` argument.

## A worked example: a real-time dashboard backend

```python
import asyncio
import json
import websockets


class DashboardServer:
    def __init__(self):
        self.subscribers: dict[str, set] = {}   # topic -> set of websockets

    async def handle(self, ws):
        # Client sends {"action": "subscribe", "topic": "..."}
        try:
            async for msg in ws:
                cmd = json.loads(msg)
                if cmd["action"] == "subscribe":
                    self.subscribers.setdefault(cmd["topic"], set()).add(ws)
                    await ws.send(json.dumps({"status": "subscribed", "topic": cmd["topic"]}))
                elif cmd["action"] == "unsubscribe":
                    self.subscribers.get(cmd["topic"], set()).discard(ws)
        finally:
            for subs in self.subscribers.values():
                subs.discard(ws)

    async def publish(self, topic: str, payload: dict):
        subs = list(self.subscribers.get(topic, ()))
        if subs:
            message = json.dumps({"topic": topic, "data": payload})
            await asyncio.gather(*(s.send(message) for s in subs), return_exceptions=True)


async def main():
    server = DashboardServer()

    async def background_publisher():
        """Pretend to publish updates."""
        counter = 0
        while True:
            await server.publish("pnl", {"pnl": counter * 100})
            await server.publish("positions", {"long": 100, "short": -50})
            counter += 1
            await asyncio.sleep(1)

    async with websockets.serve(server.handle, "localhost", 8765,
                                  ping_interval=20):
        await background_publisher()


asyncio.run(main())
```

Topic-based subscription; selective publish. Production dashboards behind 100-line Python services.

## Reconnection on the client

```python
import asyncio
import websockets


async def resilient_client(url: str):
    backoff = 1
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                backoff = 1
                async for msg in ws:
                    await handle_message(msg)
        except (websockets.ConnectionClosed, OSError) as exc:
            print(f"reconnect after {backoff}s ({exc})")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)
```

Exponential backoff up to 60s. Don't reconnect immediately — you'll hammer the server during outages.

## Pitfalls

!!! warning "Async iteration that doesn't yield"
    `async for msg in ws: heavy_sync_work()` blocks the entire connection. Offload via `asyncio.to_thread`.

!!! warning "Single set + concurrent modify"
    Adding to / removing from `CONNECTIONS` from multiple coroutines without coordination races. Either snapshot before iterating or use proper locking.

!!! warning "Unbounded message size"
    By default `websockets` accepts up to 1 MB per message. For large payloads (gigabyte file transfer), use a different protocol (HTTP file upload). For small but unbounded (event streams), set `max_size=None` and document the limits.

!!! warning "WebSocket vs Server-Sent Events"
    If you only need server → client streaming, SSE (`text/event-stream`) is simpler and works through any HTTP middleware. WebSockets are bidirectional; use them when you actually need that.

## Bottom line

For WebSockets:

- **`websockets` library** for both client and server.
- **Always `ping_interval` + `ping_timeout`** for liveness.
- **Subprotocols** for versioning.
- **Reconnection with exponential backoff** on the client.
- **TLS via `ssl=`** for production.

## End of Module 25

You now have the network programming toolkit. The next module — System programming & IPC — covers the lower-level OS primitives: processes, IPC, shared memory, signals at depth.

Continue to **[Module 26 — System programming & IPC](../26-systems-and-ipc/index.md)**.
