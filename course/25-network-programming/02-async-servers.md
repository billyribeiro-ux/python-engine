# Async servers with asyncio

For a network service handling many concurrent connections, the modern Python answer is `asyncio`. Module 1 covered the basics; this chapter is server-side: how to actually write a TCP / line-oriented protocol server that survives production.

## The minimum-viable server

```python
import asyncio


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    print(f"connected: {addr}")
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            writer.write(b"echo: " + line)
            await writer.drain()
    except asyncio.CancelledError:
        raise
    except Exception:
        import logging
        logging.exception("client error")
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"disconnected: {addr}")


async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 5555)
    print("serving on tcp/5555")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
```

That's a complete async TCP server. Each client gets its own `handle_client` coroutine; thousands of concurrent clients on one process.

## Length-prefixed protocol

Building on Module 25 chapter 1:

```python
import asyncio
import struct


async def read_message(reader: asyncio.StreamReader) -> bytes:
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">I", header)
    return await reader.readexactly(length)


async def write_message(writer: asyncio.StreamWriter, payload: bytes):
    writer.write(struct.pack(">I", len(payload)) + payload)
    await writer.drain()


async def handle_client(reader, writer):
    try:
        while True:
            msg = await read_message(reader)
            response = process(msg)
            await write_message(writer, response)
    except asyncio.IncompleteReadError:
        # Peer closed cleanly
        pass
    finally:
        writer.close()
        await writer.wait_closed()
```

`readexactly(n)` raises `IncompleteReadError` if the peer closes mid-message. Catch it; treat as disconnect.

## Backpressure with `await writer.drain()`

If the client reads slowly, the server's send buffer fills. Without backpressure, the server's memory blows up. `drain()` waits until the buffer is below a threshold:

```python
writer.write(big_payload)
await writer.drain()                  # pauses if buffer is full
```

Forgetting `drain()` is the #1 cause of "my async server uses all the RAM under load."

## Concurrency limits

For services with a max-clients SLO:

```python
import asyncio


sem = asyncio.Semaphore(1000)         # max 1000 concurrent clients


async def handle_client(reader, writer):
    async with sem:
        # ... actual handler ...
        pass
```

Above the cap, new accepts wait until a slot frees. Cleaner than letting the OS limit the file descriptor count.

## Graceful shutdown

For zero-downtime deploys:

```python
import asyncio
import signal


async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 5555)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    async with server:
        serve_task = asyncio.create_task(server.serve_forever())
        await stop.wait()
        print("shutting down...")
        server.close()
        await server.wait_closed()
        # Optionally wait for active handlers to finish
        await asyncio.sleep(1)
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
```

`server.close()` stops accepting new connections but lets active ones finish. Combined with a deployer that waits a few seconds before killing the process, you get zero-dropped-request restarts.

## Timeouts on individual operations

```python
async def handle_client(reader, writer):
    try:
        async with asyncio.timeout(30):    # whole-session timeout
            while True:
                line = await reader.readline()
                if not line: break
                # ... process ...
    except TimeoutError:
        print("session timed out")
```

`asyncio.timeout(N)` (3.11+) is the modern pattern. For per-operation timeouts:

```python
try:
    line = await asyncio.wait_for(reader.readline(), timeout=5)
except TimeoutError:
    print("no data in 5s")
```

## TLS

```python
import ssl


context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

server = await asyncio.start_server(
    handle_client, "0.0.0.0", 443, ssl=context,
)
```

Same handler code; just pass `ssl=context`. The library transparently handles the TLS handshake.

## A worked example: a market-data fanout server

Real use case: pull a price feed from one upstream; broadcast each tick to all connected clients.

```python
import asyncio
import json


class FanoutServer:
    def __init__(self):
        self.clients: set[asyncio.StreamWriter] = set()
        self.lock = asyncio.Lock()

    async def add_client(self, writer):
        async with self.lock:
            self.clients.add(writer)

    async def remove_client(self, writer):
        async with self.lock:
            self.clients.discard(writer)
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    async def broadcast(self, message: bytes):
        # Snapshot to avoid mutation during iteration
        async with self.lock:
            clients = list(self.clients)
        for w in clients:
            try:
                w.write(message)
                await w.drain()
            except (ConnectionError, BrokenPipeError):
                await self.remove_client(w)

    async def handle_client(self, reader, writer):
        await self.add_client(writer)
        try:
            # Keep alive; client doesn't speak unless disconnecting
            while True:
                if await reader.read(1) == b"":
                    break
        finally:
            await self.remove_client(writer)


async def upstream_simulator(server: FanoutServer):
    """Pretend to be an upstream feed; push a tick every 100ms."""
    seq = 0
    while True:
        tick = json.dumps({"seq": seq, "symbol": "SPY", "price": 475 + seq % 5}).encode() + b"\n"
        await server.broadcast(tick)
        seq += 1
        await asyncio.sleep(0.1)


async def main():
    server = FanoutServer()
    asyncio_server = await asyncio.start_server(server.handle_client, "127.0.0.1", 5555)
    print("fanout server on tcp/5555")
    async with asyncio_server:
        await asyncio.gather(
            asyncio_server.serve_forever(),
            upstream_simulator(server),
        )


asyncio.run(main())
```

Production fanout in 50 lines. Handles thousands of clients on a single core; gracefully removes disconnected ones.

## Where this matters

- **Custom protocols** (proprietary tick feeds, internal RPC).
- **High-fanout streaming** (broadcast price ticks, system events).
- **WebSocket alternatives** when WS overhead is too high.
- **Embedded / IoT** that doesn't fit HTTP semantics.

For HTTP, use FastAPI (Module 22 chapter 11). For WebSocket-specific, see chapter 6 of this module. Raw asyncio TCP is the foundation underneath.

## Pitfalls

!!! warning "Mixing sync and async"
    `requests.get(...)` inside an async handler blocks the entire event loop. Use `httpx.AsyncClient` or `asyncio.to_thread(...)`.

!!! warning "Not calling `drain()`"
    Write-and-forget patterns work until the buffer fills. Always `await writer.drain()` after writes.

!!! warning "Leaking writer handles"
    Forgetting `writer.close()` and `await writer.wait_closed()` keeps file descriptors open. Use `try`/`finally`.

!!! warning "Slow handlers blocking the loop"
    A handler doing CPU-bound work (compress 1MB of data) blocks the event loop. Offload via `asyncio.to_thread` or `concurrent.futures.ProcessPoolExecutor`.

## Bottom line

For async servers:

- **`asyncio.start_server`** + per-client coroutine.
- **`readexactly`** + length-prefix for binary protocols.
- **`await writer.drain()`** for backpressure.
- **Signal-based graceful shutdown** for zero-downtime deploys.
- **TLS via `ssl.SSLContext`** with `ssl=` argument.

Continue to **[gRPC](03-grpc.md)**.
