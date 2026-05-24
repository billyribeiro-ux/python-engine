# Sockets fundamentals

`socket` is the OS interface for network I/O. Almost everything (HTTP, WebSocket, gRPC) sits on top of sockets — but writing socket code directly is sometimes the right answer for custom protocols, IoT, or talking to legacy systems.

## TCP — connection-oriented

Server:

```python
import socket


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 5555))
    srv.listen(5)
    print("listening on 5555")
    while True:
        conn, addr = srv.accept()
        with conn:
            data = conn.recv(4096)
            print(f"got {data!r} from {addr}")
            conn.sendall(b"hello\n")
```

Client:

```python
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect(("127.0.0.1", 5555))
    s.sendall(b"hi from client")
    print(s.recv(4096))
```

Three patterns to internalise:

- **`SO_REUSEADDR`** before bind — without it, restarting the server fails for ~60s while the kernel cleans up.
- **`recv(N)`** reads *up to* N bytes; may return less. Always handle short reads.
- **`sendall(data)`** loops internally until everything is sent; `send(data)` may return early.

## UDP — connectionless

```python
# Server
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5555))
while True:
    data, addr = sock.recvfrom(4096)
    print(f"got {data!r} from {addr}")
    sock.sendto(b"ack", addr)


# Client
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b"hello", ("127.0.0.1", 5555))
print(sock.recvfrom(4096))
```

UDP: unordered, unreliable, but low-latency. Used for video streaming, DNS, real-time game protocols. For most application-level work, use TCP.

## Reading a delimited stream

TCP is a byte stream; messages don't have boundaries. The two patterns:

**Length-prefixed**:

```python
import struct


def send_message(sock, payload: bytes):
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recv_message(sock) -> bytes:
    header = recv_exactly(sock, 4)
    (length,) = struct.unpack(">I", header)
    return recv_exactly(sock, length)


def recv_exactly(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf
```

4-byte big-endian length prefix; then exactly that many bytes. Robust; the standard.

**Delimiter-based** (line-oriented):

```python
def recv_line(sock) -> bytes:
    buf = b""
    while True:
        ch = sock.recv(1)
        if not ch or ch == b"\n":
            return buf
        buf += ch
```

Slow (one byte at a time) but works for human-readable protocols. For binary, use length-prefix.

## Non-blocking sockets and `select`

For multiple concurrent connections without threads:

```python
import socket
import select


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 5555))
srv.listen()
srv.setblocking(False)

inputs = [srv]
while True:
    readable, _, _ = select.select(inputs, [], [])
    for s in readable:
        if s is srv:
            conn, addr = srv.accept()
            conn.setblocking(False)
            inputs.append(conn)
        else:
            data = s.recv(4096)
            if data:
                s.sendall(b"ack: " + data)
            else:
                inputs.remove(s)
                s.close()
```

The pattern: `select` returns sockets ready to read; you handle each. For high-throughput servers, `selectors.DefaultSelector` (which uses epoll on Linux, kqueue on macOS) is what you want. For modern code, just use asyncio (next chapter).

## Timeouts

```python
sock.settimeout(5.0)        # 5 seconds for any operation
try:
    data = sock.recv(4096)
except socket.timeout:
    print("recv timed out")
```

A socket without a timeout will block forever on a hung connection. Always set timeouts in production code.

## TLS / SSL

```python
import ssl


context = ssl.create_default_context()
with socket.create_connection(("example.com", 443)) as sock:
    with context.wrap_socket(sock, server_hostname="example.com") as tls:
        tls.sendall(b"GET / HTTP/1.1\r\nHost: example.com\r\nConnection: close\r\n\r\n")
        print(tls.recv(4096))
```

For server-side TLS, `context.load_cert_chain(certfile, keyfile)` to load your cert. For mutual TLS, `context.load_verify_locations(cafile)` for client cert verification.

## Unix domain sockets

For inter-process communication on the same machine, much faster than TCP:

```python
import socket
import os


SOCK_PATH = "/tmp/myapp.sock"

# Server
if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK_PATH)
srv.listen(5)


# Client
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(SOCK_PATH)
client.sendall(b"hello")
```

No TCP overhead; secured by filesystem permissions. Used by Docker, systemd, PostgreSQL for local IPC.

## A worked example: a "tail -f" server

```python
import socket
import os
import time


def watch_and_serve(filepath: str, port: int = 5555):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(5)
    print(f"streaming {filepath} on tcp/{port}")
    while True:
        conn, addr = srv.accept()
        print(f"client {addr} connected")
        try:
            with open(filepath, "rb") as f:
                f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if line:
                        conn.sendall(line)
                    else:
                        time.sleep(0.5)
        except BrokenPipeError:
            print(f"client {addr} disconnected")
            conn.close()
```

Sixty lines for a "tail -f over TCP" service. Useful for shipping logs to a remote consumer in dev environments.

## Buffer sizes and Nagle's algorithm

```python
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
```

`TCP_NODELAY` disables Nagle's algorithm — sends small packets immediately instead of buffering. For latency-sensitive interactive protocols (trading order entry, gaming), turn it on. For bulk throughput, leave it off.

## Pitfalls

!!! warning "Buffer accumulation"
    `recv(4096)` may return only 50 bytes if the kernel hasn't received more. Handle partial reads via the length-prefix loop above.

!!! warning "Connection close detection"
    `recv()` returning `b""` means the peer closed. Distinct from `recv()` raising — both can mean "disconnected" in different conditions.

!!! warning "Holding sockets in finally blocks"
    A socket left open after a process crashes can keep the port "in use" for ~60s. Use `with` blocks.

!!! warning "SIGPIPE on send to closed socket"
    `sendall` to a closed peer can raise `BrokenPipeError`. Catch it explicitly; don't let it bubble up unhandled.

## Bottom line

For raw sockets:

- **TCP for reliable byte-streaming**; length-prefix your messages.
- **UDP for low-latency one-shot**; tolerate loss.
- **Always timeout**; always SO_REUSEADDR; always handle short reads.
- **TLS via `ssl.create_default_context()`**.
- **Modern code uses asyncio** (next chapter); raw sockets are for protocol implementation.

Continue to **[Async servers with asyncio](02-async-servers.md)**.
