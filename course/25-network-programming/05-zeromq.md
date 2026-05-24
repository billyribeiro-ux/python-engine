# ZeroMQ

ZeroMQ is "sockets on steroids": a messaging library with built-in patterns (pub/sub, req/rep, push/pull, dealer/router) that handle reconnection, queueing, and multi-transport for you. No central broker required — peers connect directly. Used in HFT, distributed computing, and anywhere you need fast inter-process messaging.

## Why ZeroMQ over plain sockets

- Built-in **message framing** — no length-prefix code needed.
- Automatic **reconnection** — a peer can disconnect and reconnect transparently.
- **Multiple transports** — same API works over TCP, IPC (Unix domain), inproc (within a process), pgm (multicast).
- **Patterns** — pub/sub, req/rep, push/pull, dealer/router each codify a use case.
- **Brokerless** — no central server to deploy.

## REQ / REP — request / response

```python
# Server (REP)
import zmq


ctx = zmq.Context()
sock = ctx.socket(zmq.REP)
sock.bind("tcp://*:5555")

while True:
    msg = sock.recv()
    print(f"got: {msg!r}")
    sock.send(b"world")


# Client (REQ)
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")
sock.send(b"hello")
print(sock.recv())
```

Strict alternation: REQ sends, REP receives, REP sends, REQ receives. Like RPC but with ZMQ's reliability.

## PUB / SUB — pub/sub

```python
# Publisher
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.bind("tcp://*:5556")

while True:
    pub.send_string(f"ticker:SPY 475.30")
    pub.send_string(f"ticker:QQQ 410.50")
    time.sleep(1)


# Subscriber
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5556")
sub.setsockopt_string(zmq.SUBSCRIBE, "ticker:SPY")     # filter
while True:
    msg = sub.recv_string()
    print(msg)
```

Subscribers filter by topic prefix on the wire. Publishers don't know who's subscribed (vs MQTT, which is broker-mediated).

## PUSH / PULL — pipelined work distribution

For "fan out work to N workers, collect results":

```python
# Ventilator (PUSH)
ctx = zmq.Context()
push = ctx.socket(zmq.PUSH)
push.bind("tcp://*:5557")
for i in range(1000):
    push.send_string(f"task {i}")


# Worker (PULL)
ctx = zmq.Context()
pull = ctx.socket(zmq.PULL)
pull.connect("tcp://localhost:5557")
while True:
    task = pull.recv_string()
    process(task)
```

Workers are completely interchangeable; the broker (ventilator) round-robins among them. Adding a new worker = starting another process. Killing one mid-task = the next task goes to whoever's free.

## Asyncio integration

```python
import asyncio
import zmq
import zmq.asyncio


async def main():
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect("tcp://localhost:5556")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    while True:
        msg = await sock.recv()
        print(msg)


asyncio.run(main())
```

Same patterns, async. Drop into asyncio servers (chapter 2) without bridging threads.

## Multi-part messages

```python
# Sender
sock.send_multipart([b"header", b"payload"])


# Receiver
parts = sock.recv_multipart()
print(parts)        # [b"header", b"payload"]
```

For protocols with metadata + body. ZMQ guarantees all parts arrive together or not at all.

## A worked example: distributed task queue

```python
# coordinator.py
import zmq

ctx = zmq.Context()
push = ctx.socket(zmq.PUSH)
push.bind("tcp://*:5560")
pull = ctx.socket(zmq.PULL)
pull.bind("tcp://*:5561")

# Dispatch
for i in range(1000):
    push.send_json({"id": i, "data": [1, 2, 3]})

# Collect results
results = []
for _ in range(1000):
    results.append(pull.recv_json())
print(f"got {len(results)} results")


# worker.py
import zmq

ctx = zmq.Context()
pull = ctx.socket(zmq.PULL)
pull.connect("tcp://coordinator:5560")
push = ctx.socket(zmq.PUSH)
push.connect("tcp://coordinator:5561")

while True:
    task = pull.recv_json()
    result = compute(task)
    push.send_json({"id": task["id"], "result": result})
```

Start `coordinator.py` once; start N copies of `worker.py`; they auto-discover and distribute work. No external dependencies, no broker. ZMQ handles connection management.

## ZMQ vs alternatives

| | ZeroMQ | RabbitMQ | Kafka | gRPC |
|---|---|---|---|---|
| Broker | none | central | central | none |
| Persistence | no | yes | yes | no |
| Throughput | very high | high | very high | high |
| Latency | lowest | low | medium | low |
| Use case | inter-process, HFT-adjacent | typical messaging | event streaming | service RPC |

ZMQ wins on latency and simplicity (no broker to deploy). Loses when you need persistence (a crashed receiver loses queued messages).

## Inproc transport — zero-cost in-process messaging

```python
ctx = zmq.Context()
sock_a = ctx.socket(zmq.PAIR)
sock_a.bind("inproc://mychannel")

sock_b = ctx.socket(zmq.PAIR)
sock_b.connect("inproc://mychannel")

sock_a.send(b"hello")
print(sock_b.recv())
```

`inproc://` is a transport that uses shared memory — faster than IPC, faster than TCP loopback. Useful for talking between async tasks or different libraries in the same process.

## Pitfalls

!!! warning "REQ/REP lockstep"
    A REQ socket can't send another request until the REP socket has responded. Hung server = client stuck forever (without `RCVTIMEO`).

!!! warning "Slow subscriber drops"
    By default, PUB sockets discard messages if the SUB is slow. Set `sock.set_hwm(0)` (zero hi-water mark) for unbounded queueing — but watch memory.

!!! warning "No persistence"
    A crashed worker loses its current task. Build your protocol around acknowledgements + retry, or accept the data loss.

!!! warning "Context lifecycle"
    `ctx.term()` blocks until all sockets close. Forgetting to close sockets keeps your program alive forever.

## Bottom line

For ZeroMQ:

- **REQ/REP** for RPC-like; **PUB/SUB** for broadcast; **PUSH/PULL** for work queues.
- **No broker** — peers connect directly.
- **Multi-transport** — TCP, IPC, inproc.
- **Asyncio-compatible** via `zmq.asyncio`.
- **No persistence** — combine with a DB if needed.

Continue to **[WebSockets in depth](06-websockets.md)**.
