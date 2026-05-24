# gRPC

gRPC is Google's RPC framework: define a service in **Protocol Buffers** (`.proto`); the compiler generates client + server code in your chosen language; calls feel like local function invocations. Fast (binary protobuf over HTTP/2), supports streaming, and ubiquitous in microservices.

## The shape

```proto
// trades.proto
syntax = "proto3";

package trades;

service TradesService {
  rpc Submit(Order) returns (Ack);
  rpc Stream(stream Heartbeat) returns (stream Fill);
  rpc ListOrders(ListRequest) returns (stream Order);
}

message Order {
  string id = 1;
  string symbol = 2;
  int32 quantity = 3;
  double price = 4;
}

message Ack {
  string id = 1;
  string status = 2;
}

message ListRequest {
  string symbol = 1;
}

message Heartbeat {
  int64 ts = 1;
}

message Fill {
  string order_id = 1;
  int32 qty = 2;
  double price = 3;
}
```

`rpc Submit(...) returns (...)` is a unary call. `rpc Stream(stream ...) returns (stream ...)` is bidirectional streaming. `rpc ListOrders(...) returns (stream ...)` is server streaming.

## Generating Python code

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. trades.proto
# Generates: trades_pb2.py + trades_pb2_grpc.py
```

`trades_pb2.py` has the message classes; `trades_pb2_grpc.py` has the stub and server classes.

## Implementing a server

```python
import grpc
from concurrent import futures
import trades_pb2
import trades_pb2_grpc


class TradesService(trades_pb2_grpc.TradesServiceServicer):
    def Submit(self, request, context):
        print(f"got order: {request}")
        return trades_pb2.Ack(id=request.id, status="ACCEPTED")

    def ListOrders(self, request, context):
        for order in load_orders(request.symbol):
            yield trades_pb2.Order(
                id=order.id, symbol=order.symbol,
                quantity=order.qty, price=order.price,
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trades_pb2_grpc.add_TradesServiceServicer_to_server(TradesService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
```

Each `rpc` becomes a method on `TradesServiceServicer`. Server-streaming methods `yield` messages instead of returning.

## Client

```python
import grpc
import trades_pb2
import trades_pb2_grpc


with grpc.insecure_channel("localhost:50051") as channel:
    stub = trades_pb2_grpc.TradesServiceStub(channel)

    # Unary
    ack = stub.Submit(trades_pb2.Order(id="X1", symbol="SPY", quantity=100, price=475.30))
    print(ack)

    # Server-streaming
    for order in stub.ListOrders(trades_pb2.ListRequest(symbol="SPY")):
        print(order)
```

The client code looks like ordinary Python. Underneath, gRPC handles serialisation, HTTP/2 framing, multiplexing.

## Async gRPC

```python
import asyncio
import grpc.aio
import trades_pb2
import trades_pb2_grpc


class TradesService(trades_pb2_grpc.TradesServiceServicer):
    async def Submit(self, request, context):
        return trades_pb2.Ack(id=request.id, status="ACCEPTED")


async def serve():
    server = grpc.aio.server()
    trades_pb2_grpc.add_TradesServiceServicer_to_server(TradesService(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()


asyncio.run(serve())
```

For high-concurrency, async gRPC scales much better than thread-pool.

## Bidirectional streaming

```python
class TradesService(trades_pb2_grpc.TradesServiceServicer):
    async def Stream(self, request_iterator, context):
        async for heartbeat in request_iterator:
            print(f"heartbeat at {heartbeat.ts}")
            # Yield zero or more fills in response
            yield trades_pb2.Fill(order_id="X1", qty=10, price=475.30)
```

Bidirectional streams are useful for chat-like interactions, live order entry with progress updates, market-data streams with subscriptions.

## TLS and auth

```python
# Server
with open("server.key", "rb") as f: private_key = f.read()
with open("server.crt", "rb") as f: cert_chain = f.read()
credentials = grpc.ssl_server_credentials([(private_key, cert_chain)])
server.add_secure_port("[::]:50051", credentials)


# Client
with open("ca.crt", "rb") as f: trust = f.read()
credentials = grpc.ssl_channel_credentials(root_certificates=trust)
with grpc.secure_channel("server.example.com:50051", credentials) as channel:
    ...
```

For per-call auth (e.g., JWT tokens):

```python
def auth_interceptor(continuation, client_call_details):
    metadata = list(client_call_details.metadata or [])
    metadata.append(("authorization", f"Bearer {token}"))
    new_details = client_call_details._replace(metadata=metadata)
    return continuation(new_details, request)


channel = grpc.intercept_channel(grpc.insecure_channel(...), AuthInterceptor())
```

Interceptors are gRPC's middleware. Use for auth, logging, retries, tracing.

## Errors and status codes

```python
def Submit(self, request, context):
    if not request.symbol:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "symbol is required")
    if not authorized(context):
        context.abort(grpc.StatusCode.PERMISSION_DENIED, "not authorized")
    ...
```

gRPC has standard status codes (similar to HTTP but richer): `OK`, `INVALID_ARGUMENT`, `NOT_FOUND`, `UNAUTHENTICATED`, `PERMISSION_DENIED`, `UNAVAILABLE`, `DEADLINE_EXCEEDED`, etc. Use them; clients can dispatch on them.

## Reflection and tooling

Enable server reflection for ad-hoc tools (like `grpcurl`):

```python
from grpc_reflection.v1alpha import reflection


SERVICE_NAMES = (
    trades_pb2.DESCRIPTOR.services_by_name["TradesService"].full_name,
    reflection.SERVICE_NAME,
)
reflection.enable_server_reflection(SERVICE_NAMES, server)
```

Now `grpcurl -plaintext localhost:50051 list` enumerates your services; `grpcurl -plaintext -d '{"id":"X1"}' localhost:50051 trades.TradesService/Submit` invokes them. Crucial for debugging.

## When gRPC is the right choice

- **Microservices** internal communication (vs REST: better typing, smaller payloads, streaming).
- **Polyglot stacks** — `.proto` files become the source of truth across Python / Go / Java / Rust services.
- **Streaming** — bidirectional, server-streaming, client-streaming all first-class.
- **Schema evolution** — protobuf has explicit field numbering; old clients gracefully ignore new fields.

When NOT:

- **Public APIs** — REST is more accessible; cURL-friendly.
- **Browser clients** — gRPC needs gRPC-Web proxy; HTTP+JSON is easier.
- **Tiny services with no schema discipline** — protobuf is overkill.

## Pitfalls

!!! warning "Field renumbering"
    Once a `.proto` field has a number, NEVER change it. Renumbering breaks all existing clients. The number is the wire identifier; the name is just for code generation.

!!! warning "Streaming + connection state"
    A long-running stream keeps a connection open. Servers behind load balancers / proxies often have idle timeouts that kill them. Use keepalives (`grpc.keepalive_time_ms`).

!!! warning "Generated code in version control"
    Some teams commit `*_pb2.py` files; others generate at build time. Pick one. Mixed approaches lead to "regenerate" pain.

!!! warning "Default values are invisible"
    proto3 doesn't distinguish "unset" from "default value". A `quantity=0` looks the same as "not provided". Use `optional` for fields where unset matters.

## Bottom line

For gRPC:

- **`.proto` is the source of truth** — define services + messages there.
- **Async server (`grpc.aio`)** for high concurrency.
- **Streaming methods** for chat / live data.
- **Server reflection + grpcurl** for debugging.
- **TLS + interceptors** for production.

Continue to **[MQTT](04-mqtt.md)**.
