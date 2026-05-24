# MQTT

MQTT is the pub/sub protocol of IoT, but useful far beyond — anywhere you have many publishers and many subscribers on flexible topic hierarchies. Lightweight (small header), reliable (QoS levels), and supported by every major cloud platform.

This chapter covers using MQTT from Python with `paho-mqtt` and `aiomqtt`.

## The model

- **Topics** — hierarchical strings: `sensors/floor1/room3/temperature`.
- **Publishers** send messages to a topic.
- **Subscribers** subscribe to topic patterns (including wildcards) and receive everything published there.
- **Broker** — the central server that routes messages. Mosquitto (open source), HiveMQ, AWS IoT Core, Azure IoT Hub.

A wildcard: `sensors/+/+/temperature` matches any room on any floor.

A multi-level wildcard: `sensors/#` matches everything under `sensors/`.

## Basic publish

```python
import paho.mqtt.client as mqtt
import json


client = mqtt.Client()
client.connect("broker.example.com", 1883, 60)
client.loop_start()                          # start network thread

client.publish("sensors/lab/temperature", json.dumps({"value": 22.5, "ts": 1700000000}))
client.publish("sensors/lab/humidity", json.dumps({"value": 45.0, "ts": 1700000000}))

client.loop_stop()
client.disconnect()
```

`loop_start()` runs the network loop in a background thread. For long-running publishers, that's right; for one-shots, `client.publish(...); client.loop()` (single iteration) works too.

## Subscribe

```python
import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    print(f"connected: rc={rc}")
    client.subscribe("sensors/#")


def on_message(client, userdata, msg):
    print(f"{msg.topic} -> {msg.payload.decode()}")


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("broker.example.com", 1883, 60)
client.loop_forever()
```

`loop_forever()` blocks; for embedded use cases, that's fine. For mixed-workload services, use `loop_start()` and run other code alongside.

## QoS — quality of service

MQTT has three QoS levels for each message:

| QoS | Guarantee |
|---|---|
| **0** | at-most-once delivery; fire and forget |
| **1** | at-least-once; may deliver duplicates |
| **2** | exactly-once; slowest; rarely needed |

```python
client.publish(topic, payload, qos=1)
client.subscribe(topic, qos=1)
```

For sensor telemetry, QoS 0 is usually fine. For commands ("turn off the heater"), QoS 1. QoS 2 is heavy and almost always overkill.

## Retained messages

```python
client.publish("sensors/lab/temperature", "22.5", retain=True)
```

The broker stores this; any new subscriber to that topic immediately receives the last retained message. Useful for "the current state" of a device — a subscriber connecting later doesn't have to wait for the next update.

## Last Will and Testament (LWT)

A message the broker publishes on your behalf if your connection drops:

```python
client.will_set("clients/myservice/status", "offline", qos=1, retain=True)
client.connect(...)
client.publish("clients/myservice/status", "online", qos=1, retain=True)
```

Subscribers can monitor `clients/+/status` to know which devices are alive. Critical for IoT fleets.

## Async with `aiomqtt`

```python
import asyncio
import aiomqtt


async def main():
    async with aiomqtt.Client("broker.example.com") as client:
        await client.subscribe("sensors/#")
        async for message in client.messages:
            print(f"{message.topic} -> {message.payload.decode()}")


asyncio.run(main())
```

`aiomqtt` is the modern async wrapper. Cleaner than juggling threads.

## TLS

```python
import ssl


context = ssl.create_default_context()
client.tls_set_context(context)
client.connect("broker.example.com", 8883, 60)        # 8883 = MQTT-over-TLS port
```

For production, always TLS. For mutual TLS (client cert), use `context.load_cert_chain(certfile, keyfile)`.

## Username / password

```python
client.username_pw_set("alice", "secret")
client.connect(...)
```

For brokers that require it (mosquitto with ACLs, AWS IoT, HiveMQ Cloud).

## A worked example: device fleet monitor

```python
import asyncio
import json
import aiomqtt
from collections import defaultdict


class FleetMonitor:
    def __init__(self):
        self.devices = defaultdict(dict)        # {device_id: {state}}

    async def run(self):
        async with aiomqtt.Client("broker.example.com") as client:
            await client.subscribe("devices/+/+")
            print("monitoring fleet...")
            async for msg in client.messages:
                parts = str(msg.topic).split("/")
                # devices/<device_id>/<measurement>
                if len(parts) != 3:
                    continue
                device_id, measurement = parts[1], parts[2]
                try:
                    value = json.loads(msg.payload.decode())
                except json.JSONDecodeError:
                    value = msg.payload.decode()
                self.devices[device_id][measurement] = value
                self._maybe_alert(device_id, measurement, value)

    def _maybe_alert(self, device_id, measurement, value):
        # Example: alert if temperature > threshold
        if measurement == "temperature" and isinstance(value, dict):
            v = value.get("value")
            if v and v > 80:
                print(f"ALERT: {device_id} hot ({v}°C)")


if __name__ == "__main__":
    asyncio.run(FleetMonitor().run())
```

Monitor 10,000 devices in one Python process. Each MQTT message is ~50 bytes; the broker handles routing.

## When MQTT wins

- **Many publishers + many subscribers** — pub/sub topology is native.
- **Constrained networks** (cellular, satellite) — low overhead per message.
- **Mobile / IoT** — devices behind NAT can subscribe without inbound ports.
- **Last-known-state queries** — retained messages.
- **Asymmetric reliability** — different QoS per message type.

When NOT:

- **Request/response** patterns — better fit for HTTP / gRPC.
- **High-throughput bulk transfer** — MQTT has per-message overhead.
- **Strong ordering across publishers** — MQTT doesn't guarantee global order.

## Self-hosting Mosquitto

```bash
brew install mosquitto         # macOS
apt install mosquitto          # Ubuntu/Debian
mosquitto -v                   # run in verbose mode for testing
```

A single Mosquitto instance handles ~100k connected clients on a normal machine. For more, use the clustering variants (EMQX, VerneMQ) or a managed service.

## Pitfalls

!!! warning "Wildcards can be expensive"
    A subscription to `#` on a busy broker delivers EVERY message. Subscribe narrowly.

!!! warning "QoS 2 in production"
    QoS 2 adds latency and broker load. Almost never necessary; QoS 1 + idempotent handlers is usually better.

!!! warning "Retained-message accumulation"
    Every retained message stays until explicitly cleared (`publish(topic, "", retain=True)`). Misuse leads to brokers full of stale state.

!!! warning "Topic taxonomy drift"
    Once devices publish to `sensors/lab/temperature/value/...`, renaming the schema is painful. Plan the topic hierarchy carefully up front.

## Bottom line

For MQTT:

- **paho-mqtt** for sync; **aiomqtt** for async.
- **QoS 0 for telemetry**; QoS 1 for commands.
- **Retained messages** for "current state".
- **LWT** for liveness monitoring.
- **Always TLS** in production.

Continue to **[ZeroMQ](05-zeromq.md)**.
