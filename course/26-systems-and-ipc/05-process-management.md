# Process management and supervision

For a Python service that runs forever, "just `python service.py`" isn't enough. You need a supervisor that restarts on crash, captures logs, handles graceful shutdown, and starts on boot. This chapter covers the working choices.

## Decision matrix

| Need | Use |
|---|---|
| Single Linux box, simple service | **systemd** |
| Single Linux box, lightweight process supervisor | **supervisor** (pip-installable) |
| Container | **dumb-init** + container orchestrator |
| Kubernetes / ECS / Cloud Run | platform's restart policy |
| Cluster of machines, declarative | **Nomad** / **Kubernetes** |
| Local dev | **honcho** / **foreman** / **overmind** |

For ~95% of small-to-medium Python services on Linux: **systemd**.

## systemd service file

```ini
# /etc/systemd/system/myservice.service
[Unit]
Description=My Python service
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=myuser
WorkingDirectory=/opt/myservice
EnvironmentFile=/etc/myservice/env
ExecStart=/opt/myservice/.venv/bin/python -m myservice
Restart=on-failure
RestartSec=5s
StartLimitInterval=300
StartLimitBurst=5

# Resource limits
MemoryMax=2G
CPUQuota=200%

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=myservice

[Install]
WantedBy=multi-user.target
```

Key fields:

- **`Type=simple`** — the process stays in foreground (don't fork into background; systemd manages it).
- **`Restart=on-failure`** — restart only on non-zero exit. Use `always` for unconditional, `no` for one-shot.
- **`RestartSec=5s`** — wait before restart.
- **`StartLimitInterval` + `StartLimitBurst`** — give up after 5 restarts in 5 minutes (prevents tight crash loops).
- **`MemoryMax`** + **`CPUQuota`** — cgroup limits.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now myservice
sudo systemctl status myservice
journalctl -u myservice -f
```

## Graceful shutdown via signals

systemd sends `SIGTERM` on stop; if the process doesn't exit within 90s (default), `SIGKILL`. Configure if needed:

```ini
[Service]
TimeoutStopSec=30s
KillSignal=SIGTERM
```

Your Python code should handle SIGTERM (Module 26 chapter 3):

```python
import signal


def shutdown(signum, frame):
    log.info("got SIGTERM, cleaning up")
    # ... close connections, flush logs ...
    sys.exit(0)


signal.signal(signal.SIGTERM, shutdown)
```

Without a handler, default behaviour is immediate termination — data loss for anything mid-write.

## `supervisor` — pip-installable supervisor

For environments without systemd (some containers, older OSes):

```bash
pip install supervisor
```

```ini
# /etc/supervisor/conf.d/myservice.conf
[program:myservice]
command=/opt/myservice/.venv/bin/python -m myservice
directory=/opt/myservice
user=myuser
autostart=true
autorestart=true
startretries=5
stdout_logfile=/var/log/myservice/stdout.log
stderr_logfile=/var/log/myservice/stderr.log
environment=LOG_LEVEL="INFO",API_KEY="%(ENV_API_KEY)s"
```

```bash
supervisorctl reread
supervisorctl update
supervisorctl status myservice
supervisorctl tail myservice
```

`supervisor` has a built-in HTTP UI and CLI; useful for dev environments.

## Containers and `dumb-init`

Inside a container, your Python process is PID 1. PID 1 has special signal-handling rules — it doesn't get the default signal handlers. Without care, your container ignores SIGTERM from Docker / Kubernetes.

The fix: `dumb-init` as PID 1:

```dockerfile
RUN apt-get install -y dumb-init
ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "-m", "myservice"]
```

`dumb-init` is a minimal init that forwards signals to your Python process. Kubernetes / Docker can now stop your container cleanly.

Alternative: `tini`, `init` in newer Docker, or just `python -X faulthandler -m myservice` with explicit signal handlers.

## Process supervision in pure Python

For a parent process supervising children itself:

```python
import subprocess
import time
import signal


class Supervisor:
    def __init__(self, command: list[str], max_restarts: int = 5, restart_window: int = 300):
        self.command = command
        self.max_restarts = max_restarts
        self.restart_window = restart_window
        self.restarts: list[float] = []
        self.stopping = False

    def _should_restart(self) -> bool:
        now = time.time()
        self.restarts = [t for t in self.restarts if now - t < self.restart_window]
        if len(self.restarts) >= self.max_restarts:
            return False
        self.restarts.append(now)
        return True

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopping", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stopping", True))
        while not self.stopping:
            proc = subprocess.Popen(self.command)
            try:
                proc.wait()
            except KeyboardInterrupt:
                break
            if self.stopping:
                proc.terminate()
                proc.wait(timeout=10)
                break
            print(f"child exited {proc.returncode}; checking restart policy")
            if not self._should_restart():
                print("restart limit hit; giving up")
                break
            time.sleep(2)


if __name__ == "__main__":
    Supervisor(["python", "child.py"], max_restarts=5).run()
```

For embedded supervision (a Python launcher that manages workers), this pattern is clean and self-contained.

## Health checks

systemd / Kubernetes / load balancers all want to know "is this service alive?". Implement a health endpoint:

```python
# Inside your FastAPI app (Module 22 ch11):
@app.get("/health")
def health():
    # Check upstream dependencies; return 503 if any are down
    if not database_ok():
        raise HTTPException(503, "db down")
    return {"status": "ok"}
```

In Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

Liveness probe failure → container restart. Readiness probe failure → no traffic routed.

## Logging — to where

For systemd services:

```ini
StandardOutput=journal
StandardError=journal
```

Logs go to `journalctl` automatically. `journalctl -u myservice -f` to tail.

For containerised services: stdout / stderr → container runtime captures → log aggregation (Loki / CloudWatch / Datadog) ingests. Make sure your Python code doesn't buffer:

```bash
ENV PYTHONUNBUFFERED=1
```

In `pyproject.toml` or Dockerfile. Without it, logs may not appear until the buffer flushes.

## Resource limits

For services that might leak:

```ini
[Service]
MemoryMax=2G              # OOM-killed if exceeds
MemoryHigh=1.5G           # throttle if exceeds
CPUQuota=200%             # max 2 CPUs
TasksMax=200              # max 200 threads / processes
```

The OOM killer is a real production tool — better to crash and restart than silently degrade.

## A simpler local-dev alternative — `honcho`

```bash
pip install honcho
```

`Procfile`:

```
web: gunicorn myapp:app
worker: python worker.py
scheduler: python -m apscheduler
```

```bash
honcho start
```

Starts all three services with prefixed output. Good for local dev that mirrors production process layout.

## When to use what

- **One-off batch script**: don't supervise; just run.
- **Background daemon on one box**: systemd.
- **Container in Kubernetes**: K8s manages restarts; just need dumb-init / tini.
- **Multi-process supervised by Python parent**: pure-Python supervisor (the example above) or `multiprocessing.Process`.
- **Local dev mirroring prod**: `honcho` + Procfile.

## Pitfalls

!!! warning "Forking inside services"
    A Python service that forks creates orphan children that systemd doesn't see. Either don't fork or use `Type=forking` and a PID file.

!!! warning "Tight restart loops"
    Without `StartLimitInterval` / `StartLimitBurst`, a service that crashes immediately restarts immediately — busy loop. Always set limits.

!!! warning "Log rotation"
    `journald` rotates automatically. File-based logs need `logrotate`. Without rotation, disks fill up.

!!! warning "Time spent in shutdown"
    `TimeoutStopSec=30s` is generous for some services, tight for others. Match to how long your cleanup actually takes.

## Bottom line

For process supervision:

- **systemd on Linux** for the common case.
- **dumb-init / tini** for containers as PID 1.
- **K8s liveness / readiness probes** + `/health` endpoint.
- **`Restart=on-failure`** + StartLimit to prevent crash loops.
- **MemoryMax / CPUQuota** for resource isolation.
- **journald** for logs; `PYTHONUNBUFFERED=1` for containers.

## End of Module 26

You now have the systems programming toolkit. The next module covers stdlib + scientific computing topics — dates, regex, crypto, geospatial, and the scipy ecosystem outside of pandas / numpy.

Continue to **[Module 27 — Dates, regex, crypto, geo, scientific](../27-stdlib-and-scientific/index.md)**.
