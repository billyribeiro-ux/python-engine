# Admin tooling and remote ops

When you need to do something across many machines — deploy code, run a query everywhere, restart a service, audit configurations — Python is the right glue. This chapter covers the standard libraries for remote ops, process management, and system introspection.

## `psutil` — local system introspection

The single most useful "ops" library. Cross-platform information about CPU, memory, disk, processes, network.

```python
import psutil


# System-wide
print(f"CPU: {psutil.cpu_percent(interval=1)}%")
print(f"RAM: {psutil.virtual_memory().percent}% used "
      f"({psutil.virtual_memory().available / 1e9:.1f} GB free)")
print(f"Disk: {psutil.disk_usage('/').percent}% used")
print(f"Network bytes sent: {psutil.net_io_counters().bytes_sent}")

# Processes
for proc in psutil.process_iter(["name", "memory_info", "cpu_percent"]):
    if proc.info["name"].startswith("python"):
        print(proc.pid, proc.info["name"], proc.info["memory_info"].rss / 1e6, "MB")

# Kill a runaway process
proc = psutil.Process(pid=12345)
proc.terminate()                          # SIGTERM
try:
    proc.wait(timeout=10)
except psutil.TimeoutExpired:
    proc.kill()                            # SIGKILL — last resort
```

For monitoring scripts and watchdogs, psutil is the default tool. The whole API works identically on Linux, macOS, and Windows.

## `paramiko` — SSH from Python

For running commands on a single remote machine:

```python
import paramiko


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("server.example.com", username="deploy", key_filename="/home/me/.ssh/id_ed25519")

stdin, stdout, stderr = client.exec_command("uptime")
print(stdout.read().decode())
print(stderr.read().decode())

# File transfer
sftp = client.open_sftp()
sftp.put("/local/file", "/remote/file")
sftp.get("/remote/log", "/local/log")
sftp.close()

client.close()
```

`paramiko` is the building block; for production, wrap it.

## `fabric` — task-based deployment

[`fabric`](https://www.fabfile.org/) builds on paramiko with the "tasks across hosts" abstraction:

```python
from fabric import Connection, task, ThreadingGroup


HOSTS = ["web-1.example.com", "web-2.example.com", "web-3.example.com"]


@task
def deploy(c):
    """Run on a single host."""
    c.run("cd /opt/app && git pull")
    c.run("cd /opt/app && pip install -r requirements.txt")
    c.run("sudo systemctl restart app")


def deploy_all():
    """Run across all hosts in parallel."""
    group = ThreadingGroup(*HOSTS)
    group.run("cd /opt/app && git pull")
    group.run("sudo systemctl restart app")


# Invoke from CLI:
# fab -H web-1.example.com,web-2.example.com deploy
```

Threading group runs the same command across many hosts simultaneously — the cleanest "do this on all servers" tool.

## `pyinfra` — declarative remote ops

When you want **idempotent** remote state management (like Ansible, but in Python), `pyinfra`:

```python
# tasks.py
from pyinfra.operations import apt, files, server


apt.packages(
    name="Install required packages",
    packages=["nginx", "python3-venv"],
    update=True,
)

files.template(
    name="Deploy nginx config",
    src="templates/nginx.conf.j2",
    dest="/etc/nginx/sites-enabled/myapp.conf",
)

server.service(
    name="Ensure nginx is running",
    service="nginx",
    running=True,
    enabled=True,
)
```

Run: `pyinfra inventory.py tasks.py`. The "name=" descriptions become operation names in the output; each operation knows how to be idempotent (won't reinstall packages that are already there).

For small fleets and Python shops, pyinfra is more comfortable than Ansible.

## A worked example: parallel health check

```python
import asyncio
import asyncssh                            # async SSH library


async def check_one_host(host: str) -> dict:
    try:
        async with asyncssh.connect(host, known_hosts=None) as conn:
            uptime = (await conn.run("uptime")).stdout.strip()
            mem = (await conn.run("free -h | awk '/Mem:/ {print $4}'")).stdout.strip()
            df = (await conn.run("df -h / | awk 'NR==2 {print $4, $5}'")).stdout.strip()
            return {"host": host, "uptime": uptime, "free_mem": mem, "disk": df, "ok": True}
    except Exception as exc:
        return {"host": host, "ok": False, "error": str(exc)}


async def check_all(hosts: list[str]):
    results = await asyncio.gather(*(check_one_host(h) for h in hosts))
    for r in results:
        print(r)


if __name__ == "__main__":
    asyncio.run(check_all(["web-1", "web-2", "db-1", "queue-1"]))
```

Run a quick health audit across the fleet in seconds. The async version finishes in the time of the slowest single host.

## Working with cloud APIs

For AWS / GCP / Azure, the official SDKs are the right tool:

```python
import boto3                               # AWS

s3 = boto3.client("s3")
s3.upload_file("local.csv", "my-bucket", "data/2024-11-15.csv")

ec2 = boto3.client("ec2")
instances = ec2.describe_instances()
for reservation in instances["Reservations"]:
    for inst in reservation["Instances"]:
        print(inst["InstanceId"], inst["State"]["Name"])
```

```python
from google.cloud import storage          # GCP

client = storage.Client()
bucket = client.bucket("my-bucket")
blob = bucket.blob("data/2024-11-15.csv")
blob.upload_from_filename("local.csv")
```

For multi-region deployments, treat the SDK calls as I/O and wrap them in retry + circuit breaker patterns (Module 5 chapter 2).

## Structured output for chaining tools

If your ops script's output will feed another tool, emit JSON, not free text:

```python
import json
import sys


def main():
    result = do_audit()
    json.dump(result, sys.stdout, default=str)
    sys.stdout.write("\n")


# Consume with jq:
# python audit.py | jq '.servers[] | select(.disk_pct > 80)'
```

JSON output makes your scripts compose into Unix-pipe pipelines.

## Pitfalls

!!! warning "SSH without `known_hosts` check"
    `AutoAddPolicy()` accepts any host key — vulnerable to man-in-the-middle. For production, pin known_hosts or use a CA.

!!! warning "Storing credentials in scripts"
    Use env vars, AWS profiles, GCP service accounts, or a secrets manager. Never commit a key.

!!! warning "Running ad-hoc commands as root"
    `sudo` your way through a script and one typo wipes a filesystem. Prefer narrow `sudoers` rules (allow specific commands only) over passwordless `sudo` for everything.

!!! warning "Synchronous loops across hundreds of hosts"
    100 hosts × 5 seconds each = 8 minutes. Async or threading gives you 5 seconds.

## Bottom line

For admin tooling:

- **`psutil`** for local system state.
- **`paramiko`** for raw SSH; **`fabric`** for tasks across hosts; **`pyinfra`** for idempotent state.
- **Async / threading** for parallel across many hosts.
- **Cloud SDKs** (`boto3`, `google-cloud-*`) for cloud resources.
- **JSON output** so scripts compose.

Continue to **[Migrations and repair scripts](09-migrations-and-repair.md)**.
