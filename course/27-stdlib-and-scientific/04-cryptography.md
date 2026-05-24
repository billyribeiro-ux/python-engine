# Cryptography

The one rule of cryptography: **don't roll your own**. Use vetted libraries; understand what they give you; don't try to be clever. This chapter covers the standard tools and the patterns that don't get you fired.

## `hashlib` — non-cryptographic and cryptographic hashes

```python
import hashlib


# SHA-256 — the modern default
h = hashlib.sha256(b"hello").hexdigest()
print(h)                                    # 64-char hex

# Streaming for large files
h = hashlib.sha256()
with open("big.bin", "rb") as f:
    while chunk := f.read(64 * 1024):
        h.update(chunk)
print(h.hexdigest())
```

For:

- **File integrity**: SHA-256.
- **Content-addressed storage**: SHA-256 of the content.
- **Quick-lookup hashing** (where collisions don't matter): MD5 is fine despite cryptographic weaknesses.
- **Passwords**: NEVER SHA-256. Use a password-hashing function (bcrypt / argon2; see below).

## Password hashing — `bcrypt` or `argon2`

```python
import bcrypt


# Hash on registration
password = b"correct horse battery staple"
hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))
# Store `hashed` in DB

# Verify on login
if bcrypt.checkpw(password, hashed):
    print("ok")
```

`bcrypt` with `rounds=12` takes ~250ms per hash on modern hardware — slow on purpose, so brute-force is expensive.

For new code, **argon2** is preferred (winner of the Password Hashing Competition):

```python
from argon2 import PasswordHasher


ph = PasswordHasher()
hashed = ph.hash("correct horse battery staple")
ph.verify(hashed, "correct horse battery staple")     # ok
ph.verify(hashed, "wrong")                            # argon2.exceptions.VerifyMismatchError
```

**NEVER**:

- Use SHA-256 or any general-purpose hash for passwords.
- Implement your own "secure" password hashing.
- Hash without a per-user salt (`bcrypt`/`argon2` do this automatically; if you find yourself adding a global "pepper," reconsider).

## Symmetric encryption — `cryptography.fernet`

For encrypting blobs of data with a single shared key (configuration secrets, file contents):

```python
from cryptography.fernet import Fernet


# Generate a key (do this once; store securely)
key = Fernet.generate_key()
print(key.decode())                          # save this in secrets manager

f = Fernet(key)
ciphertext = f.encrypt(b"sensitive data")
plaintext = f.decrypt(ciphertext)
```

Fernet is AES-128-CBC + HMAC-SHA256, with versioning and a built-in timestamp. Use it instead of trying to construct AES yourself.

## Asymmetric encryption — `cryptography` library

For situations where you have public/private keypairs (TLS, signing):

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


# Generate keypair
private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
public_key = private_key.public_key()


# Sign
message = b"transferred $1000 to alice"
signature = private_key.sign(
    message,
    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
    hashes.SHA256(),
)


# Verify
public_key.verify(
    signature,
    message,
    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
    hashes.SHA256(),
)
```

For most "sign this thing" workflows, Ed25519 is a smaller, faster, modern alternative:

```python
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


sk = Ed25519PrivateKey.generate()
vk = sk.public_key()
sig = sk.sign(b"message")
vk.verify(sig, b"message")                   # raises if invalid
```

Ed25519 keys are 32 bytes; signatures are 64 bytes. Use it as the default for new code.

## Key serialisation

```python
from cryptography.hazmat.primitives import serialization


# Write private key to PEM
pem = sk.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.BestAvailableEncryption(b"my-passphrase"),
)
Path("key.pem").write_bytes(pem)


# Read back
key = serialization.load_pem_private_key(
    Path("key.pem").read_bytes(),
    password=b"my-passphrase",
)
```

PEM is the standard format for PKI files. Always encrypt private keys at rest with a passphrase.

## HMACs — message authentication codes

For "verify this message came from someone with the shared key":

```python
import hmac
import hashlib


def compute_signature(payload: bytes, secret: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def verify_signature(payload: bytes, secret: bytes, sig: str) -> bool:
    expected = compute_signature(payload, secret)
    return hmac.compare_digest(expected, sig)        # constant-time compare
```

`hmac.compare_digest` (vs `==`) prevents timing attacks. Use it always.

This is the pattern for webhook signature verification (Stripe, GitHub, Slack all use HMAC-SHA256).

## Random — cryptographic vs not

```python
import os
import secrets
import random


# CRYPTOGRAPHIC — for tokens, session IDs, passwords
token = secrets.token_urlsafe(32)            # URL-safe random string
hex_token = secrets.token_hex(32)
secrets_int = secrets.randbelow(1000)

# Lower-level
random_bytes = os.urandom(32)


# NON-CRYPTOGRAPHIC — for simulation, shuffling, picking randomly
random.seed(42)
random.randint(0, 99)
random.choice(["a", "b", "c"])
```

`random` is fast but predictable. `secrets` is slower but cryptographically secure. For anything security-relevant (API keys, session tokens, password generation), always `secrets`.

## A worked example: encrypted-at-rest config file

```python
from cryptography.fernet import Fernet
import json
import os
from pathlib import Path


def load_config(encrypted_path: Path) -> dict:
    key = os.environ["CONFIG_KEY"].encode()
    f = Fernet(key)
    encrypted = encrypted_path.read_bytes()
    return json.loads(f.decrypt(encrypted))


def save_config(config: dict, encrypted_path: Path):
    key = os.environ["CONFIG_KEY"].encode()
    f = Fernet(key)
    encrypted = f.encrypt(json.dumps(config).encode())
    encrypted_path.write_bytes(encrypted)


# Usage
config = load_config(Path("/etc/myapp/config.enc"))
print(config["api_keys"])
```

`CONFIG_KEY` is the only thing that goes into the secrets manager. The file on disk is just bytes. Compromise of the disk without the key reveals nothing.

## Pitfalls

!!! danger "Rolling your own"
    The temptation to implement "just a quick XOR cipher" or "a simple AES wrapper" leads to broken crypto. Always use a vetted library.

!!! danger "Hardcoded keys"
    Keys committed to git are public, even after a forced push. Read from env vars or secrets managers.

!!! danger "ECB mode"
    AES-ECB encrypts each block independently. Identical plaintext blocks produce identical ciphertext blocks — a visible pattern. Never use ECB. Fernet defaults to CBC; for new code use GCM or ChaCha20-Poly1305.

!!! danger "Reusing nonces / IVs"
    AES-GCM with a reused nonce catastrophically breaks. Use `os.urandom(12)` for each encryption.

!!! danger "Comparing hashes with `==`"
    Timing-attack vulnerability. Use `hmac.compare_digest(a, b)`.

## Bottom line

For crypto:

- **`hashlib` SHA-256** for integrity hashes.
- **`bcrypt` or `argon2`** for passwords; NEVER SHA-256.
- **`Fernet` from `cryptography`** for symmetric encryption.
- **Ed25519 from `cryptography`** for signing.
- **`hmac.compare_digest`** for any equality check on sensitive bytes.
- **`secrets`**, not `random`, for security-relevant random values.

Continue to **[JWT and OAuth](05-jwt-oauth.md)**.
