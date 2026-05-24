# JWT and OAuth

For modern API authentication, two standards dominate: **JWT** (JSON Web Tokens) as the credential format, and **OAuth 2.0** as the protocol for obtaining them. This chapter is the working Python implementation.

## JWT — the format

A JWT is three base64-encoded JSON blobs separated by dots:

```
eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhbGljZSIsImV4cCI6MTcwMDAwMDAwMH0.signature
[ header ].[ payload ].[ signature ]
```

- **Header**: the algorithm used.
- **Payload**: the claims — user id, expiry, custom data.
- **Signature**: HMAC or RSA-signed; proves the token is authentic.

JWTs are **not encrypted by default** — the payload is just base64. Anyone can read it. Use HTTPS for transport. For encrypted tokens, JWE (JSON Web Encryption).

## Signing and verifying with `pyjwt`

```python
import jwt
from datetime import datetime, timedelta, timezone


SECRET = "super-secret-key"


# Create
payload = {
    "sub": "alice",
    "iat": datetime.now(timezone.utc),
    "exp": datetime.now(timezone.utc) + timedelta(hours=1),
}
token = jwt.encode(payload, SECRET, algorithm="HS256")
print(token)                                       # eyJhbGc...


# Verify
try:
    decoded = jwt.decode(token, SECRET, algorithms=["HS256"])
    print(decoded)
except jwt.ExpiredSignatureError:
    print("expired")
except jwt.InvalidTokenError:
    print("invalid")
```

`exp` is checked automatically. Always include it; tokens without expiry are forever-valid credentials.

## Asymmetric signing — RS256, ES256

For tokens issued by one service and verified by many, asymmetric is right: signing service has the private key; verifiers only need the public key.

```python
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization


# Generate (once)
sk = rsa.generate_private_key(public_exponent=65537, key_size=4096)
pk_pem = sk.public_key().public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)
sk_pem = sk.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
)


# Sign
token = jwt.encode({"sub": "alice", "exp": ...}, sk_pem, algorithm="RS256")


# Verify (with just the public key)
decoded = jwt.decode(token, pk_pem, algorithms=["RS256"])
```

For modern use, Ed25519 (algorithm `EdDSA`) is preferred — smaller keys, faster signatures.

## Validation patterns

```python
import jwt
from jwt import InvalidTokenError


def validate_token(token: str, secret: str, audience: str = None, issuer: str = None) -> dict:
    try:
        return jwt.decode(
            token, secret,
            algorithms=["HS256"],
            audience=audience,
            issuer=issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("token expired")
    except jwt.InvalidAudienceError:
        raise ValueError("wrong audience")
    except InvalidTokenError as exc:
        raise ValueError(f"invalid token: {exc}")
```

**Always** validate:

- Signature (automatic with `decode`).
- Expiry (`exp`).
- Audience (`aud`) — that the token was issued for *your* service.
- Issuer (`iss`) — that the token came from a trusted issuer.

## Standard claims

| Claim | Meaning |
|---|---|
| `iss` | issuer |
| `sub` | subject (usually user ID) |
| `aud` | audience (which service the token is for) |
| `exp` | expiration timestamp |
| `nbf` | not before |
| `iat` | issued at |
| `jti` | unique token ID (for revocation lists) |

Custom claims go alongside — `{"sub": "alice", "iss": "auth.example.com", "roles": ["admin"]}`.

## Refresh tokens

Access tokens should be short-lived (15-60 min). Refresh tokens are longer-lived and used to obtain new access tokens:

```python
def issue_token_pair(user_id: str) -> dict:
    access = jwt.encode({
        "sub": user_id, "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
    }, ACCESS_SECRET, algorithm="HS256")
    refresh = jwt.encode({
        "sub": user_id, "type": "refresh",
        "exp": datetime.now(timezone.utc) + timedelta(days=30),
        "jti": str(uuid.uuid4()),
    }, REFRESH_SECRET, algorithm="HS256")
    return {"access_token": access, "refresh_token": refresh}


def refresh_access(refresh_token: str) -> str:
    decoded = jwt.decode(refresh_token, REFRESH_SECRET, algorithms=["HS256"])
    if decoded["type"] != "refresh":
        raise ValueError("not a refresh token")
    if is_revoked(decoded["jti"]):
        raise ValueError("revoked")
    return jwt.encode({
        "sub": decoded["sub"], "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
    }, ACCESS_SECRET, algorithm="HS256")
```

Refresh tokens are stored / managed differently from access tokens — typically server-side with revocation support.

## OAuth 2.0 flows

For real OAuth, use `authlib`:

```python
from authlib.integrations.requests_client import OAuth2Session


client = OAuth2Session(client_id="...", client_secret="...", scope="read:user")
authorization_url, state = client.create_authorization_url("https://github.com/login/oauth/authorize")

print(f"Visit: {authorization_url}")
# User visits, authorises, redirects back with ?code=...
code = "..."
token = client.fetch_token(
    "https://github.com/login/oauth/access_token",
    code=code,
)
print(token)                                     # access_token, refresh_token, etc.
```

For server-side OAuth flows (Authorization Code with PKCE), authlib is the standard.

The four main OAuth 2.0 flows:

| Flow | Use case |
|---|---|
| **Authorization Code (with PKCE)** | web apps, mobile apps |
| **Client Credentials** | service-to-service |
| **Device Code** | TVs, CLIs without browser |
| **Implicit / Password** | DEPRECATED; don't use |

For new code: authorization-code+PKCE for user auth; client-credentials for service auth.

## OpenID Connect — JWT + OAuth combined

OIDC adds an `id_token` (a JWT containing user identity) on top of OAuth 2.0. The pattern for "log in with Google":

1. User redirects to Google with OAuth params.
2. Google authenticates, redirects back with auth code.
3. Your server exchanges the code for tokens (including `id_token`).
4. Verify the `id_token`'s signature against Google's published JWKS.
5. Extract user identity from the verified token.

```python
import jwt
import httpx


JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
ISSUER = "https://accounts.google.com"
AUDIENCE = "your-client-id.apps.googleusercontent.com"


def get_jwks():
    return httpx.get(JWKS_URL).json()


def verify_google_id_token(id_token: str) -> dict:
    # In real code, use python-jose or authlib for proper JWKS handling
    jwks = get_jwks()
    header = jwt.get_unverified_header(id_token)
    matching_key = next(k for k in jwks["keys"] if k["kid"] == header["kid"])
    key = jwt.algorithms.RSAAlgorithm.from_jwk(matching_key)
    return jwt.decode(id_token, key, algorithms=["RS256"], audience=AUDIENCE, issuer=ISSUER)
```

For production, prefer libraries that handle JWKS rotation:

```python
from jwt import PyJWKClient


client = PyJWKClient(JWKS_URL)
signing_key = client.get_signing_key_from_jwt(id_token)
decoded = jwt.decode(id_token, signing_key.key, algorithms=["RS256"],
                     audience=AUDIENCE, issuer=ISSUER)
```

## Token blacklist / revocation

JWT's strength (stateless verification) is also its weakness (can't unilaterally revoke). Two patterns:

1. **Short access tokens + revocable refresh tokens** — server tracks valid `jti` for refresh tokens.
2. **Token blacklist** — for every access token, the verifier checks a Redis cache of recently-revoked `jti`s.

Both add latency / state. For high-security applications, accept the cost.

## Pitfalls

!!! danger "Algorithm confusion attacks"
    Old `pyjwt` versions accepted "none" or "HS256" when the server expected "RS256". Always pass `algorithms=["RS256"]` explicitly; never `algorithms=None`.

!!! danger "Long-lived JWTs"
    A token with 1-week expiry that's leaked is a 1-week breach. Short access tokens (15-60 min) limit blast radius.

!!! danger "Storing JWTs in localStorage"
    JS can read localStorage; XSS exfiltrates tokens. For browsers, use HttpOnly cookies.

!!! danger "Trusting unverified claims"
    Decode-without-verify is for inspection only. `jwt.decode(..., options={"verify_signature": False})` should never be in your auth path.

## Bottom line

For JWT/OAuth:

- **`pyjwt`** for signing/verifying JWTs.
- **`authlib`** for OAuth flows.
- **Always validate `exp`, `aud`, `iss`** in production.
- **Short access tokens + revocable refresh tokens** for the standard pattern.
- **RS256 / EdDSA** for distributed verification; HS256 for single-secret.

Continue to **[Geospatial](06-geospatial.md)**.
