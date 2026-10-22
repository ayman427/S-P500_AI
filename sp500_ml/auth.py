"""Verify Clerk/Convex integration tokens before any web API operation."""
from dataclasses import dataclass, field
from functools import lru_cache
import os
import logging
from urllib.parse import urlparse

import jwt
from jwt import PyJWKClient

from .prediction_store import load_environment


class AuthenticationError(Exception):
    def __init__(self, message="Sign in to access the platform.", status=401):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    token: str = field(repr=False)


@lru_cache(maxsize=4)
def _keys(issuer):
    return PyJWKClient(issuer + "/.well-known/jwks.json", cache_jwk_set=True, lifespan=300, timeout=10)


def authenticate(authorization):
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError()
    token = authorization[7:].strip()
    if not token or len(token) > 16384:
        raise AuthenticationError("Invalid sign-in token.")
    load_environment()
    issuer = os.environ.get("CLERK_JWT_ISSUER_DOMAIN", "").rstrip("/")
    parsed = urlparse(issuer)
    if parsed.scheme != "https" or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise AuthenticationError("Sign-in is not configured on the server.", 503)
    origins = {part.strip() for part in os.environ.get("CLERK_AUTHORIZED_PARTIES", "http://127.0.0.1:5173,http://localhost:5173").split(",") if part.strip()}
    try:
        key = _keys(issuer).get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=issuer, audience="convex", leeway=5,
                            options={"require": ["sub", "iss", "aud", "iat", "nbf", "exp", "azp"]})
    except jwt.ExpiredSignatureError as error:
        raise AuthenticationError("Your sign-in session expired. Sign in again.") from error
    except jwt.ImmatureSignatureError as error:
        raise AuthenticationError("The sign-in token is not valid yet. Check your computer's date and time, then sign in again.") from error
    except jwt.InvalidIssuerError as error:
        raise AuthenticationError("The frontend and server are using different Clerk applications. Check the Clerk configuration.") from error
    except jwt.InvalidAudienceError as error:
        raise AuthenticationError("The sign-in token has the wrong audience. Check Clerk's Convex integration settings.") from error
    except jwt.MissingRequiredClaimError as error:
        messages = {"aud": "The sign-in token has no Convex audience. Refresh your session after enabling Clerk's Convex integration.",
                    "azp": "The sign-in token has no authorized origin. Sign in from the configured application URL."}
        raise AuthenticationError(messages.get(error.claim, "The sign-in token is missing required verification claims. Sign in again.")) from error
    except jwt.PyJWKClientConnectionError as error:
        raise AuthenticationError("The server cannot reach Clerk to verify sign-in. Please try again shortly.", 503) from error
    except (jwt.PyJWTError, ValueError, TypeError) as error:
        logging.getLogger(__name__).warning("Clerk authentication rejected (%s)", type(error).__name__)
        raise AuthenticationError("Your sign-in session could not be verified. Sign in again.") from error
    if not isinstance(claims["sub"], str) or not claims["sub"] or not isinstance(claims["azp"], str) or claims["azp"] not in origins or claims.get("sts") == "pending":
        raise AuthenticationError("Your sign-in session is not authorized for this app.")
    return AuthenticatedUser(user_id=claims["sub"], token=token)
