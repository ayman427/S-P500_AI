import asyncio
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
import threading
from http.client import HTTPConnection
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from sp500_ml.auth import authenticate, AuthenticationError, AuthenticatedUser


class ClerkTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def setUp(self):
        self.config = patch.dict(os.environ, {"CLERK_JWT_ISSUER_DOMAIN": "https://issuer.example", "CLERK_AUTHORIZED_PARTIES": "http://localhost:5173"})
        self.config.start()
        self.addCleanup(self.config.stop)
        self.env = patch("sp500_ml.auth.load_environment")
        self.env.start()
        self.addCleanup(self.env.stop)
        self.keys = patch("sp500_ml.auth._keys")
        self.jwks = self.keys.start()
        self.jwks.return_value.get_signing_key_from_jwt.return_value = SimpleNamespace(key=self.key.public_key())
        self.addCleanup(self.keys.stop)

    def token(self, **changes):
        now = int(datetime.now(timezone.utc).timestamp())
        claims = dict(sub="alice", iss="https://issuer.example", aud="convex", iat=now, nbf=now-1, exp=now+60, azp="http://localhost:5173")
        claims.update(changes)
        return jwt.encode(claims, self.key, algorithm="RS256")

    def test_valid_token_has_verified_identity_and_does_not_log_token(self):
        token = self.token()
        user = authenticate("Bearer " + token)
        self.assertEqual(user.user_id, "alice")
        self.assertNotIn(token, repr(user))

    def test_invalid_expiry_issuer_audience_origin_and_pending_session_are_denied(self):
        for change in [{"exp": 1}, {"iss": "https://attacker.example"}, {"aud": "other"}, {"azp": "https://attacker.example"}, {"sts": "pending"}, {"azp": []}]:
            with self.subTest(change=change), self.assertRaises(AuthenticationError):
                authenticate("Bearer " + self.token(**change))

    def test_missing_token_and_wrong_signature_are_denied(self):
        with self.assertRaises(AuthenticationError): authenticate(None)
        attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        fake = jwt.encode(jwt.decode(self.token(), options={"verify_signature": False}), attacker, algorithm="RS256")
        with self.assertRaises(AuthenticationError): authenticate("Bearer " + fake)

    def test_only_explicitly_configured_tunnel_origin_is_accepted(self):
        origin = "https://lk56gfw0-5173.usw3.devtunnels.ms"
        with patch.dict(os.environ, {"CLERK_AUTHORIZED_PARTIES": "http://localhost:5173," + origin}):
            self.assertEqual(authenticate("Bearer " + self.token(azp=origin)).user_id, "alice")
            with self.assertRaises(AuthenticationError):
                authenticate("Bearer " + self.token(azp="https://another-tunnel.usw3.devtunnels.ms"))

    def test_verification_errors_explain_audience_and_clock_rejections(self):
        claims = jwt.decode(self.token(), options={"verify_signature": False})
        claims.pop("aud")
        with self.assertRaisesRegex(AuthenticationError, "no Convex audience"):
            authenticate("Bearer " + jwt.encode(claims, self.key, algorithm="RS256"))
        with self.assertRaisesRegex(AuthenticationError, "expired"):
            authenticate("Bearer " + self.token(exp=1))


async def invoke_asgi(method, path, body=None, authorization=None):
    from vercel_app import app
    messages = []
    headers = [(b"content-type", b"application/json")]
    if authorization: headers.append((b"authorization", authorization.encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "", "query_string": b"",
             "headers": headers, "client": ("127.0.0.1", 1), "server": ("test", 80)}
    async def receive(): return {"type": "http.request", "body": json.dumps(body or {}).encode(), "more_body": False}
    async def send(message): messages.append(message)
    await app(scope, receive, send)
    return next(m["status"] for m in messages if m["type"] == "http.response.start")


class ApiAuthorizationTests(unittest.TestCase):
    def test_every_data_route_requires_sign_in_before_any_work(self):
        for method, path in [("GET", "/api/stocks"), ("GET", "/api/state"), ("GET", "/api/quote"), ("GET", "/api/risk"), ("GET", "/api/spikes"), ("GET", "/api/risk/news"), ("GET", "/api/risk/analysis"), ("GET", "/api/predictions"), ("POST", "/api/predictions"), ("POST", "/api/prices/refresh"), ("POST", "/api/action/train")]:
            with self.subTest(path=path):
                self.assertEqual(asyncio.run(invoke_asgi(method, path)), 401)

    def test_authentication_does_not_grant_access_to_legacy_paper_signals(self):
        with patch("vercel_app.authenticate", return_value=AuthenticatedUser("alice", "token")), patch("dashboard_server._read_dashboard_state", return_value={}) as state:
            self.assertEqual(asyncio.run(invoke_asgi("GET", "/api/state")), 200)
            self.assertFalse(state.call_args.kwargs["include_paper"])
            self.assertFalse(state.call_args.args[3])
            self.assertEqual(asyncio.run(invoke_asgi("POST", "/api/action/paper-trade")), 403)

    def test_local_http_server_enforces_the_same_authentication_boundary(self):
        import dashboard_server
        server = dashboard_server.DashboardServer(("127.0.0.1", 0), dashboard_server.DashboardHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(dashboard_server, "_read_stock_directory", return_value={"stocks": []}) as stocks:
                client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                client.request("GET", "/api/stocks")
                response = client.getresponse()
                self.assertEqual(response.status, 401)
                response.read()
                stocks.assert_not_called()
                client.close()
                with patch.object(dashboard_server, "authenticate", return_value=AuthenticatedUser("alice", "token")):
                    client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                    client.request("GET", "/api/stocks")
                    response = client.getresponse()
                    self.assertEqual(response.status, 200)
                    response.read()
                    client.close()
                    stocks.assert_called_once()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
