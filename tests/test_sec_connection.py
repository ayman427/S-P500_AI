import socket
import ssl
import unittest
from unittest.mock import Mock, patch

import requests

from sp500_ml.sec_connection import public_sec_address, SecDNSAdapter, _addresses
from sp500_ml.primary_evidence import _download


class SecConnectionTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(_addresses, {}, clear=True); patcher.start(); self.addCleanup(patcher.stop)

    def test_public_dns_is_sec_only_cached_and_never_receives_sec_headers(self):
        response = Mock(); response.json.return_value = {"Status": 0, "Answer": [{"type": 1, "data": "23.11.0.58", "TTL": 60}]}
        with patch("sp500_ml.sec_connection.requests.get", return_value=response) as get:
            self.assertEqual(public_sec_address("data.sec.gov"), "23.11.0.58")
            self.assertEqual(public_sec_address("data.sec.gov"), "23.11.0.58")
            get.assert_called_once()
            self.assertNotIn("headers", get.call_args.kwargs)
            with self.assertRaises(requests.ConnectionError): public_sec_address("attacker.example")

    def test_dns_rejects_private_multicast_and_invalid_replies(self):
        for value in ("127.0.0.1", "10.0.0.1", "224.0.0.1", "not-an-address"):
            response = Mock(); response.json.return_value = {"Status": 0, "Answer": [{"type": 1, "data": value}]}
            with patch("sp500_ml.sec_connection.requests.get", return_value=response):
                with self.assertRaises(requests.ConnectionError): public_sec_address("data.sec.gov")

    def test_adapter_preserves_original_hostname_and_certificate_verification(self):
        request = requests.Request("GET", "https://data.sec.gov/submissions/CIK0000320193.json").prepare()
        adapter = SecDNSAdapter()
        with patch("sp500_ml.sec_connection.public_sec_address", return_value="23.11.0.58"), patch.object(adapter.poolmanager, "connection_from_host") as pool:
            adapter.get_connection_with_tls_context(request, True)
            args = pool.call_args.kwargs
            self.assertEqual(args["host"], "23.11.0.58")
            self.assertEqual(args["pool_kwargs"]["server_hostname"], "data.sec.gov")
            self.assertEqual(args["pool_kwargs"]["assert_hostname"], "data.sec.gov")
            if "ssl_context" in args["pool_kwargs"]:
                self.assertEqual(args["pool_kwargs"]["ssl_context"].verify_mode, ssl.CERT_REQUIRED)
            self.assertEqual(request.headers["Host"], "data.sec.gov")
            with self.assertRaises(requests.ConnectionError): adapter.get_connection_with_tls_context(request, False)
        adapter.close()

    def test_download_recovers_only_when_the_system_resolver_fails(self):
        response = Mock(status_code=200, is_redirect=False); response.iter_content.return_value = [b"verified response"]
        with patch("sp500_ml.primary_evidence.requests.get", side_effect=requests.ConnectionError), patch("sp500_ml.primary_evidence.socket.getaddrinfo", side_effect=socket.gaierror), patch("sp500_ml.primary_evidence.requests.Session") as session:
            session.return_value.__enter__.return_value.get.return_value.__enter__.return_value = response
            self.assertEqual(_download("https://data.sec.gov/submissions/CIK0000320193.json", {}), b"verified response")
            session.return_value.__enter__.return_value.mount.assert_called_once()


if __name__ == "__main__": unittest.main()
