"""SEC-only DNS recovery with the original hostname, SNI, and TLS verification."""
import ipaddress
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

_addresses = {}
SEC_HOSTS = {"data.sec.gov", "www.sec.gov"}


def public_sec_address(host):
    if host not in SEC_HOSTS:
        raise requests.ConnectionError("DNS recovery is restricted to SEC hosts")
    cached = _addresses.get(host)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    try:
        response = requests.get("https://dns.google/resolve", params={"name": host, "type": "A"}, timeout=8)
        response.raise_for_status()
        payload = response.json()
        if payload.get("Status") != 0:
            raise ValueError("DNS lookup failed")
        for answer in payload.get("Answer", []):
            if answer.get("type") == 1:
                address = ipaddress.ip_address(answer["data"])
                if address.version == 4 and address.is_global and not address.is_multicast:
                    _addresses[host] = (time.monotonic() + min(300, max(1, int(answer.get("TTL", 60)))), str(address))
                    return str(address)
    except (ValueError, TypeError, KeyError, AttributeError, requests.RequestException):
        pass
    raise requests.ConnectionError("SEC DNS recovery failed")


class SecDNSAdapter(HTTPAdapter):
    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        host = urlparse(request.url).hostname
        if host not in SEC_HOSTS or verify is False:
            raise requests.ConnectionError("SEC hostname and certificate verification are required")
        if requests.utils.select_proxy(request.url, proxies):
            return super().get_connection_with_tls_context(request, verify, proxies, cert)
        host_params, pool_kwargs = self.build_connection_pool_key_attributes(request, verify, cert)
        host_params["host"] = public_sec_address(host)
        pool_kwargs.update(server_hostname=host, assert_hostname=host)
        request.headers["Host"] = host
        return self.poolmanager.connection_from_host(**host_params, pool_kwargs=pool_kwargs)
