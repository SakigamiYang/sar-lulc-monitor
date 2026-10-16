import socket
import time

import requests
import urllib3.util.connection as connection

# Diagnostic only: force urllib3 to use IPv4.
connection.allowed_gai_family = lambda: socket.AF_INET

for url in [
    "https://oauth2.googleapis.com/token",
    "https://earthengine.googleapis.com/",
]:
    start = time.monotonic()

    try:
        response = requests.get(url, timeout=(5, 10))
        print(
            url,
            response.status_code,
            f"{time.monotonic() - start:.2f}s",
            flush=True,
        )
    except requests.RequestException as exc:
        print(type(exc).__name__, str(exc), flush=True)
