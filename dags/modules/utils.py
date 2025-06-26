import ssl
import time
import socket
import urllib.request
import requests
import time
from datetime import datetime
import os 
from airflow.models import DagRun
from airflow.utils.state import State
from airflow.utils.session import provide_session


def fetch_url_body(link: str, max_bytes=5_000_000, max_duration=10):
    """
    Fetches up to 'max_bytes' of the response body from the given URL or until
    'max_duration' seconds have elapsed, whichever comes first.
    """
    os.environ['NO_PROXY'] = '*'

    start_time = time.time()
    context = ssl._create_unverified_context()
    try:
        req = urllib.request.Request(
            link,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
                )
            },
        )
        # You can still specify a socket timeout here (connect + read),
        # but we'll also do our own time check inside the chunk-reading loop:
        with urllib.request.urlopen(req, context=context, timeout=10) as response:
            data_chunks = []
            chunk_size = 1024 * 1024  # 1 MB
            total_read = 0

            while True:
                # Check if we've run longer than max_duration:
                if time.time() - start_time > max_duration:
                    print(f"Time limit of {max_duration}s reached for {link}, truncating.")
                    break

                chunk = response.read(chunk_size)
                if not chunk:
                    break

                data_chunks.append(chunk)
                total_read += len(chunk)

                if total_read >= max_bytes:
                    print(f"Reached max {max_bytes} bytes for {link}, truncating.")
                    break

            return b"".join(data_chunks)
    except (urllib.error.URLError, socket.timeout) as e:
        print(f"Failed to fetch article: {e}")
        return None
    except Exception as e:
        print(f"Unknown error: {e}")
        return None



@provide_session
def get_last_dag_run_date(dag_id="etis_dag", session=None):
    last_run = (
        session.query(DagRun)
        .filter(DagRun.dag_id == dag_id, DagRun.state == State.SUCCESS)
        .order_by(DagRun.execution_date.desc())
        .first()
    )
    if last_run:
        return last_run.execution_date.strftime('%Y-%m-%d')
    return None