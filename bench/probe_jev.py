"""Record live System One responses as contract-test fixtures (§14.7).

Run this deliberately, never from the test suite: §14.2 forbids network access
in every tier, and CI runs with networking disabled. Its job is to capture what
the API really returns so the fixtures under `tests/contract/fixtures/jev/`
describe reality rather than someone's recollection of it.

    uv run python bench/probe_jev.py            # record fixtures
    uv run python bench/probe_jev.py --explore  # probe schema variants too

Secrets are scrubbed: only response bodies are written, never headers, and the
key is read from the environment or `.env` and never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from scruple.backends.jev import (
    API_KEY_ENV,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    LEGACY_API_KEY_ENV,
)
from scruple.env import load_env

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "contract" / "fixtures" / "jev"

STATE = "The clinic closes at five and I work two jobs, so I never managed to go."
ACCESS = (
    "The respondent cites a practical obstacle to attending: time, transport, "
    "clinic opening hours, distance or cost."
)
DISTRUST = "The respondent expresses distrust of pharmaceutical companies or their motives."

# Gateway routing metadata carries ids and timings that change every call and
# would make the fixtures churn, so it is dropped rather than recorded.
VOLATILE_METADATA = ("gateway",)


def credentials() -> tuple[str, str, str]:
    import os

    load_env(ROOT)
    key = os.environ.get(API_KEY_ENV) or os.environ.get(LEGACY_API_KEY_ENV)
    if not key:
        print(f"no {API_KEY_ENV} in the environment or .env -- nothing to probe", file=sys.stderr)
        raise SystemExit(2)
    endpoint = os.environ.get("JEV_BASE_URL") or DEFAULT_ENDPOINT
    model = os.environ.get("JEV_MODEL") or DEFAULT_MODEL
    return key, endpoint, model


def scrub(body: dict[str, object]) -> dict[str, object]:
    metadata = body.get("provider_metadata")
    if isinstance(metadata, dict):
        body["provider_metadata"] = {
            k: v for k, v in metadata.items() if k not in VOLATILE_METADATA
        }
    return body


def call(client: httpx.Client, endpoint: str, key: str, payload: dict[str, object]):  # type: ignore[no-untyped-def]
    return client.post(
        endpoint,
        json=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=120.0,
    )


def record(name: str, response: httpx.Response) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    body = scrub(response.json())
    (FIXTURES / f"{name}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"  recorded {name}.json (HTTP {response.status_code})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explore", action="store_true", help="probe schema variants as well")
    args = parser.parse_args()

    key, endpoint, model = credentials()
    print(f"probing {endpoint} with model {model}")

    with httpx.Client() as client:
        two_codes = {
            "model": model,
            "state": STATE,
            "questions": {
                "access_barrier": {"type": "noul", "instructions": ACCESS},
                "distrust_pharma": {"type": "noul", "instructions": DISTRUST},
            },
        }
        response = call(client, endpoint, key, two_codes)
        response.raise_for_status()
        record("two_codes", response)
        answers = response.json()["answers"]
        print(
            f"  access_barrier={answers['access_barrier']['noul']}  "
            f"distrust_pharma={answers['distrust_pharma']['noul']}"
        )

        # A deliberately invalid request, so the fixtures cover the error shape.
        bad = {
            "model": model,
            "state": STATE,
            "questions": {"access_barrier": {"type": "Noul", "question": ACCESS}},
        }
        record("error_bad_question_type", call(client, endpoint, key, bad))

        if args.explore:
            print("\nschema exploration:")
            variants = {
                "questions as a list": {
                    "model": model,
                    "state": STATE,
                    "questions": [{"id": "access_barrier", "question": ACCESS}],
                },
                "bare question string": {
                    "model": model,
                    "state": STATE,
                    "questions": {"access_barrier": {"type": "noul", "question": ACCESS}},
                },
                "criteria as a list": {
                    "model": model,
                    "state": STATE,
                    "questions": {"access_barrier": {"type": "noul", "criteria": [ACCESS]}},
                },
            }
            for label, payload in variants.items():
                result = call(client, endpoint, key, payload)
                body = result.json()
                message = body.get("message") or (body.get("error") or {}).get("message")
                print(f"  {label}: HTTP {result.status_code} :: {str(message)[:110]}")


if __name__ == "__main__":
    main()
