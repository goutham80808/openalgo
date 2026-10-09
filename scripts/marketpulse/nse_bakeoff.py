"""One-shot bake-off: can the library fetch the five file classes we need?

Run for each candidate (after `uv add <lib>`):
    uv run python scripts/marketpulse/nse_bakeoff.py <lib>
Records PASS/FAIL per capability. The winner goes into the Phase 0 handoff.

Capabilities checked (one recent trading day, real network calls):
    1. NSE bhavcopy with delivery (sec_bhavdata_full, whole market)
    2. BSE deliverable bhavcopy
    3. participant-wise F&O OI
    4. FII/DII flows
    5. block/bulk deals

Verified results from the Phase 0 session (2026-10-10) are in the phase-0
handoff; the winning combination ships in services/marketpulse_jobs/nse_client.py.

This is a spike, not product code: every network call lives here.
"""

import datetime as dt
import io
import sys

# Friday 2026-10-09 first, then step back over trading days in case the
# latest file has not landed yet.
CANDIDATE_DATES = ["2026-10-09", "2026-10-08", "2026-10-07", "2026-10-06"]


def _dates():
    return [dt.date.fromisoformat(d) for d in CANDIDATE_DATES]


def _csv_rows(text: str) -> int:
    import pandas as pd

    frame = pd.read_csv(io.StringIO(text))
    return len(frame)


def _archives_bhavcopy(day: dt.date):
    """Whole-market sec_bhavdata_full CSV from the NSE archives host.

    This is exactly the GET jugaad-data's full_bhavcopy_raw performs: a
    stable public URL that needs no cookies, so it doubles as the fallback
    for libraries that cannot fetch the file.
    """
    import requests

    url = (
        "https://nsearchives.nseindia.com/products/content/"
        f"sec_bhavdata_full_{day.strftime('%d%m%Y')}.csv"
    )
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def _first_csv(fetch) -> tuple[int, str] | None:
    for day in _dates():
        try:
            text = fetch(day)
            if text and len(text) > 200:
                return _csv_rows(text), day.isoformat()
        except Exception:
            continue
    return None


def run_jugaad() -> dict:
    from jugaad_data.nse import NSEArchives, full_bhavcopy_raw

    archives = NSEArchives()
    results = {}

    got = _first_csv(full_bhavcopy_raw)
    results["bhavcopy NSE w/ delivery"] = f"PASS rows={got[0]} date={got[1]}" if got else "FAIL"

    got = _first_csv(lambda day: archives.bulk_deals_raw())
    results["bulk deals"] = f"PASS rows={got[0]} (current-day only, no date arg)" if got else "FAIL"
    results["block deals"] = "FAIL (no block_deals_raw)"
    results["BSE deliverable bhavcopy"] = "FAIL (no BSE archives module)"
    results["participant-wise F&O OI"] = "FAIL (not in archives report list)"
    results["FII/DII flows"] = "FAIL (not exposed)"
    return results


def run_nsepython() -> dict:
    results = {}

    got = _first_csv(_archives_bhavcopy)
    results["bhavcopy NSE w/ delivery"] = (
        f"PASS rows={got[0]} date={got[1]} (direct archives GET; "
        "nsepython's get_bhavcopy is broken in 2.101)"
        if got
        else "FAIL"
    )

    def _json_rows(fetch) -> tuple[int, str] | None:
        for day in _dates():
            try:
                payload = fetch(day)
                rows = (
                    len(payload.get("data", payload)) if isinstance(payload, dict) else len(payload)
                )
                if rows:
                    return rows, day.isoformat()
            except Exception:
                continue
        return None

    got = _json_rows(
        lambda day: __import__("nsepython").get_fao_participant_oi(day.strftime("%d-%m-%Y"))
    )
    results["participant-wise F&O OI"] = f"PASS rows={got[0]} date={got[1]}" if got else "FAIL"

    try:
        import nsepython

        frame = nsepython.nse_fiidii()
        results["FII/DII flows"] = f"PASS rows={len(frame)} (latest days)"
    except Exception as exc:  # noqa: BLE001 - spike reports, never crashes
        results["FII/DII flows"] = f"FAIL ({type(exc).__name__})"

    def _deals(mode: str) -> str:
        try:
            import nsepython

            frame = nsepython.nse_largedeals_historical("09-10-2026", "09-10-2026", mode=mode)
            return f"PASS rows={len(frame)} (dated range)"
        except Exception as exc:  # noqa: BLE001 - spike reports, never crashes
            return f"FAIL ({type(exc).__name__}: {str(exc)[:60]})"

    results["bulk deals"] = _deals("bulk_deals")
    results["block deals"] = _deals("block_deals")
    results["BSE deliverable bhavcopy"] = (
        "FAIL (bseindia.com download host serves its SPA page to plain requests)"
    )
    return results


def main(lib: str) -> int:
    try:
        if lib == "jugaad-data":
            results = run_jugaad()
        elif lib == "nsepython":
            results = run_nsepython()
        else:
            print(f"unknown lib {lib}")
            return 2
    except Exception as exc:  # noqa: BLE001 - spike reports, never crashes
        results = {"import": f"FAILED: {exc}"}
    for capability, ok in results.items():
        print(f"{capability}: {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
