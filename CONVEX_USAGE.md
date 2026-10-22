# Convex free-tier usage controls

The supplied usage panel shows 1.08 GB / 1 GB data egress, 299.78 MB / 1 GB database I/O and 187.59 MB / 1 GB file storage. Egress is already above the displayed allowance. These changes reduce future work; they cannot erase transferred bytes or reset the billing window.

## Downloads and metadata

`sp500_ml/cloud_data.py` retains up to 128 MiB of verified **compressed** content in process RAM, rather than 32 MiB of decompressed data. The metadata inventory found 519 price files containing about 269 MiB uncompressed. Content hashes still select the version; both compressed and uncompressed checksums are verified on first download. Concurrent readers of the same content share a download. Confirmed uploads populate the cache after Convex validates their compressed checksum. Superseded content is retired after a successful replacement, preventing daily versions from evicting other stocks' live files.

Dataset metadata has a bounded 10-second, deployment/credential-scoped cache. Writes bypass cached metadata and invalidate it afterward, retaining optimistic version checks and same-process read-after-write freshness. Missing metadata and failed writes do not become successful cache entries. No disk fallback or local dataset is introduced.

Caches belong to each Python process. Restarting it loses them; frequent restarts and serverless cold starts still download cold data. Other-process changes can take up to 10 seconds to appear. The compressed cache can use up to 128 MiB of RAM plus transient decompression memory.

## Shared market scan

The discovery scanner fetches only the last 180 calendar days directly from Yahoo and evaluates up to 80 completed bars. It never reads or rewrites full Convex price files. Missing/latest-session bars fail the stock rather than falling back to a full cloud download. Full historical price synchronization remains in the existing search/update paths; a scan itself no longer updates that dataset. Yahoo rate limits can still leave stocks unavailable.

Three workers process batches of 15 stocks, then confirm one cloud checkpoint per batch. A 503-stock scan uses 34 checkpoint writes instead of 503. A crash can repeat at most one uncommitted batch; it does not invent progress. Existing renewable leases and confirmed-record checks remain in place.

Completed current-session scan reads are cached for up to one hour per backend process; active/old-session results for two seconds. The expected completed market session is checked on every access, so a new session invalidates the old cache immediately. Local rescan requests discard the cache; successful checkpoints publish the new cached snapshot. Manual rescans performed on a different backend instance can take up to one hour to appear here; the scan's update timestamp stays visible.

The browser polls completed scans every five minutes and active scans every 15 seconds. Hidden documents make no scan requests and check visibility again after one minute. Explicit rescans and retries remain available. Prediction algorithms and ownership rules are unchanged.

## Storage and operational limits

Existing unchanged-content upload deduplication and delayed removal of superseded blobs remain active. Historical model generations, reports, research and predictions are retained. Large offline evaluations and training still consume storage and bandwidth; run these deliberately rather than repeatedly. No paid tier or new paid service is enabled.

Restart the Python backend and refresh/rebuild the frontend to use these changes. Compare the **rate of new usage** on the Convex dashboard after restart; already accumulated usage remains. A single metadata-only inventory was used for this investigation; no full-market download benchmark was performed. These controls cannot guarantee staying under the free allowance across many instances, frequent cold starts or repeated bulk training.
