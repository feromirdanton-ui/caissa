"""A single HTTP identity for every outbound request (Article 14, session-0005).

The Week In Chess serves its weekly issue zips behind a filter that answers **HTTP 406
Not Acceptable** to the default ``python-requests`` User-Agent. That filter stayed invisible
for the whole build because the local test fixture — like a plain static file server — answers
every client; the first run on a clean runner (2026-10-11) failed with an empty probe window.
``HEAD twic1524g.zip`` from the sandbox reproduced the 406, and the same request carrying a
descriptive User-Agent returned ``200 OK`` with ``Last-Modified``.

Every loader therefore identifies itself the same way, and this constant is the one place that
identity lives — so a future site that filters on the string is a one-line change, not a hunt.
"""

from __future__ import annotations

USER_AGENT = "Caissa-1/0.2.0 (+https://github.com/feromirdanton-ui/caissa)"
