# aes / nangate45 — committed, but NOT usable as evidence

`results_funnel_campaigns.jsonl` here is kept for provenance only. Do not cite it,
mine it, or use it to compare policies. Two independent problems:

**1. 444 of 446 F3 builds are `PARSE_FAIL`.** Every one has all metrics `null`
and a reward pinned at the −20 failure-ladder penalty. The campaign burned
3.75 h (plus 0.46 h in the companion `campaign_0_1782700126`, 60/61 failed) and
produced **zero usable data points**. Any statistic computed over these rows —
reward distribution, knob correlation, best-config — is a statistic about the
failure penalty, not about silicon.

**2. TinyMAC field contamination.** The `obs` rows carry `"lanes": 4,
"acc_w": 24` on an AES design. That is the bug fixed in `f661fad` ("stop
fabricating TinyMAC baselines for generic-design campaigns"); this log predates
the fix and still carries the fabricated fields.

It is also `reward_version: 1` (pre audit-F18), so even the two non-failing rows
were scored on raw sampled-clock power.

If you want real aes/nangate45 numbers, re-run on the current stack. The
aes/asap7 log in the sibling directory is a genuine (if small: 7 F3 builds)
campaign and is the better starting point.
