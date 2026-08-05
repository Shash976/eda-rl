# campaigns/ — committed campaign logs

One directory per `<design>/<platform>`. Two kinds of file live here:

| file | what |
|---|---|
| `results_funnel_campaigns.jsonl` | the canonical log `eda-rl optimize` appends to, and what `--design/--platform` resolves. Committed **only where it is small**. |
| `campaign_records.jsonl` | a distilled, committable extract of a full log that was too large to commit. Same row schema, fewer rows — read it with `--log`. |

## Why `campaign_records.jsonl` exists

The npuload / tensordyne / nangate45 checkdesigns campaigns produced 185 MB of
log across three files (92 MB for tensordyne alone). That is far past what
belongs in git — GitHub warns over 50 MB and hard-rejects over 100 MB, and every
future clone would pay the cost forever. But 99.9% of that bulk is F0 legality
checks and F2 synthesis-proxy rows; the **F3 rows — the real RTL→GDS builds, the
actual performance records — are 117 rows totalling 0.2 MB**.

So each full log was distilled to `campaign_records.jsonl`, keeping in original
line order:

1. **every `campaign_summary` row** — per-campaign stats (F3 yield, kills,
   elapsed, sampler/promotion/seed).
2. **every F3 row from every campaign** — no build record is dropped.
3. **every row of one "showcase" campaign** — full F0/F2/F3 funnel detail so the
   funnel and kill-distribution charts still render. The showcase is the
   campaign with the most F3 builds among those under 7 MB.

Result: 185 MB → 7.3 MB, with all 117 F3 build records and all 7 campaign
summaries preserved.

| design | full log | records | showcase campaign | F3 builds kept |
|---|---|---|---|---|
| npuload | 43.4 MB | 0.49 MB | `campaign_0_1784781980` | 18 |
| tensordyne | 92.4 MB | 6.11 MB | `campaign_0_1784801859` | 44 |
| tpu | 49.6 MB | 0.74 MB | `campaign_0_1784781933` | 55 |

The full logs stay on the machine that produced them and are gitignored by path.
Nothing regenerates them — if you need the discarded F0/F2 rows, they are only on
that machine.

## Using them

`campaign_records.jsonl` is ordinary campaign JSONL, so every reader takes it via
`--log`:

```bash
eda-rl collect --log eda_rl/campaigns/tpu/nangate45/campaign_records.jsonl \
               --campaign all --no-baseline --top 3 --open
```

```bash
eda-rl report  --log eda_rl/campaigns/npuload/nangate45/campaign_records.jsonl \
               --campaign all --open
```

`collect` is the fast path for showing performance examples. `report` on
tensordyne's 4793-row showcase takes several minutes to render — that is a
report-side scalability limit, not something the distillation introduced (the
43k-row original is far slower); pass `--campaign <id>` to render a smaller one.

Note `--design/--platform` resolution looks for `results_funnel_campaigns.jsonl`
and will **not** find these files. That is deliberate: on the machine that ran
the campaigns, the full 185 MB log still occupies that canonical path, and
committing different content there would leave the working tree permanently
dirty and break the next `eda-rl optimize` append.
