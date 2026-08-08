# eda-rl campaign dashboard (frontend)

React + TypeScript (Vite) frontend for the eda-rl campaign dashboard. Talks
to the FastAPI backend in `../eda_rl/api/` — see the top-level `AGENTS.md`
for how the two fit together and how to deploy.

```bash
npm install
npm run dev        # Vite dev server, proxies /api -> http://127.0.0.1:8000
                    # (start the backend separately: `eda-rl serve --port 8000`)

npm run build       # production build -> dist/
npm run lint         # oxlint
npm test             # vitest
npm run gen:api      # regenerate src/api/schema.d.ts from the backend's
                      # OpenAPI schema (needs `pip install -e '..[api]'`)
npm run gen:api -- --check   # fail if schema.d.ts is stale (what CI runs)
```

`src/api/schema.d.ts` is generated, not hand-written — re-run `npm run
gen:api` after changing any route/model in `eda_rl/api/`.
