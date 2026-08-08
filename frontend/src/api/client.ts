import createClient from "openapi-fetch";
import type { paths } from "./schema";

// Paths in the generated schema already carry the "/api" prefix (the
// FastAPI router is mounted with prefix="/api"), so baseUrl stays empty —
// the Vite dev server proxies "/api" to the backend (see vite.config.ts),
// and in production the built app is served same-origin by eda-rl serve.
export const api = createClient<paths>({ baseUrl: "" });
