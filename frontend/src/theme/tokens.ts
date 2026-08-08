// Hand-ported from eda_rl/viz/report.py's _CSS (the static HTML report's
// visual language) so the live dashboard matches a static `eda-rl report`
// someone might already have open, rather than looking like a second product.

export const colors = {
  headerFrom: "#1e3a5f",
  headerTo: "#1f2937",
  background: "#f1f5f9",
  card: "#ffffff",
  cardBorder: "#e2e8f0",
  text: "#1e293b",
  textMuted: "#475569",
  green: "#059669",
  amber: "#d97706",
  red: "#dc2626",
} as const;

// eda_rl.viz.report._FIDELITY_COLORS, F0 (killed earliest) -> F3 (reached
// full flow) — keep in sync if that palette changes.
export const fidelityColors: Record<string, string> = {
  F0: "#d62728",
  F1: "#ff7f0e",
  F2: "#9467bd",
  F3: "#2ca02c",
};

export const fontStack = "system-ui, -apple-system, sans-serif";
