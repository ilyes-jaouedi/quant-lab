const euro0 = new Intl.NumberFormat("en-IE", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 0,
})
const euro2 = new Intl.NumberFormat("en-IE", { style: "currency", currency: "EUR" })
const num = new Intl.NumberFormat("en-IE", { maximumFractionDigits: 2 })
const dateFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" })

const missing = "–"

export const fmt = {
  euro: (v: number | null | undefined) => (v == null ? missing : euro0.format(v)),
  euroCents: (v: number | null | undefined) => (v == null ? missing : euro2.format(v)),
  pct: (v: number | null | undefined, digits = 1) =>
    v == null ? missing : `${(v * 100).toFixed(digits)}%`,
  signedPct: (v: number | null | undefined, digits = 1) =>
    v == null ? missing : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(digits)}%`,
  num: (v: number | null | undefined, digits = 2) =>
    v == null ? missing : v.toFixed(digits),
  int: (v: number | null | undefined) => (v == null ? missing : num.format(Math.round(v))),
  date: (iso: string) => dateFmt.format(new Date(iso)),
  dateTime: (iso: string) =>
    new Intl.DateTimeFormat("en-GB", {
      day: "numeric",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(iso)),
}

/** Human name for a strategy label, e.g. "sma_10m" -> "SMA 10m", "buy_and_hold" -> "Buy and hold". */
export function prettyLabel(label: string): string {
  const known: Record<string, string> = {
    buy_and_hold: "Buy and hold",
    vol_target: "Vol targeting",
    sma_10m: "SMA 10-month",
    l1_trend: "L1 trend",
    equal_weight: "Equal weight",
    risk_parity: "Risk parity",
    tsmom_12m: "Momentum 12m",
    tsmom_12m_rp: "Momentum 12m, risk parity",
    sma_10m_rp: "SMA 10m, risk parity",
    ema_10m_rp: "EMA 10m, risk parity",
  }
  if (known[label]) return known[label]
  const s = label.replace(/[_-]+/g, " ")
  return s.charAt(0).toUpperCase() + s.slice(1)
}

export function prettyExperiment(name: string): string {
  const known: Record<string, string> = {
    world_cto: "MSCI World, CTO hobby account",
    sp500_timing: "S&P 500 timing since 1990",
    multi_asset_trend: "Trend across 8 asset classes",
  }
  return known[name] ?? prettyLabel(name)
}
