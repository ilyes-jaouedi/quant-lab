export type Metrics = Partial<
  Record<
    | "cagr"
    | "vol"
    | "sharpe"
    | "sortino"
    | "max_drawdown"
    | "calmar"
    | "irr"
    | "final_value"
    | "contributed"
    | "avg_exposure"
    | "turnover"
    | "costs"
    | "n_trades"
    | "years",
    number | null
  >
>

export interface Experiment {
  name: string
  description: string
  start: string
  strategies: string[]
}

/** Paired stationary-bootstrap tests against the benchmark (absent for the benchmark itself). */
export interface VsBenchmark {
  psr: number | null
  dsr: number | null
  n_trials: number
  n?: number
  return_diff?: number
  return_diff_ci?: [number, number]
  return_diff_p?: number
  sharpe_diff?: number
  sharpe_diff_ci?: [number, number]
  sharpe_diff_p?: number
}

export interface RunSummary {
  id: string
  label: string
  strategy: string
  params: Record<string, unknown>
  metrics: Metrics
  vs_benchmark?: VsBenchmark | null
}

export interface RunSeries {
  dates: string[]
  value: (number | null)[]
  growth: (number | null)[]
  drawdown: (number | null)[]
  exposure: (number | null)[]
}

export interface BatchConfig {
  start: string
  end: string
  initial_capital: number
  contribution: number
  contribution_freq: string
  costs: { bps: number; fixed: number }
  cash_rate: number | string
  execution: string
}

export interface Batch {
  batch: string
  experiment: string
  created: string
  start: string
  end: string
  benchmark: string
  config: BatchConfig
  runs: RunSummary[]
}

export interface BatchDetail extends Batch {
  runs: (RunSummary & { series: RunSeries })[]
}

export interface Trade {
  date: string
  ticker: string
  shares: number
  price: number
  notional: number
  cost: number
}

export interface CachedSeries {
  ticker: string
  provider: string
  first: string
  last: string
  days: number
}

export interface PriceSeries {
  ticker: string
  dates: string[]
  close: (number | null)[]
  adj_close: (number | null)[]
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new Error("Can't reach the API. Start it with `uv run fin serve`.")
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

const enc = encodeURIComponent

export const api = {
  experiments: () => request<Experiment[]>("/api/experiments"),
  runExperiment: (name: string) =>
    request<{ batch: string }>(`/api/experiments/${enc(name)}/run`, { method: "POST" }),
  batches: () => request<Batch[]>("/api/batches"),
  batch: (id: string) => request<BatchDetail>(`/api/batches/${enc(id)}`),
  trades: (runId: string) => request<Trade[]>(`/api/runs/${enc(runId)}/trades`),
  data: () => request<CachedSeries[]>("/api/data"),
  prices: (ticker: string) => request<PriceSeries>(`/api/data/${enc(ticker)}`),
  download: (ticker: string) =>
    request<{ ticker: string; days: number }>(`/api/data/${enc(ticker)}/fetch`, {
      method: "POST",
    }),
}
