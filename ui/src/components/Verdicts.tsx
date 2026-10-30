import { Minus, TrendingDown, TrendingUp } from "lucide-react"

import type { RunSummary, VsBenchmark } from "@/lib/api"
import { fmt, prettyLabel } from "@/lib/format"

type Outcome = "beat" | "smoother" | "lost"

const SIGNIFICANT = 0.05

/** Plain-language reading of the paired bootstrap tests. */
function luckSentence(v: VsBenchmark): string {
  const part = (label: string, diff: number, p: number, value: string) => {
    if (diff <= 0) return `${label} ${value}: no edge to test.`
    return p < SIGNIFICANT
      ? `${label} ${value} is unlikely to be luck (p = ${p.toFixed(2)}).`
      : `${label} ${value} could easily be luck (p = ${p.toFixed(2)}).`
  }
  const ret = part(
    "Growth",
    v.return_diff ?? 0,
    v.return_diff_p ?? 1,
    `${fmt.signedPct(v.return_diff, 1)} a year`,
  )
  const sharpe = part(
    "Sharpe",
    v.sharpe_diff ?? 0,
    v.sharpe_diff_p ?? 1,
    `${(v.sharpe_diff ?? 0) >= 0 ? "+" : "−"}${Math.abs(v.sharpe_diff ?? 0).toFixed(2)}`,
  )
  return `${ret} ${sharpe}`
}

function outcome(run: RunSummary, bench: RunSummary): Outcome {
  const m = run.metrics
  const b = bench.metrics
  if ((m.final_value ?? 0) > (b.final_value ?? 0)) return "beat"
  const smoother =
    (m.sharpe ?? -Infinity) > (b.sharpe ?? Infinity) ||
    (m.max_drawdown ?? -1) > (b.max_drawdown ?? 0) + 0.05
  return smoother ? "smoother" : "lost"
}

const BADGE: Record<Outcome, { text: string; icon: typeof TrendingUp; tone: string }> = {
  beat: { text: "Beat holding", icon: TrendingUp, tone: "text-good" },
  smoother: { text: "Smoother, less money", icon: Minus, tone: "text-muted-foreground" },
  lost: { text: "Lost to holding", icon: TrendingDown, tone: "text-bad" },
}

interface Props {
  runs: RunSummary[]
  benchmark: string
  colors: Record<string, string>
}

/** One plain sentence per strategy: did it beat simply holding? */
export function Verdicts({ runs, benchmark, colors }: Props) {
  const bench = runs.find((r) => r.label === benchmark)
  if (!bench) return null
  const others = runs.filter((r) => r.label !== benchmark)
  const b = bench.metrics

  return (
    <ul className="flex flex-col divide-y divide-border">
      {others.map((run) => {
        const m = run.metrics
        const o = outcome(run, bench)
        const badge = BADGE[o]
        const Icon = badge.icon
        const diff =
          m.final_value != null && b.final_value ? m.final_value / b.final_value - 1 : null
        return (
          <li key={run.id} className="grid gap-x-6 gap-y-1 py-5 first:pt-0 md:grid-cols-[1fr_auto]">
            <p className="max-w-[62ch] text-xl leading-snug text-pretty md:text-2xl">
              <span
                aria-hidden
                className="mr-2 inline-block size-3 -translate-y-0.5 rounded-full"
                style={{ background: colors[run.label] }}
              />
              <span className="font-semibold">{prettyLabel(run.label)}</span> ended at{" "}
              <span className="font-semibold">{fmt.euro(m.final_value)}</span>, against{" "}
              {fmt.euro(b.final_value)} for {prettyLabel(benchmark).toLowerCase()} (
              {fmt.signedPct(diff, 0)}).
            </p>
            <p
              className={`inline-flex items-center gap-1.5 self-start text-sm font-medium md:justify-self-end ${badge.tone}`}
            >
              <Icon className="size-4" aria-hidden />
              {badge.text}
            </p>
            <p className="text-sm text-muted-foreground md:col-span-2">
              Worst loss {fmt.pct(m.max_drawdown)} vs {fmt.pct(b.max_drawdown)}. Sharpe{" "}
              {fmt.num(m.sharpe)} vs {fmt.num(b.sharpe)}. Fees {fmt.euro(m.costs)} over{" "}
              {fmt.int(m.n_trades)} orders. Invested {fmt.pct(m.avg_exposure, 0)} of the time.
            </p>
            {run.vs_benchmark?.return_diff_p != null && (
              <p className="text-sm md:col-span-2">
                <span className="font-medium">Could it be luck? </span>
                <span className="text-muted-foreground">{luckSentence(run.vs_benchmark)}</span>
              </p>
            )}
          </li>
        )
      })}
    </ul>
  )
}
