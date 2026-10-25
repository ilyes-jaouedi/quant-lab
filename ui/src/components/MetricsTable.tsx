import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { Metrics, RunSummary } from "@/lib/api"
import { fmt, prettyLabel } from "@/lib/format"

interface Row {
  key: keyof Metrics
  label: string
  hint: string
  format: (v: number | null | undefined) => string
  better?: "high" | "low"
}

const ROWS: Row[] = [
  { key: "final_value", label: "Final value", hint: "Account value at the end", format: fmt.euro, better: "high" },
  { key: "cagr", label: "Annual return", hint: "Compound annual growth (time-weighted)", format: (v) => fmt.pct(v, 2), better: "high" },
  { key: "vol", label: "Volatility", hint: "Annualized standard deviation of daily returns", format: fmt.pct, better: "low" },
  { key: "sharpe", label: "Sharpe ratio", hint: "Return above cash per unit of volatility", format: (v) => fmt.num(v), better: "high" },
  { key: "sortino", label: "Sortino ratio", hint: "Like Sharpe, counting only downside volatility", format: (v) => fmt.num(v), better: "high" },
  { key: "max_drawdown", label: "Worst loss", hint: "Largest fall from a previous peak", format: fmt.pct, better: "high" },
  { key: "calmar", label: "Calmar ratio", hint: "Annual return divided by worst loss", format: (v) => fmt.num(v), better: "high" },
  { key: "avg_exposure", label: "Time invested", hint: "Average share of the account in the market", format: (v) => fmt.pct(v, 0) },
  { key: "turnover", label: "Turnover", hint: "Portfolio traded per year (1 = the whole account once)", format: (v) => fmt.num(v) },
  { key: "costs", label: "Fees paid", hint: "Spread and fixed fees over the whole period", format: fmt.euro, better: "low" },
  { key: "n_trades", label: "Orders", hint: "Number of buy or sell orders", format: fmt.int },
]

function bestIndex(values: (number | null | undefined)[], better?: "high" | "low") {
  if (!better) return -1
  let best = -1
  values.forEach((v, i) => {
    if (v == null) return
    if (best === -1) best = i
    else if (better === "high" ? v > values[best]! : v < values[best]!) best = i
  })
  return best
}

interface Props {
  runs: RunSummary[]
  benchmark: string
  colors: Record<string, string>
}

export function MetricsTable({ runs, benchmark, colors }: Props) {
  const ordered = [...runs].sort((a, b) =>
    a.label === benchmark ? -1 : b.label === benchmark ? 1 : 0,
  )
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="sm:w-48" />
          {ordered.map((r) => (
            <TableHead key={r.id} className="text-right align-bottom">
              <span className="inline-flex flex-col items-end gap-1 py-1">
                {r.label === benchmark && <Badge variant="secondary">Benchmark</Badge>}
                <span className="inline-flex items-center gap-1.5 text-foreground">
                  <span
                    aria-hidden
                    className="inline-block size-2.5 rounded-full"
                    style={{ background: colors[r.label] }}
                  />
                  {prettyLabel(r.label)}
                </span>
              </span>
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {ROWS.map((row) => {
          const values = ordered.map((r) => r.metrics[row.key])
          const best = bestIndex(values, row.better)
          return (
            <TableRow key={row.key}>
              <TableCell>
                <span className="font-medium">{row.label}</span>
                <span className="hidden text-xs text-muted-foreground sm:block">{row.hint}</span>
              </TableCell>
              {values.map((v, i) => (
                <TableCell
                  key={ordered[i].id}
                  className={`text-right tabular-nums ${i === best ? "font-semibold" : "text-muted-foreground"}`}
                >
                  {row.format(v)}
                </TableCell>
              ))}
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}
