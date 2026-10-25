import { ArrowLeft } from "lucide-react"
import { useMemo, useState } from "react"

import { ErrorNote } from "@/components/ErrorNote"
import { Button } from "@/components/ui/button"
import { MetricsTable } from "@/components/MetricsTable"
import { TimeChart, type ChartLine } from "@/components/TimeChart"
import { Verdicts } from "@/components/Verdicts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { api, type BatchDetail } from "@/lib/api"
import { fmt, prettyExperiment, prettyLabel } from "@/lib/format"
import { href } from "@/lib/router"
import { usePalette } from "@/lib/theme"
import { useAsync } from "@/lib/use-async"

const RANGES = [
  { key: "1", label: "1Y" },
  { key: "3", label: "3Y" },
  { key: "5", label: "5Y" },
  { key: "10", label: "10Y" },
  { key: "all", label: "All" },
]

const formatGrowth = (v: number) => `${v.toFixed(2)}×`
const formatPct = (v: number) => `${(v * 100).toFixed(1)}%`
const formatExposure = (v: number) => `${Math.round(v * 100)}%`

/** First index on or after `years` before the last date. */
function rangeStart(dates: string[], range: string): number {
  if (range === "all" || dates.length === 0) return 0
  const last = new Date(dates[dates.length - 1])
  last.setFullYear(last.getFullYear() - Number(range))
  const iso = last.toISOString().slice(0, 10)
  const i = dates.findIndex((d) => d >= iso)
  return i === -1 ? 0 : i
}

/** Rebase growth to 1 at the start of the window and recompute drawdowns inside it. */
function windowed(values: (number | null)[], start: number) {
  const g = values.slice(start)
  const base = g.find((v) => v != null) ?? 1
  const growth = g.map((v) => (v == null ? null : v / base))
  let peak = -Infinity
  const drawdown = growth.map((v) => {
    if (v == null) return null
    peak = Math.max(peak, v)
    return v / peak - 1
  })
  return { growth, drawdown }
}

function describeSetup(b: BatchDetail): string {
  const c = b.config
  const parts = [
    c.initial_capital ? `Start capital ${fmt.euro(c.initial_capital)}` : null,
    c.contribution ? `${fmt.euro(c.contribution)} added each month` : null,
    `fees ${c.costs.bps} bps${c.costs.fixed ? ` + ${fmt.euroCents(c.costs.fixed)} per order` : ""}`,
    typeof c.cash_rate === "number"
      ? `cash earns ${fmt.pct(c.cash_rate)}`
      : `cash earns the ${c.cash_rate} rate`,
  ].filter(Boolean)
  const s = parts.join(", ")
  return s.charAt(0).toUpperCase() + s.slice(1) + "."
}

export function BatchPage({ id }: { id: string }) {
  const { data, error, loading } = useAsync(() => api.batch(id), [id])

  if (error) return <ErrorNote message={error} />
  if (loading || !data) {
    return (
      <div className="flex flex-col gap-6">
        <Skeleton className="h-10 w-2/3" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-80 w-full" />
      </div>
    )
  }
  return <BatchView batch={data} />
}

function BatchView({ batch }: { batch: BatchDetail }) {
  const palette = usePalette()
  const [range, setRange] = useState("all")
  const [log, setLog] = useState(true)
  const others = useMemo(
    () => batch.runs.filter((r) => r.label !== batch.benchmark),
    [batch],
  )
  const [selected, setSelected] = useState(others[0]?.id ?? batch.runs[0].id)

  // Color follows the strategy (config order), never its rank; the benchmark is ink.
  const colors = useMemo(() => {
    const out: Record<string, string> = { [batch.benchmark]: palette.bench }
    others.forEach((r, i) => (out[r.label] = palette.series[i % palette.series.length]))
    return out
  }, [batch, palette, others])

  const dates = batch.runs[0].series.dates
  const start = rangeStart(dates, range)
  const shownDates = useMemo(() => dates.slice(start), [dates, start])

  const { growthLines, drawdownLines } = useMemo(() => {
    const ordered = [...others, ...batch.runs.filter((r) => r.label === batch.benchmark)]
    const growthLines: ChartLine[] = []
    const drawdownLines: ChartLine[] = []
    for (const r of ordered) {
      const w = windowed(r.series.growth, start)
      const base = {
        key: r.label,
        label: prettyLabel(r.label),
        color: colors[r.label],
        emphasis: r.label === batch.benchmark,
      }
      growthLines.push({ ...base, values: w.growth })
      drawdownLines.push({ ...base, values: w.drawdown })
    }
    return { growthLines, drawdownLines }
  }, [batch, others, colors, start])

  const run = batch.runs.find((r) => r.id === selected) ?? batch.runs[0]
  const exposureLines = useMemo<ChartLine[]>(
    () => [
      {
        key: run.label,
        label: prettyLabel(run.label),
        color: colors[run.label],
        values: run.series.exposure.slice(start),
      },
    ],
    [run, colors, start],
  )

  return (
    <div className="flex flex-col gap-12">
      <header className="flex flex-col gap-4">
        <a
          href={href({ page: "experiments" })}
          className="inline-flex w-fit items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-4" aria-hidden />
          All experiments
        </a>
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
          {prettyExperiment(batch.experiment)}
        </h1>
        <p className="max-w-[75ch] text-muted-foreground">
          From {fmt.date(batch.start)} to {fmt.date(batch.end)}. {describeSetup(batch)} Ran{" "}
          {fmt.dateTime(batch.created)}.
        </p>
      </header>

      <section aria-label="Verdicts against the benchmark">
        <Verdicts runs={batch.runs} benchmark={batch.benchmark} colors={colors} />
      </section>

      <section aria-labelledby="charts-heading" className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <h2 id="charts-heading" className="text-xl font-semibold">
            Over time
          </h2>
          <div className="flex flex-wrap items-center gap-2">
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={range}
              onValueChange={(v) => v && setRange(v)}
              aria-label="Time window"
            >
              {RANGES.map((r) => (
                <ToggleGroupItem key={r.key} value={r.key}>
                  {r.label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
            <ToggleGroup
              type="single"
              variant="outline"
              size="sm"
              value={log ? "log" : "linear"}
              onValueChange={(v) => v && setLog(v === "log")}
              aria-label="Vertical scale"
            >
              <ToggleGroupItem value="log">Log</ToggleGroupItem>
              <ToggleGroupItem value="linear">Linear</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>Growth of 1 €</CardTitle>
            <CardDescription>
              Time-weighted, since {fmt.date(shownDates[0])}. The thick line is{" "}
              {prettyLabel(batch.benchmark).toLowerCase()}.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TimeChart
              dates={shownDates}
              lines={growthLines}
              format={formatGrowth}
              log={log}
              height={320}
              ariaLabel="Growth of one euro for each strategy"
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Fall from the previous peak</CardTitle>
            <CardDescription>How deep each strategy was under water, and for how long.</CardDescription>
          </CardHeader>
          <CardContent>
            <TimeChart
              dates={shownDates}
              lines={drawdownLines}
              format={formatPct}
              height={220}
              ariaLabel="Drawdown from peak for each strategy"
            />
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="metrics-heading" className="flex flex-col gap-4">
        <h2 id="metrics-heading" className="text-xl font-semibold">
          Full-period numbers
        </h2>
        <p className="-mt-2 text-sm text-muted-foreground">Bold marks the best value in each row.</p>
        <Card className="py-0">
          <CardContent className="px-2 md:px-4">
            <MetricsTable runs={batch.runs} benchmark={batch.benchmark} colors={colors} />
          </CardContent>
        </Card>
      </section>

      <section aria-labelledby="inside-heading" className="flex flex-col gap-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <h2 id="inside-heading" className="text-xl font-semibold">
            Inside a strategy
          </h2>
          <Select value={selected} onValueChange={setSelected}>
            <SelectTrigger className="w-56" aria-label="Strategy">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {batch.runs.map((r) => (
                <SelectItem key={r.id} value={r.id}>
                  {prettyLabel(r.label)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>Share of the account invested</CardTitle>
            <CardDescription>The rest sits in cash, earning the cash rate.</CardDescription>
          </CardHeader>
          <CardContent>
            <TimeChart
              dates={shownDates}
              lines={exposureLines}
              format={formatExposure}
              height={180}
              ariaLabel={`Share of the account invested by ${prettyLabel(run.label)}`}
            />
          </CardContent>
        </Card>
        <Trades runId={run.id} />
      </section>
    </div>
  )
}

function Trades({ runId }: { runId: string }) {
  const { data, error, loading } = useAsync(() => api.trades(runId), [runId])
  const [showAll, setShowAll] = useState(false)
  if (error) return <ErrorNote message={error} />
  if (loading || !data) return <Skeleton className="h-40 w-full" />
  const newestFirst = [...data].reverse()
  const recent = showAll ? newestFirst : newestFirst.slice(0, 12)
  return (
    <Card className="py-0">
      <CardContent className="px-2 md:px-4">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Date</TableHead>
              <TableHead>Order</TableHead>
              <TableHead className="text-right">Amount</TableHead>
              <TableHead className="hidden text-right sm:table-cell">Price</TableHead>
              <TableHead className="text-right">Fees</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {recent.map((t, i) => (
              <TableRow key={`${t.date}-${t.ticker}-${i}`}>
                <TableCell className="text-muted-foreground">{fmt.date(t.date)}</TableCell>
                <TableCell>
                  {t.notional >= 0 ? "Buy" : "Sell"} {t.ticker}
                </TableCell>
                <TableCell className="text-right tabular-nums">
                  {fmt.euroCents(Math.abs(t.notional))}
                </TableCell>
                <TableCell className="hidden text-right text-muted-foreground tabular-nums sm:table-cell">
                  {fmt.euroCents(t.price)}
                </TableCell>
                <TableCell className="text-right text-muted-foreground tabular-nums">
                  {fmt.euroCents(t.cost)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        <div className="flex items-center justify-between gap-3 px-2 pb-4 text-sm text-muted-foreground">
          <span>
            {data.length > recent.length
              ? `Latest ${recent.length} of ${data.length} orders.`
              : `${data.length} orders.`}
          </span>
          {data.length > 12 && (
            <Button variant="ghost" size="sm" onClick={() => setShowAll((v) => !v)}>
              {showAll ? "Show latest 12" : `Show all ${data.length}`}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
