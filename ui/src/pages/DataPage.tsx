import { Download, Loader2 } from "lucide-react"
import { useMemo, useState, type FormEvent } from "react"
import { toast } from "sonner"

import { ErrorNote } from "@/components/ErrorNote"
import { TimeChart, type ChartLine } from "@/components/TimeChart"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
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
import { api } from "@/lib/api"
import { fmt } from "@/lib/format"
import { href, navigate } from "@/lib/router"
import { usePalette } from "@/lib/theme"
import { useAsync } from "@/lib/use-async"

const formatPrice = (v: number) => v.toLocaleString("en-IE", { maximumFractionDigits: 2 })

export function DataPage({ ticker }: { ticker?: string }) {
  const list = useAsync(api.data, [])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)

  async function download(e: FormEvent) {
    e.preventDefault()
    const t = input.trim()
    if (!t) return
    setBusy(true)
    try {
      const res = await api.download(t)
      toast.success(`Downloaded ${t}`, { description: `${fmt.int(res.days)} trading days` })
      setInput("")
      list.reload()
      navigate({ page: "data", ticker: t })
    } catch (err) {
      toast.error(`Couldn't download ${t}`, { description: (err as Error).message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-12">
      <header className="flex max-w-[65ch] flex-col gap-3">
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Market data</h1>
        <p className="text-lg text-muted-foreground text-pretty">
          Daily prices kept on this computer. Strategies and notebooks read from here, so
          downloading happens once.
        </p>
      </header>

      <form onSubmit={download} className="flex max-w-md items-end gap-2">
        <label className="flex flex-1 flex-col gap-1.5 text-sm font-medium">
          Yahoo Finance ticker
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g. EUNL.DE or ^STOXX"
            autoComplete="off"
          />
        </label>
        <Button type="submit" disabled={busy || !input.trim()}>
          {busy ? <Loader2 className="animate-spin" aria-hidden /> : <Download aria-hidden />}
          {busy ? "Downloading…" : "Download"}
        </Button>
      </form>

      {ticker && <PriceChart key={ticker} ticker={ticker} />}

      <section aria-labelledby="cached-heading" className="flex flex-col gap-4">
        <h2 id="cached-heading" className="text-xl font-semibold">
          Downloaded series
        </h2>
        {list.error && <ErrorNote message={list.error} />}
        {list.loading && !list.data && <Skeleton className="h-40 w-full" />}
        {list.data && list.data.length === 0 && (
          <p className="text-muted-foreground">Nothing downloaded yet. Add a ticker above.</p>
        )}
        {list.data && list.data.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Ticker</TableHead>
                <TableHead>From</TableHead>
                <TableHead>To</TableHead>
                <TableHead className="text-right">Trading days</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {list.data.map((s) => (
                <TableRow
                  key={`${s.provider}/${s.ticker}`}
                  className="relative"
                  data-state={s.ticker === ticker ? "selected" : undefined}
                >
                  <TableCell className="font-medium">
                    <a
                      href={href({ page: "data", ticker: s.ticker })}
                      className="after:absolute after:inset-0"
                    >
                      {s.ticker}
                    </a>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{fmt.date(s.first)}</TableCell>
                  <TableCell className="text-muted-foreground">{fmt.date(s.last)}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmt.int(s.days)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>
    </div>
  )
}

function PriceChart({ ticker }: { ticker: string }) {
  const { data, error, loading } = useAsync(() => api.prices(ticker), [ticker])
  const palette = usePalette()
  const [log, setLog] = useState(true)
  const lines = useMemo<ChartLine[]>(
    () =>
      data
        ? [
            {
              key: "adj",
              label: "Total return (adjusted)",
              color: palette.series[0],
              values: data.adj_close,
            },
          ]
        : [],
    [data, palette],
  )

  if (error) return <ErrorNote message={error} />
  if (loading || !data) return <Skeleton className="h-96 w-full" />
  return (
    <Card>
      <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-3">
        <div className="flex flex-col gap-1.5">
          <CardTitle>{ticker}</CardTitle>
          <CardDescription>
            {fmt.date(data.dates[0])} to {fmt.date(data.dates[data.dates.length - 1])}, prices
            adjusted for dividends and splits.
          </CardDescription>
        </div>
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
      </CardHeader>
      <CardContent>
        <TimeChart
          dates={data.dates}
          lines={lines}
          format={formatPrice}
          log={log}
          height={340}
          ariaLabel={`${ticker} adjusted price history`}
        />
      </CardContent>
    </Card>
  )
}
