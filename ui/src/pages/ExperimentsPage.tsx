import { Loader2, Play } from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"

import { ErrorNote } from "@/components/ErrorNote"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { api, type Batch } from "@/lib/api"
import { fmt, prettyExperiment, prettyLabel } from "@/lib/format"
import { href, navigate } from "@/lib/router"
import { useAsync } from "@/lib/use-async"

function bestVsHolding(batch: Batch): string {
  const bench = batch.runs.find((r) => r.label === batch.benchmark)
  const others = batch.runs.filter((r) => r.label !== batch.benchmark)
  if (!bench || !others.length) return "–"
  const best = others.reduce((a, b) =>
    (b.metrics.final_value ?? 0) > (a.metrics.final_value ?? 0) ? b : a,
  )
  const diff = (best.metrics.final_value ?? 0) / (bench.metrics.final_value || 1) - 1
  return `${prettyLabel(best.label)}, ${fmt.signedPct(diff, 0)}`
}

export function ExperimentsPage() {
  const experiments = useAsync(api.experiments, [])
  const batches = useAsync(api.batches, [])
  const [running, setRunning] = useState<string | null>(null)

  async function run(name: string) {
    setRunning(name)
    try {
      const { batch } = await api.runExperiment(name)
      toast.success(`Finished ${prettyExperiment(name)}`)
      navigate({ page: "batch", id: batch })
    } catch (e) {
      toast.error(`Couldn't run ${prettyExperiment(name)}`, { description: (e as Error).message })
    } finally {
      setRunning(null)
    }
  }

  return (
    <div className="flex flex-col gap-14">
      <header className="flex max-w-[65ch] flex-col gap-3">
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Experiments</h1>
        <p className="text-lg text-muted-foreground text-pretty">
          Each experiment backtests a few strategies on the same data, fees and cash rate, so they
          can be compared fairly with simply holding the asset.
        </p>
      </header>

      <section aria-labelledby="run-heading" className="flex flex-col gap-4">
        <h2 id="run-heading" className="text-xl font-semibold">
          Run an experiment
        </h2>
        {experiments.error && <ErrorNote message={experiments.error} />}
        {experiments.loading && <Skeleton className="h-24 w-full" />}
        <ul className="flex flex-col divide-y divide-border border-y">
          {experiments.data?.map((exp) => (
            <li key={exp.name} className="grid gap-3 py-5 md:grid-cols-[1fr_auto] md:items-center">
              <div className="flex flex-col gap-2">
                <p className="font-medium">
                  {prettyExperiment(exp.name)}{" "}
                  <span className="text-sm font-normal text-muted-foreground">
                    configs/{exp.name}.yaml
                  </span>
                </p>
                <p className="max-w-[75ch] text-sm text-muted-foreground">{exp.description}</p>
                <div className="flex flex-wrap gap-1.5">
                  {exp.strategies.map((s) => (
                    <Badge key={s} variant="outline">
                      {prettyLabel(s)}
                    </Badge>
                  ))}
                </div>
              </div>
              <Button
                onClick={() => run(exp.name)}
                disabled={running !== null}
                className="md:justify-self-end"
              >
                {running === exp.name ? (
                  <Loader2 className="animate-spin" aria-hidden />
                ) : (
                  <Play aria-hidden />
                )}
                {running === exp.name ? "Running…" : "Run"}
              </Button>
            </li>
          ))}
        </ul>
      </section>

      <section aria-labelledby="results-heading" className="flex flex-col gap-4">
        <h2 id="results-heading" className="text-xl font-semibold">
          Results
        </h2>
        {batches.error && <ErrorNote message={batches.error} />}
        {batches.loading && <Skeleton className="h-40 w-full" />}
        {batches.data && batches.data.length === 0 && (
          <p className="text-muted-foreground">
            No results yet. Run an experiment above to see how its strategies compare with holding.
          </p>
        )}
        {batches.data && batches.data.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Experiment</TableHead>
                <TableHead>Period</TableHead>
                <TableHead className="hidden md:table-cell">Strategies</TableHead>
                <TableHead>Best against holding</TableHead>
                <TableHead className="hidden text-right sm:table-cell">Ran</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {batches.data.map((b) => (
                <TableRow key={b.batch} className="relative">
                  <TableCell className="font-medium">
                    <a
                      href={href({ page: "batch", id: b.batch })}
                      className="after:absolute after:inset-0 focus-visible:outline-2"
                    >
                      {prettyExperiment(b.experiment)}
                    </a>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {b.start.slice(0, 4)} to {b.end.slice(0, 4)}
                  </TableCell>
                  <TableCell className="hidden text-muted-foreground md:table-cell">
                    {b.runs.length}
                  </TableCell>
                  <TableCell>{bestVsHolding(b)}</TableCell>
                  <TableCell className="hidden text-right text-muted-foreground sm:table-cell">
                    {fmt.dateTime(b.created)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>
    </div>
  )
}
