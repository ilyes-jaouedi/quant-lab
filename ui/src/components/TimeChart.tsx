import {
  ColorType,
  CrosshairMode,
  LineStyle,
  PriceScaleMode,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts"
import { useEffect, useMemo, useRef, useState } from "react"

import { usePalette } from "@/lib/theme"

export interface ChartLine {
  key: string
  label: string
  color: string
  values: (number | null)[]
  /** The reference line (benchmark): drawn thicker so strategies read against it. */
  emphasis?: boolean
}

interface Props {
  dates: string[]
  lines: ChartLine[]
  format: (v: number) => string
  height?: number
  log?: boolean
  ariaLabel: string
}

/** Daily time series with a crosshair; the legend doubles as the hover readout. */
export function TimeChart({ dates, lines, format, height = 280, log = false, ariaLabel }: Props) {
  const container = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<Map<string, ISeriesApi<"Line">>>(new Map())
  const palette = usePalette()
  const [hover, setHover] = useState<{ date: string; values: Record<string, number> } | null>(null)

  const lastValues = useMemo(() => {
    const out: Record<string, number> = {}
    for (const line of lines) {
      for (let i = line.values.length - 1; i >= 0; i--) {
        const v = line.values[i]
        if (v != null) {
          out[line.key] = v
          break
        }
      }
    }
    return out
  }, [lines])

  useEffect(() => {
    if (!container.current) return
    const chart = createChart(container.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: palette.muted,
        fontFamily: "'Instrument Sans Variable', system-ui, sans-serif",
        fontSize: 11,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: palette.grid } },
      rightPriceScale: {
        borderVisible: false,
        mode: log ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal,
      },
      timeScale: { borderVisible: false, fixLeftEdge: true, fixRightEdge: true },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: palette.muted, style: LineStyle.Dashed, labelBackgroundColor: palette.text },
        horzLine: { visible: false, labelVisible: false },
      },
      localization: { priceFormatter: format, locale: "en-GB" },
      handleScroll: false,
      handleScale: false,
    })
    chartRef.current = chart
    const seriesMap = new Map<string, ISeriesApi<"Line">>()
    for (const line of lines) {
      const s = chart.addLineSeries({
        color: line.color,
        lineWidth: line.emphasis ? 3 : 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: palette.text,
      })
      s.setData(
        dates.map((d, i) => {
          const v = line.values[i]
          return v == null ? { time: d as Time } : { time: d as Time, value: v }
        }),
      )
      seriesMap.set(line.key, s)
    }
    seriesRef.current = seriesMap
    chart.timeScale().fitContent()

    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.point) {
        setHover(null)
        return
      }
      const values: Record<string, number> = {}
      for (const [key, s] of seriesMap) {
        const d = param.seriesData.get(s) as { value?: number } | undefined
        if (d?.value != null) values[key] = d.value
      }
      setHover({ date: String(param.time), values })
    })
    return () => {
      chart.remove()
      chartRef.current = null
    }
  }, [dates, lines, format, log, palette])

  const shown = hover?.values ?? lastValues

  return (
    <figure className="flex flex-col gap-3">
      <figcaption className="flex min-h-10 flex-wrap items-baseline gap-x-5 gap-y-1 text-sm">
        <span className="w-24 shrink-0 text-muted-foreground">
          {hover ? hover.date : "Latest"}
        </span>
        {lines.map((line) => (
          <span key={line.key} className="inline-flex items-baseline gap-1.5">
            <span
              aria-hidden
              className="inline-block h-0.5 w-3.5 translate-y-[-3px] rounded-full"
              style={{ background: line.color, height: line.emphasis ? 3 : 2 }}
            />
            <span className="text-muted-foreground">{line.label}</span>
            <span className="font-medium">
              {shown[line.key] != null ? format(shown[line.key]) : "–"}
            </span>
          </span>
        ))}
      </figcaption>
      <div ref={container} role="img" aria-label={ariaLabel} style={{ height }} className="w-full" />
    </figure>
  )
}
