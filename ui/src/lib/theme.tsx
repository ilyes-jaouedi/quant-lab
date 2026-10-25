import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react"

type Theme = "light" | "dark"

const STORAGE_KEY = "fin-theme"

function storedTheme(): Theme | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === "light" || v === "dark" ? v : null
  } catch {
    return null
  }
}

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"
}

const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "light",
  toggle: () => {},
})

export function ThemeProvider({ children }: { children: ReactNode }) {
  // The class is set before the state changes so that colors read during render
  // (see usePalette) already belong to the new theme.
  const [theme, setThemeState] = useState<Theme>(() => {
    const t = storedTheme() ?? systemTheme()
    document.documentElement.classList.toggle("dark", t === "dark")
    return t
  })
  const setTheme = (t: Theme) => {
    document.documentElement.classList.toggle("dark", t === "dark")
    setThemeState(t)
  }

  useEffect(() => {
    // Follow the OS setting until the person picks a theme explicitly.
    const mq = window.matchMedia("(prefers-color-scheme: dark)")
    const onChange = () => {
      if (!storedTheme()) setTheme(systemTheme())
    }
    mq.addEventListener("change", onChange)
    return () => mq.removeEventListener("change", onChange)
  }, [])

  const value = useMemo(
    () => ({
      theme,
      toggle: () => {
        const next: Theme = theme === "dark" ? "light" : "dark"
        try {
          localStorage.setItem(STORAGE_KEY, next)
        } catch {
          // Storage can be unavailable (private mode); the toggle still works for this visit.
        }
        setTheme(next)
      },
    }),
    [theme],
  )
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = () => useContext(ThemeContext)

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export interface Palette {
  bench: string
  series: string[]
  text: string
  muted: string
  grid: string
  border: string
  good: string
  bad: string
}

/** Resolved chart colors for the current theme (canvas charts can't read CSS variables). */
export function usePalette(): Palette {
  const { theme } = useTheme()
  return useMemo(() => {
    void theme // recompute when the theme class changes
    return {
      bench: cssVar("--bench"),
      series: Array.from({ length: 8 }, (_, i) => cssVar(`--series-${i + 1}`)),
      text: cssVar("--foreground"),
      muted: cssVar("--muted-foreground"),
      grid: cssVar("--grid"),
      border: cssVar("--border"),
      good: cssVar("--good"),
      bad: cssVar("--bad"),
    }
  }, [theme])
}
