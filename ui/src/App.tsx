import { Moon, Sun } from "lucide-react"

import { Button } from "@/components/ui/button"
import { href, useRoute, type Route } from "@/lib/router"
import { useTheme } from "@/lib/theme"
import { BatchPage } from "@/pages/BatchPage"
import { DataPage } from "@/pages/DataPage"
import { ExperimentsPage } from "@/pages/ExperimentsPage"

const NAV: { label: string; route: Route; match: Route["page"][] }[] = [
  { label: "Experiments", route: { page: "experiments" }, match: ["experiments", "batch"] },
  { label: "Market data", route: { page: "data" }, match: ["data"] },
]

export default function App() {
  const route = useRoute()
  const { theme, toggle } = useTheme()

  return (
    <div className="min-h-svh">
      <header className="border-b">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-8 px-4 md:px-8">
          <a href="#/" className="text-base font-semibold tracking-tight">
            fin lab
          </a>
          <nav aria-label="Main" className="flex items-center gap-1">
            {NAV.map((item) => {
              const active = item.match.includes(route.page)
              return (
                <a
                  key={item.label}
                  href={href(item.route)}
                  aria-current={active ? "page" : undefined}
                  className={`rounded-md px-3 py-1.5 text-sm transition-colors hover:text-foreground ${
                    active ? "bg-secondary font-medium text-foreground" : "text-muted-foreground"
                  }`}
                >
                  {item.label}
                </a>
              )
            })}
          </nav>
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto"
            onClick={toggle}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {theme === "dark" ? <Sun aria-hidden /> : <Moon aria-hidden />}
          </Button>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-10 md:px-8 md:py-14">
        {route.page === "experiments" && <ExperimentsPage />}
        {route.page === "batch" && <BatchPage key={route.id} id={route.id} />}
        {route.page === "data" && <DataPage ticker={route.ticker} />}
      </main>
    </div>
  )
}
