import { useEffect, useState } from "react"

export type Route =
  | { page: "experiments" }
  | { page: "batch"; id: string }
  | { page: "data"; ticker?: string }

function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent)
  if (parts[0] === "batch" && parts[1]) return { page: "batch", id: parts[1] }
  if (parts[0] === "data") return { page: "data", ticker: parts[1] || undefined }
  return { page: "experiments" }
}

export function href(route: Route): string {
  switch (route.page) {
    case "batch":
      return `#/batch/${encodeURIComponent(route.id)}`
    case "data":
      return route.ticker ? `#/data/${encodeURIComponent(route.ticker)}` : "#/data"
    default:
      return "#/"
  }
}

export function navigate(route: Route) {
  window.location.hash = href(route)
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash))
  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash))
      window.scrollTo(0, 0)
    }
    window.addEventListener("hashchange", onChange)
    return () => window.removeEventListener("hashchange", onChange)
  }, [])
  return route
}
