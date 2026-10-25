import { AlertCircle } from "lucide-react"

export function ErrorNote({ message }: { message: string }) {
  return (
    <p role="alert" className="flex items-start gap-2 text-sm text-bad">
      <AlertCircle className="mt-0.5 size-4 shrink-0" aria-hidden />
      {message}
    </p>
  )
}
