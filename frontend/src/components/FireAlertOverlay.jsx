import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

const POLL_MS = 2500

/* Fire/smoke alerts are not a Toast: Toast auto-dismisses after a few seconds, which is
 * fine for "saved successfully" and wrong for "there may be a fire" - this stays on
 * screen until an operator explicitly acknowledges it.
 *
 * Sourced from /api/events/live, which /api/dev/simulate-alert also feeds (dev builds
 * only) so this can be exercised before the real fire/smoke detector exists. Polled
 * independently of the History page's 10 s interval - a fire alert popping up 8 seconds
 * late is a worse trade than one extra request every 2.5 s.
 */
export default function FireAlertOverlay() {
  const [queue, setQueue] = useState([])
  const seen = useRef(new Set())

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      let page
      try {
        page = await api.liveEvents(20)
      } catch {
        return // the status rail already shows the backend is unreachable
      }
      if (cancelled) return

      const fresh = (page.events || []).filter((event) => {
        if (event.kind !== 'fire' && event.kind !== 'smoke') return false
        const key = `${event.ts}|${event.kind}|${event.message}`
        if (seen.current.has(key)) return false
        seen.current.add(key)
        return true
      })
      if (fresh.length) {
        // Oldest first into the queue, so acknowledging always clears the alert that
        // has been waiting longest rather than whichever arrived in the last poll.
        setQueue((q) => [...q, ...fresh.reverse()])
      }
    }

    poll()
    const id = setInterval(poll, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  if (queue.length === 0) return null

  const active = queue[0]
  const acknowledge = () => setQueue((q) => q.slice(1))
  const acknowledgeAll = () => setQueue([])

  return (
    <div
      role="alertdialog"
      aria-live="assertive"
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
    >
      <div className="w-full max-w-md rounded-2xl border-2 border-alarm-400 bg-ink-900 shadow-2xl shadow-alarm-900/50">
        <div className="flex items-center gap-3 rounded-t-2xl border-b border-alarm-900 bg-alarm-900/40 px-5 py-4">
          <span className="text-[26px] leading-none">{active.kind === 'smoke' ? '💨' : '🔥'}</span>
          <div className="min-w-0 flex-1">
            <p className="text-[15px] font-semibold uppercase tracking-[0.04em] text-alarm-400">
              {active.kind === 'smoke' ? 'Smoke detected' : 'Fire detected'}
            </p>
            <p className="text-[11.5px] text-ink-400">{formatWhen(active.ts)}</p>
          </div>
        </div>

        <div className="px-5 py-4">
          <p className="text-[13.5px] leading-relaxed text-ink-100">{active.message}</p>
          {queue.length > 1 && (
            <p className="mt-3 text-[11.5px] text-ink-400">
              {queue.length - 1} more alert{queue.length - 1 === 1 ? '' : 's'} waiting.
            </p>
          )}
        </div>

        <div className="flex gap-2 border-t border-ink-800 px-5 py-3.5">
          <button
            type="button"
            autoFocus
            onClick={acknowledge}
            className="flex-1 rounded-lg bg-alarm-400 px-3 py-2 text-[13px] font-semibold text-[#2a0808] transition-opacity hover:opacity-85"
          >
            Acknowledge
          </button>
          {queue.length > 1 && (
            <button
              type="button"
              onClick={acknowledgeAll}
              className="rounded-lg border border-ink-700 px-3 py-2 text-[13px] text-ink-300 transition-colors hover:bg-ink-800"
            >
              Acknowledge all
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function formatWhen(iso) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}
