import { useMemo } from 'react'
import BoundaryCanvas from '../components/BoundaryCanvas'
import { Card, CardHead, EmptyState } from '../components/ui'
import { api, streamUrl } from '../api'
import { usePolling } from '../lib/usePolling'
import { ZONE_TYPES } from '../lib/zones'

export default function LiveView({ zones }) {
  // The stream URL must be stable, otherwise every re-render tears down the MJPEG
  // connection and restarts it - which shows as a visible stutter every two seconds.
  const src = useMemo(() => streamUrl(Date.now()), [])
  const { data } = usePolling(() => api.liveEvents(20), 3000)
  const events = data?.events ?? []

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_330px]">
      <Card className="p-0">
        <BoundaryCanvas src={src} zones={zones} />
      </Card>

      <div className="flex flex-col gap-4">
        <Card>
          <CardHead title="Recent activity" />
          {events.length === 0 ? (
            <EmptyState>Nothing yet.</EmptyState>
          ) : (
            <ul className="p-1.5">
              {events.map((event, i) => (
                <li
                  key={`${event.ts}-${i}`}
                  className="flex items-baseline gap-2.5 rounded-lg px-2.5 py-2 text-[12.5px] not-first:border-t not-first:border-ink-800"
                >
                  <span className="shrink-0 tabular-nums text-ink-400">
                    {new Date(event.ts).toLocaleTimeString(undefined, { hour12: false })}
                  </span>
                  <span className={event.subtype === 'entry' ? 'text-ok-400' : 'text-warn-400'}>
                    {event.message}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card>
          <CardHead title="Legend" />
          <ul className="space-y-2 p-4">
            {Object.entries(ZONE_TYPES).map(([key, spec]) => (
              <li key={key} className="flex items-center gap-2.5 text-[12.5px]">
                <i
                  className="h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ background: spec.colour }}
                />
                <span className="text-ink-200">{spec.label}</span>
                <span className="ml-auto text-ink-400">{spec.hint}</span>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  )
}
