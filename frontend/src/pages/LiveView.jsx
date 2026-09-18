import { useMemo } from 'react'
import BoundaryCanvas from '../components/BoundaryCanvas'
import { Card, CardHead, EmptyState, Stat } from '../components/ui'
import { api, streamUrl } from '../api'
import { usePolling } from '../lib/usePolling'
import { ZONE_TYPES } from '../lib/zones'

export default function LiveView({ cameraId, zones }) {
  // The stream URL must be stable per camera, otherwise every re-render tears down the
  // MJPEG connection and restarts it - which shows as a visible stutter every two
  // seconds. Re-derives only when the camera being viewed actually changes.
  const src = useMemo(() => streamUrl(cameraId, Date.now()), [cameraId])
  const { data } = usePolling(() => api.liveEvents(20, cameraId), 3000)
  const events = data?.events ?? []

  // Per-camera stats (people tracked, zone occupancy, feed state) now live on the
  // camera detail response rather than the old global /api/health, since any number of
  // cameras can be running independently.
  const { data: camera } = usePolling(() => api.getCamera(cameraId), 2000)
  const { data: summary } = usePolling(() => api.eventsSummary(cameraId), 10000)
  const occupancy = camera?.stats?.zone_occupancy ?? {}
  const entriesToday = summary?.entries_by_zone_today ?? {}
  const countableZones = zones.filter((zone) => zone.type === 'polygon')

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_330px]">
      <Card className="p-0">
        <BoundaryCanvas src={src} zones={zones} />
      </Card>

      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="People on camera" value={camera?.stats?.people_tracked ?? 0} />
          <Stat
            label="Inside boundaries"
            value={Object.values(occupancy).reduce((a, b) => a + b, 0)}
          />
        </div>

        <Card>
          <CardHead title="Occupancy" />
          {countableZones.length === 0 ? (
            <EmptyState>Draw a boundary (type Zone) to count people inside it.</EmptyState>
          ) : (
            <ul className="p-1.5">
              {countableZones.map((zone) => (
                <li
                  key={zone.id}
                  className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[12.5px] not-first:border-t not-first:border-ink-800"
                >
                  <span className="min-w-0 flex-1 truncate text-ink-200">
                    {zone.name || zone.id}
                  </span>
                  <span className="tabular-nums font-semibold text-ink-100">
                    {occupancy[zone.id] ?? 0} now
                  </span>
                  <span className="tabular-nums text-ink-400">
                    {entriesToday[zone.id] ?? 0} today
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>

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
