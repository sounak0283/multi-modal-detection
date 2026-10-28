import BoundaryCanvas from '../components/BoundaryCanvas'
import { Card, CardHead, Dot, EmptyState, EventPill, SeverityChip, Stat, TelemetryChip } from '../components/ui'
import { api } from '../api'
import { usePolling } from '../lib/usePolling'
import { useRetryingStream } from '../lib/useRetryingStream'
import { ZONE_TYPES } from '../lib/zones'

const FEED_TONE = { live: 'ok', starting: 'warn', lost: 'alarm' }

export default function LiveView({ cameraId, zones }) {
  // The stream URL must be stable across re-renders, otherwise every render tears down
  // the MJPEG connection and restarts it (a visible stutter every two seconds). It
  // changes only on mount (keyed on cameraId) and after a load error, to retry.
  const { src, onError } = useRetryingStream(cameraId)
  const { data } = usePolling(() => api.liveEvents(20, cameraId), 3000)
  const events = data?.events ?? []

  // Per-camera stats (people tracked, zone occupancy, feed state) now live on the
  // camera detail response rather than the old global /api/health, since any number of
  // cameras can be running independently.
  const { data: camera } = usePolling(() => api.getCamera(cameraId), 2000)
  const { data: summary } = usePolling(() => api.eventsSummary(cameraId), 10000)
  const stats = camera?.stats ?? {}
  const occupancy = stats.zone_occupancy ?? {}
  const entriesToday = summary?.entries_by_zone_today ?? {}
  const countableZones = zones.filter((zone) => zone.type === 'polygon')
  const inside = Object.values(occupancy).reduce((a, b) => a + b, 0)
  const feedTone = FEED_TONE[stats.feed_state] ?? null
  const num = (value, digits = 1) => (typeof value === 'number' ? value.toFixed(digits) : '—')

  return (
    <div className="space-y-3">
      {/* Telemetry strip: the same numbers the camera reports, as 20px mono chips. */}
      <div className="flex flex-wrap items-center gap-2">
        <TelemetryChip>
          <span className="flex items-center gap-1.5">
            <Dot tone={feedTone} live={stats.feed_state === 'live'} />
            <span className="uppercase">{stats.feed_state ?? 'no signal'}</span>
          </span>
        </TelemetryChip>
        <TelemetryChip label="Cam">{cameraId}</TelemetryChip>
        <TelemetryChip label="FPS">{num(stats.render_fps)}</TelemetryChip>
        <TelemetryChip label="MS">{num(stats.inference_ms)}</TelemetryChip>
        <TelemetryChip label="Tracked">{stats.people_tracked ?? 0}</TelemetryChip>
        <TelemetryChip label="Inside">{inside}</TelemetryChip>
        <TelemetryChip label="Events">{stats.events_fired ?? 0}</TelemetryChip>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="People on camera" value={stats.people_tracked ?? 0} icon="groups" iconTone="indigo" />
        <Stat label="Inside boundaries" value={inside} icon="polyline" iconTone="purple" />
        <Stat
          label="Events fired"
          value={stats.events_fired ?? 0}
          icon="notifications_active"
          iconTone="rose"
        />
        <Stat label="Inference ms" value={num(stats.inference_ms)} icon="speed" iconTone="sky" />
      </div>

      <div className="grid grid-cols-1 items-start gap-3 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Card className="relative p-0">
          <BoundaryCanvas src={src} onImageError={onError} zones={zones} />
          {/* Overlay never takes pointer events - the canvas below is interactive. */}
          <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2">
            <span className="flex h-5 items-center gap-1.5 rounded-sm border border-ink-700 bg-black/70 px-1.5 font-mono text-[11px] uppercase text-white">
              <Dot tone={feedTone} live={stats.feed_state === 'live'} />
              {cameraId}
            </span>
          </div>
        </Card>

        <div className="flex flex-col gap-3">
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
                    <span className="font-mono font-semibold tabular-nums text-ink-100">
                      {occupancy[zone.id] ?? 0} now
                    </span>
                    <span className="font-mono tabular-nums text-ink-400">
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
              <ul className="max-h-[360px] space-y-2 overflow-y-auto p-2">
                {events.map((event, i) => (
                  <li
                    key={`${event.ts}-${i}`}
                    className="rounded-lg border border-ink-700 bg-ink-850 px-3 py-2 text-[12.5px]"
                  >
                    <div className="flex items-center gap-2">
                      <EventPill event={event}>
                        {event.subtype || event.kind || 'event'}
                      </EventPill>
                      {event.severity && <SeverityChip severity={event.severity} />}
                      <span className="ml-auto font-mono text-[11px] tabular-nums text-ink-400">
                        {new Date(event.ts).toLocaleTimeString(undefined, { hour12: false })}
                      </span>
                    </div>
                    <p className="mt-1.5 leading-snug text-ink-200">{event.message}</p>
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
    </div>
  )
}
