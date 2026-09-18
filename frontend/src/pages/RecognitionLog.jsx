import { useCallback, useEffect, useState } from 'react'
import { Badge, Button, Card, CardFoot, CardHead, EmptyState, Select, Stat } from '../components/ui'
import { api, eventSnapshotUrl } from '../api'

/* Every confirmed zone ENTRY with identity enabled resolves to one of three statuses
 * (identity/resolver.py) - this page is the audit trail of that resolution, not a
 * separate detection: same `events` collection AlertHistory reads, filtered to rows
 * that carry an identity_status at all. "Known" is what the ask was for; "Unrecognised"
 * is kept alongside it rather than dropped, since a run of unrecognised faces at one
 * camera is exactly the kind of thing this log exists to make visible. */
const STATUS_OPTIONS = [
  { value: '', label: 'All identity events' },
  { value: 'known', label: 'Recognised only' },
  { value: 'unknown_face', label: 'Unrecognised only' },
]

export default function RecognitionLog({ cameras }) {
  const [filters, setFilters] = useState({ cameraId: '', identityStatus: '', limit: 200 })
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)
  const [viewing, setViewing] = useState(null)

  const load = useCallback(async () => {
    try {
      // identity_status is unset server-side would return every event; this page only
      // ever wants rows identity actually resolved, so "all" here means "known or
      // unknown_face", never the boundary/fire/smoke events AlertHistory already covers.
      const wanted = filters.identityStatus ? [filters.identityStatus] : ['known', 'unknown_face']
      const pages = await Promise.all(
        wanted.map((identityStatus) =>
          api.events({ ...filters, identityStatus, limit: filters.limit }),
        ),
      )
      const merged = pages.flatMap((p) => p.events)
      merged.sort((a, b) => new Date(b.ts) - new Date(a.ts))
      setEvents(merged.slice(0, filters.limit))
      setError(null)
    } catch (err) {
      setError(err.message)
    }
  }, [filters])

  useEffect(() => {
    load()
    const id = setInterval(load, 10000)
    return () => clearInterval(id)
  }, [load])

  const set = (key) => (event) => setFilters((f) => ({ ...f, [key]: event.target.value }))
  const cameraName = (id) => cameras?.find((c) => c.id === id)?.name || id

  const knownCount = events.filter((e) => e.identity_status === 'known').length
  const unknownCount = events.length - knownCount

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Recognised" value={knownCount} tone="ok" />
        <Stat label="Unrecognised" value={unknownCount} tone={unknownCount > 0 ? 'warn' : undefined} />
      </div>

      <Card>
        <CardHead title="Recognition log">
          <div className="flex flex-wrap gap-2">
            <Select value={filters.identityStatus} onChange={set('identityStatus')} className="w-auto py-1.5">
              {STATUS_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </Select>
            <Select value={filters.cameraId} onChange={set('cameraId')} className="w-auto py-1.5">
              <option value="">All cameras</option>
              {(cameras ?? []).map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name || camera.id}
                </option>
              ))}
            </Select>
            <Select value={filters.limit} onChange={set('limit')} className="w-auto py-1.5">
              <option value="50">Last 50</option>
              <option value="200">Last 200</option>
              <option value="500">Last 500</option>
            </Select>
            <Button onClick={load}>Refresh</Button>
          </div>
        </CardHead>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="bg-ink-800">
                {['Time', 'Person', 'Camera', 'Boundary', 'Track', 'Evidence'].map((head) => (
                  <th
                    key={head}
                    className="whitespace-nowrap px-4 py-2.5 text-left text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-400"
                  >
                    {head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {events.map((event) => (
                <tr key={event.id ?? event.ts} className="border-t border-ink-800 hover:bg-ink-800">
                  <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-ink-300">
                    {formatWhen(event.ts)}
                  </td>
                  <td className="min-w-[160px] px-4 py-2.5 font-medium">
                    {event.identity_status === 'known' ? (
                      <Badge tone="ok">{event.identity_name || 'Known'}</Badge>
                    ) : (
                      <Badge tone="warn">Unrecognised</Badge>
                    )}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-ink-200">
                    {cameraName(event.camera_id)}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-ink-200">
                    {event.zone_name || event.zone_id || '—'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-ink-300">
                    {event.track_id != null ? `#${event.track_id}` : '—'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5">
                    {event.snapshot_path ? (
                      <button
                        type="button"
                        onClick={() => setViewing(event)}
                        className="block h-10 w-14 overflow-hidden rounded-md border border-ink-700 transition-colors hover:border-brand-500"
                      >
                        <img
                          src={eventSnapshotUrl(event.id)}
                          alt="Recognition snapshot"
                          className="h-full w-full object-cover"
                        />
                      </button>
                    ) : (
                      <span className="text-ink-500">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {events.length === 0 && (
          <EmptyState>
            No identity resolutions recorded yet. This fills in once a boundary with People
            detection and at least one enrolled person has a confirmed entry.
          </EmptyState>
        )}

        <CardFoot>
          <p className="text-[11.5px] text-ink-400">
            {error ? `Could not load recognition log: ${error}` : 'Stored in MongoDB, same events collection as Alert history.'}
          </p>
        </CardFoot>
      </Card>

      {viewing && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={() => setViewing(null)}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
        >
          <div className="w-full max-w-xl" onClick={(e) => e.stopPropagation()}>
            <Card>
              <CardHead
                title={
                  viewing.identity_status === 'known'
                    ? viewing.identity_name || 'Known person'
                    : 'Unrecognised face'
                }
                aside={<span className="text-[11.5px] text-ink-400">{formatWhen(viewing.ts)}</span>}
              >
                <Button variant="ghost" onClick={() => setViewing(null)}>
                  Close
                </Button>
              </CardHead>
              <div className="space-y-3 p-4">
                {viewing.snapshot_path ? (
                  <img
                    src={eventSnapshotUrl(viewing.id)}
                    alt="Recognition snapshot"
                    className="w-full rounded-lg border border-ink-800"
                  />
                ) : (
                  <EmptyState>No evidence was captured for this event.</EmptyState>
                )}
              </div>
            </Card>
          </div>
        </div>
      )}
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
