import { useCallback, useEffect, useState } from 'react'
import { Badge, Button, Card, CardFoot, CardHead, EmptyState, Select, Stat } from '../components/ui'
import { api } from '../api'
import { emitsEvents } from '../lib/zones'

const KIND_LABEL = { boundary: 'Boundary', fire: 'Fire', smoke: 'Smoke', health: 'Camera' }

export default function AlertHistory({ zones }) {
  const [filters, setFilters] = useState({ kind: '', zoneId: '', limit: 200 })
  const [events, setEvents] = useState([])
  const [summary, setSummary] = useState({ available: false, counts: {}, today: {} })
  const [source, setSource] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    try {
      const [page, totals] = await Promise.all([api.events(filters), api.eventsSummary()])
      setEvents(page.events)
      setSource(page.source)
      setSummary(totals)
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

  const stats = deriveStats(summary, events)
  const set = (key) => (event) => setFilters((f) => ({ ...f, [key]: event.target.value }))

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {stats.map(({ label, value, tone }) => (
          <Stat key={label} label={label} value={value} tone={tone} />
        ))}
      </div>

      <Card>
        <CardHead title="Alerts">
          <div className="flex flex-wrap gap-2">
            <Select value={filters.kind} onChange={set('kind')} className="w-auto py-1.5">
              <option value="">All types</option>
              <option value="boundary">Boundary</option>
              <option value="fire">Fire</option>
              <option value="smoke">Smoke</option>
              <option value="health">Camera health</option>
            </Select>
            <Select value={filters.zoneId} onChange={set('zoneId')} className="w-auto py-1.5">
              <option value="">All boundaries</option>
              {zones.filter(emitsEvents).map((zone) => (
                <option key={zone.id} value={zone.id}>
                  {zone.name || zone.id}
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

        {/* Wide content scrolls inside its own box; the page never scrolls sideways. */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="bg-ink-800">
                {['Time', 'Alert', 'Boundary', 'Event', 'Track', 'Severity'].map((head) => (
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
                  <td className="min-w-[210px] px-4 py-2.5 font-medium">{event.message}</td>
                  <td className="whitespace-nowrap px-4 py-2.5 text-ink-200">
                    {event.zone_name || event.zone_id || '—'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5">
                    <Badge tone={toneFor(event)}>
                      {event.subtype || KIND_LABEL[event.kind] || event.kind}
                    </Badge>
                  </td>
                  <td className="whitespace-nowrap px-4 py-2.5 tabular-nums text-ink-300">
                    {event.track_id != null ? `#${event.track_id}` : '—'}
                  </td>
                  <td
                    className={`whitespace-nowrap px-4 py-2.5 ${
                      { high: 'font-semibold text-alarm-400', medium: 'text-warn-400' }[
                        event.severity
                      ] || 'text-ink-400'
                    }`}
                  >
                    {event.severity || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {events.length === 0 && <EmptyState>No alerts recorded yet.</EmptyState>}

        <CardFoot>
          <p className="text-[11.5px] text-ink-400">
            {error
              ? `Could not load history: ${error}`
              : source === 'postgres'
                ? 'Stored in PostgreSQL.'
                : 'PostgreSQL unavailable — showing recent alerts held in memory only. '
                  + 'These are NOT being recorded.'}
          </p>
        </CardFoot>
      </Card>
    </div>
  )
}

function formatWhen(iso) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  // Local time: an alert log is read against wall-clock memory ("what happened just
  // before 2am"), so the viewer's own zone is the useful one.
  return date.toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

function toneFor(event) {
  if (event.kind === 'fire' || event.kind === 'smoke') return 'alarm'
  if (event.kind === 'health') return 'info'
  return event.subtype === 'entry' ? 'ok' : 'warn'
}

/* Without PostgreSQL there is nothing to aggregate, but showing "0 today" above a table
 * listing today's alerts is a flat contradiction an operator would rightly distrust.
 * Fall back to counting what is actually on screen, and say so. */
function deriveStats(summary, events) {
  const sum = (counts) => Object.values(counts).reduce((a, b) => a + b, 0)

  if (summary.available) {
    const today = summary.today || {}
    return [
      { label: 'Today', value: sum(today) },
      { label: 'Boundary today', value: today.boundary || 0 },
      {
        label: 'Fire / smoke today',
        value: (today.fire || 0) + (today.smoke || 0),
        tone: (today.fire || 0) + (today.smoke || 0) > 0 ? 'alarm' : undefined,
      },
      { label: 'All time', value: sum(summary.counts || {}) },
    ]
  }

  const midnight = new Date()
  midnight.setHours(0, 0, 0, 0)
  const today = {}
  events.forEach((event) => {
    if (new Date(event.ts) >= midnight) today[event.kind] = (today[event.kind] || 0) + 1
  })

  return [
    { label: 'Today (in memory)', value: sum(today) },
    { label: 'Boundary today', value: today.boundary || 0 },
    { label: 'Fire / smoke today', value: (today.fire || 0) + (today.smoke || 0) },
    { label: 'Held in memory', value: events.length },
  ]
}
