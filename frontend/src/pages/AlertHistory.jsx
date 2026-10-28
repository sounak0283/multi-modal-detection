import { useCallback, useEffect, useState } from 'react'
import {
  Button,
  Card,
  CardFoot,
  CardHead,
  DataTable,
  EmptyState,
  EventPill,
  Input,
  SegmentedControl,
  Select,
  SeverityChip,
  Stat,
  Td,
  Th,
  Tr,
} from '../components/ui'
import { maskStoragePath } from '../lib/secure'
import { api, eventClipUrl, eventSnapshotUrl } from '../api'
import { emitsEvents } from '../lib/zones'
import { useAIAvailable } from '../lib/useAIAvailable'
import AlertAI from '../components/ai/AlertAI'

const KIND_LABEL = {
  boundary: 'Boundary',
  crowd: 'Crowd',
  fire: 'Fire',
  smoke: 'Smoke',
  ppe: 'PPE',
  access: 'Restricted access',
  welding: 'Welding sparks',
  health: 'Camera health',
}

const PERIODS = [
  ['today', 'Today'],
  ['week', 'This week'],
  ['month', 'This month'],
  ['6months', 'Last 6 months'],
  ['', 'All time'],
]
const PERIOD_LABEL = Object.fromEntries(PERIODS)

const DEFAULT_FILTERS = { kind: '', zoneId: '', cameraId: '', limit: 200, period: '' }

/** Start of a period in the viewer's own timezone, as an ISO timestamp (or '' = all time).
 * Recomputed on every refresh, so "Today" rolls over at the viewer's midnight. */
function periodStart(period, now = new Date()) {
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()) // local midnight
  if (period === 'today') return start.toISOString()
  if (period === 'week') {
    start.setDate(start.getDate() - ((start.getDay() + 6) % 7)) // back to Monday
    return start.toISOString()
  }
  if (period === 'month') return new Date(now.getFullYear(), now.getMonth(), 1).toISOString()
  if (period === '6months') {
    start.setMonth(start.getMonth() - 6)
    return start.toISOString()
  }
  return ''
}

export default function AlertHistory({ cameras, zones }) {
  const [filters, setFilters] = useState(DEFAULT_FILTERS)
  const [events, setEvents] = useState([])
  const [window_, setWindow] = useState(null) // counts per kind for period + camera + boundary
  const [error, setError] = useState(null)
  const [viewing, setViewing] = useState(null) // null | an event with evidence to show
  // Plain-English search (AI): while set, the list AND the cards show its results.
  const [aiResult, setAiResult] = useState(null)
  const aiAvailable = useAIAvailable()

  const load = useCallback(async () => {
    // One `since` for both requests, so the cards and the list count the same window.
    const since = periodStart(filters.period)
    try {
      const [page, totals] = await Promise.all([
        api.events({ ...filters, since }),
        api.eventsSummary(filters.cameraId, { zoneId: filters.zoneId, since }),
      ])
      setEvents(page.events)
      setWindow(totals.available ? totals.window || {} : null)
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

  const set = (key) => (value) => setFilters((f) => ({ ...f, [key]: value }))
  const active = Object.keys(DEFAULT_FILTERS).some(
    (key) => key !== 'limit' && filters[key] !== DEFAULT_FILTERS[key],
  )
  const shown = aiResult ? aiResult.events : events
  const stats = aiResult
    ? deriveStats(tally(aiResult.events), filters, shown.length, 'AI search')
    : deriveStats(window_, filters, events.length)

  return (
    <div className="space-y-4">
      {aiAvailable && <AiSearch result={aiResult} onResult={setAiResult} />}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {stats.map(({ label, value, tone, icon, iconTone }) => (
          <Stat key={label} label={label} value={value} tone={tone} icon={icon} iconTone={iconTone} />
        ))}
      </div>

      <Card>
        <div className="space-y-3 border-b border-ink-800 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <PeriodPills value={filters.period} onChange={set('period')} />
            <div className="flex items-center gap-2">
              {active && (
                <Button variant="ghost" onClick={() => setFilters(DEFAULT_FILTERS)}>
                  Clear filters
                </Button>
              )}
              <Button onClick={load}>Refresh</Button>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <FilterSelect label="Type" value={filters.kind} onChange={set('kind')}>
              <option value="">All types</option>
              {Object.entries(KIND_LABEL).map(([kind, label]) => (
                <option key={kind} value={kind}>
                  {label}
                </option>
              ))}
            </FilterSelect>
            <FilterSelect label="Boundary" value={filters.zoneId} onChange={set('zoneId')}>
              <option value="">All boundaries</option>
              {zones.filter(emitsEvents).map((zone) => (
                <option key={zone.id} value={zone.id}>
                  {zone.name || zone.id}
                </option>
              ))}
            </FilterSelect>
            <FilterSelect label="Camera" value={filters.cameraId} onChange={set('cameraId')}>
              <option value="">All cameras</option>
              {(cameras ?? []).map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name || camera.id}
                </option>
              ))}
            </FilterSelect>
            <FilterSelect label="Show" value={filters.limit} onChange={set('limit')}>
              <option value="50">Newest 50</option>
              <option value="200">Newest 200</option>
              <option value="500">Newest 500</option>
            </FilterSelect>
          </div>
        </div>

        <DataTable>
          <thead>
            <tr>
              {['Time', 'Alert', 'Boundary', 'Event', 'Track', 'Severity', 'Evidence'].map((head) => (
                <Th key={head}>{head}</Th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((event) => (
              <Tr key={event.id ?? event.ts}>
                <Td mono>{formatWhen(event.ts)}</Td>
                <Td className="min-w-[210px] whitespace-normal font-medium">{event.message}</Td>
                <Td className="text-ink-200">{event.zone_name || event.zone_id || '—'}</Td>
                <Td>
                  <EventPill event={event}>
                    {event.subtype || KIND_LABEL[event.kind] || event.kind}
                  </EventPill>
                </Td>
                <Td mono>{event.track_id != null ? `#${event.track_id}` : '—'}</Td>
                <Td>
                  <SeverityChip severity={event.severity} />
                </Td>
                <Td className="py-1">
                  {event.snapshot_path ? (
                    <button
                      type="button"
                      onClick={() => setViewing(event)}
                      className="block h-8 w-12 overflow-hidden rounded-sm border border-ink-700 transition-colors hover:border-brand-500"
                    >
                      <img
                        src={eventSnapshotUrl(event.id)}
                        alt="Alert snapshot"
                        className="h-full w-full object-cover"
                      />
                    </button>
                  ) : (
                    <span className="text-ink-500">—</span>
                  )}
                </Td>
              </Tr>
            ))}
          </tbody>
        </DataTable>

        {shown.length === 0 && (
          <EmptyState>
            {aiResult
              ? 'No alerts match that search.'
              : active
                ? 'No alerts match these filters.'
                : 'No alerts recorded yet.'}
          </EmptyState>
        )}

        <CardFoot>
          <p className="text-[11.5px] text-ink-400">
            {error ? `Could not load history: ${error}` : 'Stored in MongoDB.'}
          </p>
        </CardFoot>
      </Card>

      {viewing && <EvidenceLightbox event={viewing} onClose={() => setViewing(null)} />}
    </div>
  )
}

/* Full snapshot + clip playback for one alert (Expansion Plan Phase C). A plain
 * `Card`-based overlay, same look `FireAlertOverlay.jsx` already established for a
 * modal over the whole dashboard. */
function EvidenceLightbox({ event, onClose }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
    >
      {/* Card itself does not spread onClick, so the stop-propagation guard lives on a
          plain wrapper - clicking anywhere inside the card must not bubble up to the
          overlay's own onClick and close the lightbox. */}
      <div
        className="max-h-[92vh] w-full max-w-2xl space-y-3 overflow-y-auto"
        onClick={(event_) => event_.stopPropagation()}
      >
        <Card>
          <CardHead
            title={event.message || 'Alert evidence'}
            aside={<span className="text-[11.5px] text-ink-400">{formatWhen(event.ts)}</span>}
          >
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </CardHead>
          <div className="space-y-3 p-4">
            {event.snapshot_path && (
              <img
                src={eventSnapshotUrl(event.id)}
                alt="Alert snapshot"
                className="w-full rounded-lg border border-ink-800"
              />
            )}
            {event.clip_path && (
              // eslint-disable-next-line jsx-a11y/media-has-caption
              <video
                src={eventClipUrl(event.id)}
                controls
                className="w-full rounded-lg border border-ink-800"
              />
            )}
            {!event.snapshot_path && !event.clip_path && (
              <EmptyState>
                {event.clip_status === 'pending'
                  ? 'The video is still being saved - reopen this alert in a minute.'
                  : event.clip_status === 'failed'
                    ? 'The video could not be saved after several attempts.'
                    : 'No evidence was captured for this alert.'}
              </EmptyState>
            )}
            {/* Storage reference for admin audit, masked: the real bucket and key are not
                shown on screen. Playback links are minted fresh per view (they expire). */}
            {event.clip_uri && (
              <p className="break-all text-[11.5px] text-ink-400">
                Stored at{' '}
                <span className="font-mono text-ink-300">{maskStoragePath(event.clip_uri)}</span>
              </p>
            )}
          </div>
        </Card>
        <AlertAI event={event} chatHeight="h-[320px]" />
      </div>
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

/** The cards count exactly what the list is filtered to: same period, camera and boundary
 * (the server applies one shared `since`). "Listed" makes the Show cap visible - the list
 * holds the newest N of the matching alerts, never silently fewer. */
function deriveStats(counts, filters, listed, period = PERIOD_LABEL[filters.period]) {
  if (!counts) {
    return [
      { label: `Alerts · ${period}`, value: '—', icon: 'notifications_active', iconTone: 'rose' },
      { label: 'Boundary', value: '—', icon: 'polyline', iconTone: 'purple' },
      { label: 'Fire / smoke', value: '—', icon: 'local_fire_department', iconTone: 'amber' },
      { label: 'Listed below', value: listed, icon: 'list', iconTone: 'sky' },
    ]
  }
  const total = Object.values(counts).reduce((a, b) => a + b, 0)
  const fireSmoke = (counts.fire || 0) + (counts.smoke || 0)
  const matching = filters.kind ? counts[filters.kind] || 0 : total
  return [
    { label: `Alerts · ${period}`, value: total, icon: 'notifications_active', iconTone: 'rose' },
    { label: `Boundary · ${period}`, value: counts.boundary || 0, icon: 'polyline', iconTone: 'purple' },
    {
      label: `Fire / smoke · ${period}`,
      value: fireSmoke,
      tone: fireSmoke > 0 ? 'alarm' : undefined,
      icon: 'local_fire_department',
      iconTone: 'amber',
    },
    {
      label: 'Listed below',
      value: matching > listed ? `${listed} of ${matching}` : listed,
      icon: 'list',
      iconTone: 'sky',
    },
  ]
}

const tally = (list) =>
  list.reduce((counts, e) => ({ ...counts, [e.kind]: (counts[e.kind] || 0) + 1 }), {})

/* "fire alerts at gate 2 last night" -> the server has the AI pick filters, runs them, and
 * says in one line what it searched; that line is always shown so a wrong reading is
 * obvious. Clear goes back to the normal filters. */
function AiSearch({ result, onResult }) {
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const run = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setBusy(true)
    setError(null)
    try {
      onResult(await api.alertSearch(query))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <form onSubmit={run} className="flex flex-wrap items-center gap-2 p-3">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder='Search in plain English, e.g. "fire alerts on cam_01 last night"'
          aria-label="Search alerts in plain English"
          className="min-w-55 flex-1"
        />
        <Button type="submit" variant="primary" disabled={busy || !query.trim()}>
          {busy ? 'Searching…' : 'Search'}
        </Button>
        {result && (
          <Button
            variant="ghost"
            onClick={() => {
              onResult(null)
              setQuery('')
            }}
          >
            Clear search
          </Button>
        )}
      </form>
      {(result || error) && (
        <p
          className={`border-t border-ink-800 px-4 py-2 text-[12px] ${error ? 'text-alarm-400' : 'text-ink-300'}`}
        >
          {error ||
            `Showing: ${result.explanation || 'search results'} — ${result.events.length} alert${
              result.events.length === 1 ? '' : 's'
            }${result.events.length >= 200 ? ' (newest 200)' : ''}. The filters below are not applied.`}
        </p>
      )}
    </Card>
  )
}

function PeriodPills({ value, onChange }) {
  return (
    <SegmentedControl
      value={value}
      onChange={onChange}
      options={PERIODS.map(([period, label]) => ({ value: period, label }))}
    />
  )
}

function FilterSelect({ label, value, onChange, children }) {
  return (
    <label className="block">
      <span className="mb-1 block font-mono text-[10.5px] font-medium uppercase tracking-[0.08em] text-ink-400">
        {label}
      </span>
      <Select value={value} onChange={(event) => onChange(event.target.value)}>
        {children}
      </Select>
    </label>
  )
}
