import { useCallback, useMemo, useState } from 'react'
import BoundaryCanvas from '../components/BoundaryCanvas'
import ZoneProperties from '../components/ZoneProperties'
import { Card, CardHead, EmptyState } from '../components/ui'
import { snapshotUrl } from '../api'
import { ZONE_TYPES, newZone, typeOf } from '../lib/zones'
import { useToast } from '../components/Toast'

export default function Boundaries({ zones, setZones, selected, setSelected, markDirty }) {
  const [drawing, setDrawing] = useState(null)
  const toast = useToast()

  // A frozen still, not the live stream: precise clicking against moving video is
  // miserable, and the boundary is static anyway.
  const backdrop = useMemo(() => snapshotUrl(Date.now()), [])

  const update = useCallback(
    (next) => {
      setZones(next)
      markDirty()
    },
    [setZones, markDirty],
  )

  const finish = useCallback(
    (points) => {
      if (!drawing) return
      const spec = ZONE_TYPES[drawing.type]
      if (points.length < spec.minPoints) {
        toast(`A ${spec.label.toLowerCase()} needs at least ${spec.minPoints} points.`, 'error')
        return
      }
      const zone = newZone(drawing.type, points, zones)
      update([...zones, zone])
      setSelected(zones.length)
      setDrawing(null)
    },
    [drawing, zones, update, setSelected, toast],
  )

  const hint = drawing
    ? ZONE_TYPES[drawing.type].area
      ? `Click to place corners · double-click or Enter to close · Esc to cancel (${drawing.points.length} placed)`
      : 'Click the two ends of the line · Esc to cancel'
    : zones.length
      ? 'Click a boundary to select · drag a handle to move · right-click a handle to delete'
      : 'Choose a boundary type below to draw your first one'

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_330px]">
      <div className="flex min-w-0 flex-col gap-4">
        <Card className="p-0">
          <BoundaryCanvas
            src={backdrop}
            zones={zones}
            selectedIndex={selected}
            drawing={drawing}
            editable
            hint={hint}
            onDrawingChange={setDrawing}
            onZonesChange={update}
            onSelect={setSelected}
            onFinish={finish}
          />
        </Card>

        <Card>
          <CardHead title="Add boundary" />
          <div className="grid gap-2 p-4 sm:grid-cols-2 xl:grid-cols-4">
            {Object.entries(ZONE_TYPES).map(([type, spec]) => (
              <button
                key={type}
                type="button"
                data-type={type}
                title={spec.description}
                onClick={() => {
                  setDrawing({ type, points: [] })
                  setSelected(-1)
                }}
                className={`flex items-center gap-2.5 rounded-lg border px-3 py-2.5 text-left transition-colors ${
                  drawing?.type === type
                    ? 'border-brand-500 bg-brand-900'
                    : 'border-ink-700 bg-ink-800 hover:border-ink-600'
                }`}
              >
                <i
                  className="h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ background: spec.colour }}
                />
                <span className="leading-tight">
                  <strong className="block text-[13px] font-semibold">{spec.label}</strong>
                  <small className="block text-[11px] text-ink-400">{spec.hint}</small>
                </span>
              </button>
            ))}
          </div>
          {drawing && (
            <div className="flex gap-2 border-t border-ink-800 px-4 py-3">
              <button
                type="button"
                id="btn-finish"
                onClick={() => finish(drawing.points)}
                className="rounded-lg border border-ink-700 bg-ink-800 px-3 py-1.5 text-[13px] hover:bg-ink-700"
              >
                Finish shape
              </button>
              <button
                type="button"
                onClick={() => setDrawing(null)}
                className="rounded-lg border border-ink-700 bg-ink-800 px-3 py-1.5 text-[13px] hover:bg-ink-700"
              >
                Cancel
              </button>
            </div>
          )}
        </Card>
      </div>

      <div className="flex flex-col gap-4">
        <Card>
          <CardHead
            title="Boundaries"
            aside={
              <span className="rounded-full bg-ink-700 px-2 py-0.5 text-[11px] text-ink-200">
                {zones.length}
              </span>
            }
          />
          {zones.length === 0 ? (
            <EmptyState>No boundaries yet.</EmptyState>
          ) : (
            <ul className="p-1.5">
              {zones.map((zone, index) => (
                <li key={zone.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(index)}
                    className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors ${
                      index === selected
                        ? 'border-brand-600 bg-brand-900'
                        : 'border-transparent hover:bg-ink-800'
                    }`}
                  >
                    <i
                      className="h-2.5 w-2.5 shrink-0 rounded-sm"
                      style={{ background: typeOf(zone).colour }}
                    />
                    <span className="min-w-0 flex-1 truncate text-[13px]">
                      {zone.name || zone.id}
                    </span>
                    <span className="shrink-0 text-[11px] text-ink-400">
                      {typeOf(zone).label}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        {selected >= 0 && zones[selected] && (
          <ZoneProperties
            zone={zones[selected]}
            onChange={(next) => update(zones.map((z, i) => (i === selected ? next : z)))}
            onDelete={() => {
              update(zones.filter((_, i) => i !== selected))
              setSelected(-1)
            }}
          />
        )}
      </div>
    </div>
  )
}
