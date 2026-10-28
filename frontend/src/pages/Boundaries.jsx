import { useCallback, useMemo, useState } from 'react'
import BoundaryCanvas from '../components/BoundaryCanvas'
import ZoneProperties from '../components/ZoneProperties'
import { Badge, Button, Card, CardHead, EmptyState } from '../components/ui'
import { snapshotUrl } from '../api'
import { ZONE_TYPES, newZone, typeForPoints, typeOf } from '../lib/zones'
import { useToast } from '../components/Toast'

export default function Boundaries({
  cameraId,
  zones,
  setZones,
  selected,
  setSelected,
  markDirty,
  readOnly = false,
  backdropSrc = null,
}) {
  const [drawing, setDrawing] = useState(null)
  const toast = useToast()

  // A frozen still, not the live stream: precise clicking against moving video is
  // miserable, and the boundary is static anyway.
  const backdrop = useMemo(
    () => backdropSrc ?? snapshotUrl(cameraId, Date.now()),
    [cameraId, backdropSrc],
  )

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
      const zone = newZone(typeForPoints(points), points, zones)
      update([...zones, zone])
      setSelected(zones.length)
      setDrawing(null)
    },
    [drawing, zones, update, setSelected, toast],
  )

  const hint = readOnly
    ? 'View only - boundaries can be changed by an admin'
    : drawing
    ? ZONE_TYPES[drawing.type].area
      ? `Click to place points (2 = a line, 3+ = an area) · double-click or Enter to finish · Esc to cancel (${drawing.points.length} placed)`
      : ''
    : zones.length
      ? 'Click a boundary to select · drag a handle to move · right-click a handle to delete'
      : 'Press Draw boundary above to draw your first one'

  const removeAt = useCallback(
    (index) => {
      update(zones.filter((_, i) => i !== index))
      if (index === selected) setSelected(-1)
      else if (index < selected) setSelected(selected - 1)
    },
    [zones, update, selected, setSelected],
  )

  const clearAll = useCallback(() => {
    if (!zones.length) return
    const noun = zones.length === 1 ? 'boundary' : 'boundaries'
    // A native confirm rather than a custom modal: this is the one bulk-destructive
    // action on the page, unsaved until "Save boundaries" is clicked, and does not
    // warrant a whole dialog component for a single use.
    if (!window.confirm(`Remove all ${zones.length} ${noun}? This cannot be undone once saved.`)) {
      return
    }
    update([])
    setSelected(-1)
  }, [zones, update, setSelected])

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <div className="flex min-w-0 flex-col gap-3">
        {!readOnly && (
          <div className="flex flex-wrap items-center gap-2 rounded-xl border border-ink-700 bg-ink-900 px-3 py-2">
            <button
              type="button"
              data-type="draft"
              onClick={() => {
                setDrawing({ type: 'draft', points: [] })
                setSelected(-1)
              }}
              className={`flex h-8 items-center gap-2 rounded-lg border px-3 text-[13px] font-medium transition-colors ${
                drawing
                  ? 'border-brand-500 bg-brand-900 text-brand-100'
                  : 'border-ink-700 bg-ink-950 text-ink-100 hover:border-ink-hover'
              }`}
            >
              <i className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: ZONE_TYPES.draft.colour }} />
              Draw boundary
            </button>
            {drawing && (
              <>
                <Button id="btn-finish" onClick={() => finish(drawing.points)}>
                  Finish shape
                </Button>
                <Button onClick={() => setDrawing(null)}>Cancel</Button>
              </>
            )}
            <p className="ml-auto text-[12px] text-ink-400">
              Draw the shape first, then choose which model uses it in its properties.
            </p>
          </div>
        )}

        <Card className="p-0">
          <BoundaryCanvas
            src={backdrop}
            zones={zones}
            selectedIndex={selected}
            drawing={drawing}
            editable={!readOnly}
            hint={hint}
            onDrawingChange={setDrawing}
            onZonesChange={update}
            onSelect={setSelected}
            onFinish={finish}
          />
        </Card>
      </div>

      <div className="flex flex-col gap-3">
        <Card>
          <CardHead
            title="Boundaries"
            aside={<Badge>{zones.length}</Badge>}
          >
            {zones.length > 0 && !readOnly && (
              <button
                type="button"
                onClick={clearAll}
                className="ml-auto text-[11.5px] text-ink-400 hover:text-alarm-400"
              >
                Clear all
              </button>
            )}
          </CardHead>
          {zones.length === 0 ? (
            <EmptyState>No boundaries yet.</EmptyState>
          ) : (
            <ul className="p-1.5">
              {zones.map((zone, index) => (
                <li key={zone.id}>
                  <div
                    className={`group flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 transition-colors ${
                      index === selected
                        ? 'border-brand-500/40 bg-brand-900'
                        : 'border-transparent hover:bg-ink-800'
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => setSelected(index)}
                      className="flex min-w-0 flex-1 items-center gap-2.5 text-left"
                    >
                      <i
                        className="h-2.5 w-2.5 shrink-0 rounded-sm"
                        style={{ background: typeOf(zone).colour }}
                      />
                      <span className="min-w-0 flex-1 truncate text-[13px]">
                        {zone.name || zone.id}
                      </span>
                      <span className="shrink-0 font-mono text-[10.5px] uppercase tracking-[0.04em] text-ink-400">
                        {typeOf(zone).label}
                      </span>
                    </button>
                    {!readOnly && (
                      <button
                        type="button"
                        title={`Delete ${zone.name || zone.id}`}
                        onClick={(e) => {
                          e.stopPropagation()
                          removeAt(index)
                        }}
                        className="shrink-0 rounded-md px-1.5 py-1 text-ink-500 opacity-0 transition-opacity hover:bg-alarm-900 hover:text-alarm-400 group-hover:opacity-100"
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>

        {selected >= 0 && zones[selected] && (
          // A disabled fieldset disables every input and button inside it, so an
          // operator can inspect a boundary's rules without editing what they can't save.
          <fieldset disabled={readOnly} className="contents">
            <ZoneProperties
              zone={zones[selected]}
              onChange={(next) => update(zones.map((z, i) => (i === selected ? next : z)))}
              onDelete={() => removeAt(selected)}
            />
          </fieldset>
        )}
      </div>
    </div>
  )
}
