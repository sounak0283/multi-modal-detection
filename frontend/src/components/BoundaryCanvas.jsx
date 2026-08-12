import { useCallback, useEffect, useRef, useState } from 'react'
import { ZONE_TYPES, typeOf } from '../lib/zones'

/* Canvas boundary editor.
 *
 * Coordinates are normalised 0-1 everywhere and converted to pixels only at draw time.
 * That is what lets a boundary survive a resolution change or a swap to the camera's
 * low-resolution substream - the same reason the backend stores them that way.
 *
 * The wrapper is shrink-to-fit around the rendered image so the canvas's bounding rect
 * is exactly the image box. Pointer positions are normalised against that rect, so any
 * mismatch would place boundaries in the wrong part of the frame.
 */

const HIT_RADIUS = 9

export default function BoundaryCanvas({
  src,
  zones = [],
  selectedIndex = -1,
  drawing = null,
  editable = false,
  hint = '',
  onDrawingChange,
  onZonesChange,
  onSelect,
  onFinish,
}) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  const [hover, setHover] = useState(null)
  const dragRef = useRef(null)

  /* -------------------------------------------------- sizing */

  const measure = useCallback(() => {
    const image = imageRef.current
    if (!image) return
    const rect = image.getBoundingClientRect()
    if (rect.width && rect.height) setSize({ width: rect.width, height: rect.height })
  }, [])

  useEffect(() => {
    measure()
    const observer = new ResizeObserver(measure)
    if (imageRef.current) observer.observe(imageRef.current)
    window.addEventListener('resize', measure)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [measure])

  /* -------------------------------------------------- drawing */

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !size.width) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(size.width * dpr)
    canvas.height = Math.round(size.height * dpr)
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.width, size.height)

    zones.forEach((zone, index) => {
      paintShape(ctx, zone.points, typeOf(zone), size, index === selectedIndex, false, hover)
      paintLabel(ctx, zone, size)
      if (!typeOf(zone).area) paintArrow(ctx, zone, size)
    })

    if (drawing?.points?.length) {
      paintShape(ctx, drawing.points, ZONE_TYPES[drawing.type], size, false, true, null)
    }
  }, [zones, selectedIndex, drawing, size, hover])

  /* -------------------------------------------------- pointer */

  const toNormalised = (event) => {
    const rect = canvasRef.current.getBoundingClientRect()
    return {
      x: clamp((event.clientX - rect.left) / rect.width),
      y: clamp((event.clientY - rect.top) / rect.height),
    }
  }

  const findVertex = (point) => {
    for (let z = zones.length - 1; z >= 0; z -= 1) {
      const points = zones[z].points
      for (let i = 0; i < points.length; i += 1) {
        const dx = (points[i][0] - point.x) * size.width
        const dy = (points[i][1] - point.y) * size.height
        if (Math.hypot(dx, dy) <= HIT_RADIUS) return { zone: z, point: i }
      }
    }
    return null
  }

  const handleMouseDown = (event) => {
    if (!editable || event.button !== 0) return
    const point = toNormalised(event)

    if (drawing) {
      const points = [...drawing.points, [point.x, point.y]]
      onDrawingChange({ ...drawing, points })
      // A two-point tripwire is complete the moment the second point lands.
      if (!ZONE_TYPES[drawing.type].area && points.length === 2) onFinish(points)
      return
    }

    const vertex = findVertex(point)
    if (vertex) {
      dragRef.current = vertex
      onSelect(vertex.zone)
      return
    }
    onSelect(findZoneAt(zones, point))
  }

  const handleMouseMove = (event) => {
    if (!editable) return
    const point = toNormalised(event)

    if (dragRef.current) {
      const { zone, point: index } = dragRef.current
      const next = zones.map((z, i) =>
        i === zone
          ? { ...z, points: z.points.map((p, j) => (j === index ? [point.x, point.y] : p)) }
          : z,
      )
      onZonesChange(next)
      return
    }
    setHover(findVertex(point))
  }

  useEffect(() => {
    const stop = () => {
      dragRef.current = null
    }
    window.addEventListener('mouseup', stop)
    return () => window.removeEventListener('mouseup', stop)
  }, [])

  const handleContextMenu = (event) => {
    if (!editable) return
    event.preventDefault()
    const vertex = findVertex(toNormalised(event))
    if (!vertex) return

    const zone = zones[vertex.zone]
    const spec = typeOf(zone)
    if (zone.points.length <= spec.minPoints) return

    onZonesChange(
      zones.map((z, i) =>
        i === vertex.zone ? { ...z, points: z.points.filter((_, j) => j !== vertex.point) } : z,
      ),
    )
  }

  const handleDoubleClick = (event) => {
    if (!editable || !drawing) return
    event.preventDefault()
    onFinish(drawing.points)
  }

  useEffect(() => {
    if (!editable) return undefined
    const onKey = (event) => {
      if (event.key === 'Enter' && drawing) {
        event.preventDefault()
        onFinish(drawing.points)
      }
      if (event.key === 'Escape') onDrawingChange(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [editable, drawing, onFinish, onDrawingChange])

  const cursor = drawing ? 'crosshair' : hover ? 'grab' : 'default'

  return (
    <div className="flex justify-center bg-[#05070a]">
      <div className="relative max-w-full leading-none">
        <img
          ref={imageRef}
          src={src}
          alt="Camera view"
          onLoad={measure}
          className="block w-auto max-w-full"
          style={{ maxHeight: 'calc(100vh - 250px)' }}
        />
        <canvas
          ref={canvasRef}
          // Stable id: tools/ui_e2e.py drives real clicks against this element, and it
          // is the only test that exercises canvas coordinate mapping.
          id="overlay"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onContextMenu={handleContextMenu}
          onDoubleClick={handleDoubleClick}
          className="absolute inset-0 h-full w-full"
          style={{ cursor: editable ? cursor : 'default' }}
        />
        {hint && (
          /* pointer-events-none is load-bearing: without it this panel swallows clicks
             in the bottom-left of the frame, and a boundary drawn near that corner
             silently loses a vertex. */
          <div className="pointer-events-none absolute bottom-3 left-3 max-w-[65%] rounded-lg border border-ink-700 bg-black/80 px-3 py-1.5 text-[12px] leading-snug text-ink-300">
            {hint}
          </div>
        )}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- painting */

const clamp = (v) => Math.min(1, Math.max(0, v))

function paintShape(ctx, points, spec, size, selected, active, hover) {
  if (!points?.length) return
  const pts = points.map(([x, y]) => [x * size.width, y * size.height])

  ctx.lineWidth = selected ? 3 : 2
  ctx.strokeStyle = spec.colour
  ctx.setLineDash(active ? [6, 4] : [])

  ctx.beginPath()
  ctx.moveTo(pts[0][0], pts[0][1])
  pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))

  if (spec.area && pts.length >= 3) {
    ctx.closePath()
    ctx.fillStyle = `${spec.colour}2e`
    ctx.fill()
  }
  ctx.stroke()
  ctx.setLineDash([])

  if (!selected && !active) return
  pts.forEach(([x, y], i) => {
    ctx.beginPath()
    ctx.arc(x, y, hover?.point === i && selected ? 7 : 5, 0, Math.PI * 2)
    ctx.fillStyle = '#fff'
    ctx.fill()
    ctx.lineWidth = 2
    ctx.strokeStyle = spec.colour
    ctx.stroke()
  })
}

function paintLabel(ctx, zone, size) {
  if (!zone.points?.length) return
  const [x, y] = [zone.points[0][0] * size.width, zone.points[0][1] * size.height]
  ctx.fillStyle = typeOf(zone).colour
  ctx.font = '600 12px ui-sans-serif, system-ui, sans-serif'
  ctx.fillText(zone.name || zone.id, x + 6, y - 8)
}

/** Perpendicular arrow showing which crossing direction counts as an entry. */
function paintArrow(ctx, zone, size) {
  if (zone.points.length < 2) return
  const [a, b] = zone.points
  const mid = [((a[0] + b[0]) / 2) * size.width, ((a[1] + b[1]) / 2) * size.height]
  const dx = (b[0] - a[0]) * size.width
  const dy = (b[1] - a[1]) * size.height
  const length = Math.hypot(dx, dy) || 1

  const flip = zone.direction === 'b_to_a' ? -1 : 1
  const nx = ((-dy / length) * 26) * flip
  const ny = ((dx / length) * 26) * flip
  const tip = [mid[0] + nx, mid[1] + ny]
  const colour = typeOf(zone).colour

  ctx.strokeStyle = colour
  ctx.fillStyle = colour
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(mid[0], mid[1])
  ctx.lineTo(tip[0], tip[1])
  ctx.stroke()

  const angle = Math.atan2(ny, nx)
  ctx.beginPath()
  ctx.moveTo(tip[0], tip[1])
  ctx.lineTo(tip[0] - 9 * Math.cos(angle - 0.4), tip[1] - 9 * Math.sin(angle - 0.4))
  ctx.lineTo(tip[0] - 9 * Math.cos(angle + 0.4), tip[1] - 9 * Math.sin(angle + 0.4))
  ctx.closePath()
  ctx.fill()

  if (zone.direction === 'both') {
    ctx.beginPath()
    ctx.moveTo(mid[0], mid[1])
    ctx.lineTo(mid[0] - nx, mid[1] - ny)
    ctx.stroke()
  }
}

function findZoneAt(zones, point) {
  for (let z = zones.length - 1; z >= 0; z -= 1) {
    const points = zones[z].points
    if (points.length < 3) continue
    let inside = false
    for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
      const [xi, yi] = points[i]
      const [xj, yj] = points[j]
      if (yi > point.y !== yj > point.y
        && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi) inside = !inside
    }
    if (inside) return z
  }
  return -1
}
