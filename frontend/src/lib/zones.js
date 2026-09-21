/* Boundary type metadata, shared by the canvas, the list and the toolbar so the
 * colour-to-meaning mapping is learned once and never contradicts itself.
 */

export const ZONE_TYPES = {
  polygon: {
    label: 'Area',
    hint: 'entry / exit',
    colour: '#34d399',
    area: true,
    minPoints: 3,
    description: 'Alerts when someone enters or leaves the enclosed area.',
  },
  tripwire: {
    label: 'Tripwire',
    hint: 'directional line',
    colour: '#fbbf24',
    area: false,
    minPoints: 2,
    description: 'Alerts when someone crosses the line, optionally in one direction only.',
  },
  exclusion: {
    label: 'Exclusion',
    hint: 'ignore region',
    colour: '#f87171',
    area: true,
    minPoints: 3,
    description:
      'Suppresses detections inside this region. The highest-value setting here: the worst '
      + 'false alarms come from fixed sources such as a welding bay or a rotating beacon.',
  },
  fire_roi: {
    label: 'Fire ROI',
    hint: 'extra sensitive',
    colour: '#fb923c',
    area: true,
    minPoints: 3,
    description:
      'Lowers the fire and smoke threshold inside this region, for places where a missed '
      + 'fire costs more than a false alarm.',
  },
}

// What the user draws: one generic shape. It only becomes a specific kind of boundary
// (people entry/exit, fire/smoke sensitivity, ignore region) once a model is chosen for it.
ZONE_TYPES.draft = {
  label: 'Boundary',
  hint: 'draw, then choose the model',
  colour: '#60a5fa',
  area: true,
  minPoints: 2,
  description: 'Click the corners of the area, or two points for a line.',
}

// What each kind of boundary is for, grouped by the model that acts on it.
export const MODEL_CHOICES = [
  { group: 'People detection', type: 'polygon', label: 'Alert on entry / exit of the area', needs: 3 },
  { group: 'People detection', type: 'tripwire', label: 'Alert when the line is crossed', needs: 2 },
  { group: 'Fire / smoke detection', type: 'fire_roi', label: 'More sensitive inside the area', needs: 3 },
  { group: 'Fire / smoke detection', type: 'exclusion', label: 'Ignore detections inside the area', needs: 3 },
]

export const DAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

export const typeOf = (zone) => ZONE_TYPES[zone?.type] ?? ZONE_TYPES.polygon

export const emitsEvents = (zone) => zone?.type === 'polygon' || zone?.type === 'tripwire'

export function nextZoneId(zones, type) {
  let n = 1
  while (zones.some((z) => z.id === `${type}_${n}`)) n += 1
  return `${type}_${n}`
}

export function newZone(type, points, zones) {
  const zone = {
    id: nextZoneId(zones, type),
    type,
    name: '',
    points,
    severity: 'medium',
  }
  if (emitsEvents(zone)) {
    zone.detect = ['person']
    zone.events = ['entry', 'exit']
    if (type === 'tripwire') zone.direction = 'both'
  } else {
    // An exclusion or ROI that applies to nothing is silently useless, and the API
    // rejects it - so default to the classes each one exists to affect.
    zone.applies_to = ['fire', 'smoke']
    if (type === 'fire_roi') zone.conf_delta = -0.08
  }
  return zone
}

/** A drawn shape becomes a line for two points and an area for three or more; the default
 * model for it is people detection (the common case), changeable afterwards. */
export const typeForPoints = (points) => (points.length === 2 ? 'tripwire' : 'polygon')

/** Re-purpose a boundary for another model, keeping what is not model-specific. */
export function convertZone(zone, type) {
  const fresh = newZone(type, zone.points, [])
  return {
    ...fresh,
    id: zone.id,
    name: zone.name,
    severity: zone.severity ?? fresh.severity,
    ...(zone.schedule ? { schedule: zone.schedule } : {}),
  }
}
