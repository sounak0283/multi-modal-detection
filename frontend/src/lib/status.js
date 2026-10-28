/* One colour map for severity and event pills, so "high" and "entry" look identical on
 * the History table, the Evidence viewer, the Live feed and the Recognition log.
 *
 * `tone` names a Badge tone (see components/ui.jsx); it is the only place a status maps
 * to a colour. Keys are the values the backend actually emits. */

export const SEVERITY = {
  critical: { tone: 'alarm', label: 'Critical' },
  high: { tone: 'alarm', label: 'High' },
  medium: { tone: 'warn', label: 'Medium' },
  low: { tone: 'info', label: 'Low' },
}

export const severityOf = (severity) => SEVERITY[severity] ?? { tone: 'neutral', label: severity || '—' }

/** Event pill tone from an event row: fire/smoke critical, entry safe, PPE amber,
 * crowd cyan, everything else amber like an exit or a generic boundary breach. */
export function eventTone(event) {
  const kind = event?.kind
  const sub = String(event?.subtype ?? '').toLowerCase()
  if (kind === 'fire') return 'alarm'
  if (kind === 'smoke') return 'warn'
  if (kind === 'health' || kind === 'welding' || sub.includes('crowd') || sub.includes('dispers')) return 'info'
  if (sub === 'entry') return 'ok'
  return 'warn'
}
