/* Fire/smoke alerts an operator has acknowledged, remembered in this browser.
 *
 * The server keeps recent alerts in memory until it restarts, and the popup re-reads
 * them on every page load - without this, a reload brought back every alert that had
 * already been acknowledged. Per browser (not shared between operators), capped so it
 * cannot grow forever. */

const KEY = 'perimeter.ackedAlerts'
const CAP = 500

export const alertKey = (event) => `${event.ts}|${event.kind}|${event.message}`

function load() {
  try {
    const list = JSON.parse(localStorage.getItem(KEY) || '[]')
    return Array.isArray(list) ? list : []
  } catch {
    return []
  }
}

export function isAcknowledged(event) {
  return load().includes(alertKey(event))
}

export function acknowledge(events) {
  const list = load()
  for (const event of events) {
    const key = alertKey(event)
    if (!list.includes(key)) list.push(key)
  }
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(-CAP)))
  } catch {
    /* storage blocked - acknowledged for this page view only */
  }
}
