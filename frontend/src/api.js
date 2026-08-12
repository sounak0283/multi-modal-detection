/* Thin API client.
 *
 * Every call funnels through `request` so a backend that is down produces one
 * predictable Error shape rather than a mix of TypeErrors and JSON parse failures
 * scattered across the pages.
 */

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(path, {
      headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
      ...options,
    })
  } catch {
    throw new Error('Cannot reach the server.')
  }

  if (!response.ok) {
    // FastAPI puts the useful message in `detail`; the zone validator's messages are
    // written to be shown to an installer verbatim, so they are surfaced unchanged.
    const problem = await response.json().catch(() => ({}))
    throw new Error(problem.detail || `Request failed (${response.status})`)
  }
  return response.status === 204 ? null : response.json()
}

export const api = {
  health: () => request('/api/health'),

  getZones: () => request('/api/zones'),
  saveZones: (zones) =>
    request('/api/zones', { method: 'PUT', body: JSON.stringify({ zones }) }),
  validateZone: (type, points) =>
    request('/api/zones/validate', { method: 'POST', body: JSON.stringify({ type, points }) }),

  getCamera: () => request('/api/camera'),
  saveCamera: (payload) =>
    request('/api/camera', { method: 'PUT', body: JSON.stringify(payload) }),
  testCamera: (payload) =>
    request('/api/camera/test', { method: 'POST', body: JSON.stringify(payload) }),

  // /events is what has been RECORDED; /events/live is what is HAPPENING. They diverge
  // exactly when storage is broken, which is when the difference matters most.
  events: ({ limit = 200, kind = '', zoneId = '' } = {}) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (kind) params.set('kind', kind)
    if (zoneId) params.set('zone_id', zoneId)
    return request(`/api/events?${params}`)
  },
  liveEvents: (limit = 20) => request(`/api/events/live?limit=${limit}`),
  eventsSummary: () => request('/api/events/summary'),
}

export const streamUrl = (bust) => `/api/stream?t=${bust}`
export const snapshotUrl = (bust) => `/api/snapshot?t=${bust}`
