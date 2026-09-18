/* Thin API client.
 *
 * Every call funnels through `request` so a backend that is down produces one
 * predictable Error shape rather than a mix of TypeErrors and JSON parse failures
 * scattered across the pages.
 *
 * Multi-camera (Expansion Plan Phase A): almost everything that used to be a single
 * global resource (camera settings, zones, the live stream) is now scoped to a
 * `cameraId`, since any number of cameras can exist and each has its own configuration.
 *
 * Auth (Expansion Plan Phase B): every route now requires a session cookie, which
 * same-origin `fetch` already sends automatically - no client-side change needed for
 * that. What IS needed: the moment any call comes back 401 (session expired, or never
 * logged in), the app should fall back to the login screen rather than every poller
 * independently erroring forever. `onUnauthorized` is that one hook.
 */

let onUnauthorized = null

/** Registered once by App.jsx. Called (in addition to the normal throw) on every 401,
 * so a session expiring mid-use drops back to the login screen immediately rather than
 * on next full reload. */
export function setUnauthorizedHandler(fn) {
  onUnauthorized = fn
}

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

  if (response.status === 401) onUnauthorized?.()

  if (!response.ok) {
    // FastAPI puts the useful message in `detail`; the zone validator's messages are
    // written to be shown to an installer verbatim, so they are surfaced unchanged.
    const problem = await response.json().catch(() => ({}))
    throw new Error(problem.detail || `Request failed (${response.status})`)
  }
  return response.status === 204 ? null : response.json()
}

const withParams = (base, params) => {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== '' && v != null),
  )
  const qs = query.toString()
  return qs ? `${base}?${qs}` : base
}

export const api = {
  health: () => request('/api/health'),

  // -- auth (Expansion Plan Phase B) -----------------------------------------
  login: (email, password) =>
    request('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  logout: () => request('/api/auth/logout', { method: 'POST' }),
  me: () => request('/api/auth/me'),

  // -- users (admin only) -----------------------------------------------------
  listUsers: () => request('/api/users'),
  createUser: (payload) => request('/api/users', { method: 'POST', body: JSON.stringify(payload) }),
  updateUser: (userId, payload) =>
    request(`/api/users/${encodeURIComponent(userId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),
  deleteUser: (userId) =>
    request(`/api/users/${encodeURIComponent(userId)}`, { method: 'DELETE' }),

  // -- persons / identity gallery (Expansion Plan Phase F) ---------------------
  listPersons: () => request('/api/persons'),
  createPerson: (payload) =>
    request('/api/persons', { method: 'POST', body: JSON.stringify(payload) }),
  updatePerson: (personId, payload) =>
    request(`/api/persons/${encodeURIComponent(personId)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deletePerson: (personId) =>
    request(`/api/persons/${encodeURIComponent(personId)}`, { method: 'DELETE' }),

  // -- alert config (Expansion Plan Phase D) -----------------------------------
  getAlertConfig: () => request('/api/alert-config'),
  saveAlertConfig: (config) =>
    request('/api/alert-config', { method: 'PUT', body: JSON.stringify(config) }),

  // -- cameras --------------------------------------------------------------
  listCameras: () => request('/api/cameras'),
  createCamera: (payload) =>
    request('/api/cameras', { method: 'POST', body: JSON.stringify(payload) }),
  getCamera: (cameraId) => request(`/api/cameras/${cameraId}`),
  updateCamera: (cameraId, payload) =>
    request(`/api/cameras/${cameraId}`, { method: 'PUT', body: JSON.stringify(payload) }),
  deleteCamera: (cameraId) => request(`/api/cameras/${cameraId}`, { method: 'DELETE' }),
  testCamera: (payload) =>
    request('/api/cameras/test', { method: 'POST', body: JSON.stringify(payload) }),

  // -- zones (per camera) -----------------------------------------------------
  getZones: (cameraId) => request(`/api/cameras/${cameraId}/zones`),
  saveZones: (cameraId, zones) =>
    request(`/api/cameras/${cameraId}/zones`, { method: 'PUT', body: JSON.stringify({ zones }) }),
  validateZone: (type, points) =>
    request('/api/zones/validate', { method: 'POST', body: JSON.stringify({ type, points }) }),

  // -- events (across cameras, optionally filtered) ----------------------------
  // /events is what has been RECORDED; /events/live is what is HAPPENING. They diverge
  // exactly when storage is broken, which is when the difference matters most.
  events: ({ limit = 200, kind = '', zoneId = '', cameraId = '', identityStatus = '' } = {}) =>
    request(
      withParams('/api/events', {
        limit,
        kind,
        zone_id: zoneId,
        camera_id: cameraId,
        identity_status: identityStatus,
      }),
    ),
  liveEvents: (limit = 20, cameraId = '') =>
    request(withParams('/api/events/live', { limit, camera_id: cameraId })),
  eventsSummary: (cameraId = '') => request(withParams('/api/events/summary', { camera_id: cameraId })),

  // Dev-only: 404s unless the server has PERIMETER_DEV_TOOLS=true. Injects a fake fire/smoke
  // event so the popup can be exercised before a real detector produces one.
  simulateAlert: (kind, cameraId) =>
    request('/api/dev/simulate-alert', {
      method: 'POST',
      body: JSON.stringify({ kind, camera_id: cameraId }),
    }),
}

export const streamUrl = (cameraId, bust) => `/api/cameras/${cameraId}/stream?t=${bust}`
export const snapshotUrl = (cameraId, bust) => `/api/cameras/${cameraId}/snapshot?t=${bust}`

// Evidence (Expansion Plan Phase C). Plain URLs, not routed through request() - an
// <img>/<video> tag sends the session cookie automatically for a same-origin request,
// same reasoning already applied to the live snapshot/stream URLs above.
export const eventSnapshotUrl = (eventId) => `/api/events/${eventId}/snapshot`
export const eventClipUrl = (eventId) => `/api/events/${eventId}/clip`
