import { useCallback, useEffect, useState } from 'react'
import Shell from './components/Shell'
import LiveView from './pages/LiveView'
import LiveGrid from './pages/LiveGrid'
import Boundaries from './pages/Boundaries'
import Cameras from './pages/Cameras'
import Users from './pages/Users'
import Persons from './pages/Persons'
import RecognitionLog from './pages/RecognitionLog'
import Alerts from './pages/Alerts'
import AlertHistory from './pages/AlertHistory'
import Login from './pages/Login'
import { Button, Select } from './components/ui'
import { ToastProvider, useToast } from './components/Toast'
import FireAlertOverlay from './components/FireAlertOverlay'
import AlertSoundPlayer from './components/AlertSoundPlayer'
import { api, setUnauthorizedHandler } from './api'
import { usePolling } from './lib/usePolling'

const VIEWS = {
  live: { title: 'Live view', subtitle: 'Detections and boundaries as they happen.' },
  boundaries: { title: 'Boundaries', subtitle: 'Draw the areas and lines that raise alerts.' },
  cameras: { title: 'Cameras', subtitle: 'Add cameras and choose which modules run on each.' },
  alerts: { title: 'Alerts', subtitle: 'Who gets emailed, and the browser alert sound.' },
  persons: { title: 'People', subtitle: 'Enrol faces for restricted-zone access control.' },
  recognition: { title: 'Recognition log', subtitle: 'Every identity resolution, with camera and timestamp.' },
  users: { title: 'Accounts', subtitle: 'Who can sign in, and what they can change.' },
  history: { title: 'Alert history', subtitle: 'Every alert this system has recorded.' },
}

// Views where the per-camera boundary/zone data is relevant and a camera switcher makes
// sense in the header. "cameras" manages every camera at once; "history" can span all
// cameras at once (its own filter row already covers that).
const NEEDS_CAMERA_SWITCHER = new Set(['live', 'boundaries'])

function Dashboard({ user, onLogout }) {
  const isAdmin = user?.role === 'admin'
  const [view, setView] = useState('live')
  const [liveGrid, setLiveGrid] = useState(false)
  const [cameras, setCameras] = useState([])
  const [cameraId, setCameraId] = useState(null)
  const [zones, setZones] = useState([])
  const [selected, setSelected] = useState(-1)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  const { data: health } = usePolling(api.health, 2000)

  const loadCameras = useCallback(async () => {
    try {
      const body = await api.listCameras()
      const list = body.cameras ?? []
      setCameras(list)
      setCameraId((current) => {
        if (current && list.some((c) => c.id === current)) return current
        return list[0]?.id ?? null
      })
    } catch {
      /* the status rail already shows the backend is unreachable */
    }
  }, [])

  useEffect(() => {
    loadCameras()
    const id = setInterval(loadCameras, 5000)
    return () => clearInterval(id)
  }, [loadCameras])

  const loadZones = useCallback(async () => {
    if (!cameraId) {
      setZones([])
      return
    }
    try {
      const data = await api.getZones(cameraId)
      setZones(data.zones ?? [])
      if (data.error) {
        toast(`Boundary config rejected — previous config kept: ${data.error}`, 'error')
      }
    } catch {
      /* the status rail already shows the backend is unreachable */
    }
  }, [cameraId, toast])

  useEffect(() => {
    setSelected(-1)
    setDirty(false)
    loadZones()
  }, [cameraId, loadZones])

  // Re-sync from the server while idle, but never clobber unsaved edits.
  useEffect(() => {
    if (dirty) return undefined
    const id = setInterval(loadZones, 10000)
    return () => clearInterval(id)
  }, [dirty, loadZones])

  useEffect(() => {
    if (!dirty) return undefined
    const warn = (event) => {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])

  const save = async () => {
    setSaving(true)
    try {
      const result = await api.saveZones(cameraId, zones)
      setDirty(false)
      const noun = result.saved === 1 ? 'boundary' : 'boundaries'
      toast(`Saved ${result.saved} ${noun}. Applied without a restart.`, 'ok')
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  const cameraSwitcher = NEEDS_CAMERA_SWITCHER.has(view) && cameras.length > 0 && (
    <Select
      value={cameraId ?? ''}
      onChange={(e) => {
        setCameraId(e.target.value)
        setLiveGrid(false)
      }}
      className="w-auto py-1.5"
    >
      {cameras.map((camera) => (
        <option key={camera.id} value={camera.id}>
          {camera.name || camera.id}
        </option>
      ))}
    </Select>
  )

  // Grid ("see all cameras") only makes sense in Live view - Boundaries edits one
  // camera's zones at a time, so switching it there would have nothing to show.
  const liveModeToggle = view === 'live' && cameras.length > 1 && (
    <div className="flex overflow-hidden rounded-lg border border-ink-700">
      {[
        [false, 'Single'],
        [true, 'All cameras'],
      ].map(([value, label]) => (
        <button
          key={label}
          type="button"
          onClick={() => setLiveGrid(value)}
          className={`px-3 py-1.5 text-[12.5px] transition-colors not-first:border-l not-first:border-ink-700 ${
            liveGrid === value
              ? 'bg-brand-500 font-semibold text-[#04122a]'
              : 'bg-ink-800 text-ink-200 hover:bg-ink-700'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  )

  const actions = (
    <>
      {liveModeToggle}
      {cameraSwitcher}
      {view === 'boundaries' && isAdmin && (
        <Button id="btn-save" variant="primary" onClick={save} disabled={!dirty || saving || !cameraId}>
          {saving ? 'Saving…' : 'Save boundaries'}
        </Button>
      )}
    </>
  )

  const noCameras = cameras.length === 0

  return (
    <>
      <Shell
        view={view}
        onNavigate={setView}
        health={health}
        title={VIEWS[view].title}
        subtitle={VIEWS[view].subtitle}
        actions={actions}
        user={user}
        onLogout={onLogout}
      >
        {view === 'live' &&
          (noCameras ? (
            <NoCameras onGoToCameras={() => setView('cameras')} />
          ) : liveGrid ? (
            <LiveGrid
              cameras={cameras}
              onSelect={(id) => {
                setCameraId(id)
                setLiveGrid(false)
              }}
            />
          ) : (
            // Keyed on cameraId: usePolling's interval keeps running with whatever
            // closure it started with (see lib/usePolling.js), so switching cameras
            // without remounting would keep showing the previous camera's stats until
            // the next tick. A fresh key forces a clean remount and an immediate fetch.
            <LiveView key={cameraId} cameraId={cameraId} zones={zones} />
          ))}
        {view === 'boundaries' &&
          (noCameras ? (
            <NoCameras onGoToCameras={() => setView('cameras')} />
          ) : (
            <Boundaries
              key={cameraId}
              cameraId={cameraId}
              zones={zones}
              setZones={setZones}
              selected={selected}
              setSelected={setSelected}
              markDirty={() => setDirty(true)}
              readOnly={!isAdmin}
            />
          ))}
        {view === 'cameras' && (
          <Cameras devTools={health?.dev_tools} onChanged={loadCameras} readOnly={!isAdmin} />
        )}
        {view === 'alerts' && isAdmin && <Alerts />}
        {view === 'persons' && isAdmin && <Persons identityEnabled={health?.identity_enabled} />}
        {view === 'recognition' && <RecognitionLog cameras={cameras} />}
        {view === 'users' && isAdmin && <Users currentUser={user} />}
        {view === 'history' && <AlertHistory cameras={cameras} zones={zones} />}
      </Shell>
      {/* Mounted outside any single view so a fire/smoke alert surfaces no matter which
          page is open - not just when History or Live view happens to be on screen. */}
      <FireAlertOverlay />
      <AlertSoundPlayer />
    </>
  )
}

function NoCameras({ onGoToCameras }) {
  return (
    <div className="rounded-xl border border-dashed border-ink-700 px-6 py-16 text-center">
      <p className="text-[13px] text-ink-300">No cameras configured yet.</p>
      <Button variant="primary" className="mt-4" onClick={onGoToCameras}>
        Add a camera
      </Button>
    </div>
  )
}

export default function App() {
  // null = signed out (or not checked yet); the boot check below tells the two apart
  // only in that a not-yet-checked session renders nothing rather than flashing Login.
  const [user, setUser] = useState(null)
  const [checked, setChecked] = useState(false)

  useEffect(() => {
    // Registered before the boot check fires, so a 401 anywhere (including this first
    // call) reliably lands the app on the login screen rather than racing it.
    setUnauthorizedHandler(() => setUser(null))
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setChecked(true))
  }, [])

  const logout = async () => {
    try {
      await api.logout()
    } finally {
      setUser(null)
    }
  }

  return (
    <ToastProvider>
      {!checked ? null : user ? (
        <Dashboard user={user} onLogout={logout} />
      ) : (
        <Login onSuccess={setUser} />
      )}
    </ToastProvider>
  )
}
