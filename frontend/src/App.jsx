import { useCallback, useEffect, useState } from 'react'
import Shell from './components/Shell'
import LiveView from './pages/LiveView'
import Boundaries from './pages/Boundaries'
import CameraSettings from './pages/CameraSettings'
import AlertHistory from './pages/AlertHistory'
import { Button } from './components/ui'
import { ToastProvider, useToast } from './components/Toast'
import { api } from './api'
import { usePolling } from './lib/usePolling'

const VIEWS = {
  live: { title: 'Live view', subtitle: 'Detections and boundaries as they happen.' },
  boundaries: { title: 'Boundaries', subtitle: 'Draw the areas and lines that raise alerts.' },
  camera: { title: 'Camera', subtitle: 'Source, resolution and capture rate.' },
  history: { title: 'Alert history', subtitle: 'Every alert this system has recorded.' },
}

function Dashboard() {
  const [view, setView] = useState('live')
  const [zones, setZones] = useState([])
  const [selected, setSelected] = useState(-1)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  const { data: health } = usePolling(api.health, 2000)

  const loadZones = useCallback(async () => {
    try {
      const data = await api.getZones()
      setZones(data.zones ?? [])
      if (data.error) {
        toast(`Boundary file rejected — previous config kept: ${data.error}`, 'error')
      }
    } catch {
      /* the status rail already shows the backend is unreachable */
    }
  }, [toast])

  useEffect(() => {
    loadZones()
  }, [loadZones])

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
      const result = await api.saveZones(zones)
      setDirty(false)
      const noun = result.saved === 1 ? 'boundary' : 'boundaries'
      toast(`Saved ${result.saved} ${noun}. Applied without a restart.`, 'ok')
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  const actions =
    view === 'boundaries' ? (
      <Button id="btn-save" variant="primary" onClick={save} disabled={!dirty || saving}>
        {saving ? 'Saving…' : 'Save boundaries'}
      </Button>
    ) : null

  return (
    <Shell
      view={view}
      onNavigate={setView}
      health={health}
      title={VIEWS[view].title}
      subtitle={VIEWS[view].subtitle}
      actions={actions}
    >
      {view === 'live' && <LiveView zones={zones} />}
      {view === 'boundaries' && (
        <Boundaries
          zones={zones}
          setZones={setZones}
          selected={selected}
          setSelected={setSelected}
          markDirty={() => setDirty(true)}
        />
      )}
      {view === 'camera' && <CameraSettings onSaved={loadZones} />}
      {view === 'history' && <AlertHistory zones={zones} />}
    </Shell>
  )
}

export default function App() {
  return (
    <ToastProvider>
      <Dashboard />
    </ToastProvider>
  )
}
