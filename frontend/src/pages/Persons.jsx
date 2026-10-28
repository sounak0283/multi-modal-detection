import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardFoot,
  CardHead,
  Drawer,
  EmptyState,
  Field,
  Input,
} from '../components/ui'
import { api } from '../api'
import { useToast } from '../components/Toast'

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

// One photo per pose is required (backend/src/perimeter/api/app.py's REQUIRED_POSES) -
// a single canonical photo matched poorly against a live, off-angle boundary crossing;
// matching against whichever of 5 stored poses is closest fixes that.
const POSES = [
  { key: 'front', label: 'Front', hint: 'Looking straight at the camera' },
  { key: 'left', label: 'Left', hint: 'Head turned to their left' },
  { key: 'right', label: 'Right', hint: 'Head turned to their right' },
  { key: 'up', label: 'Up', hint: 'Chin tilted up' },
  { key: 'down', label: 'Down', hint: 'Chin tilted down' },
]

/* Admin-only enrolment (Expansion Plan Phase F). Mirrors Users.jsx's list + detail-panel
 * shape. Each pose photo is sent as base64 and embedded server-side (YuNet + SFace) -
 * this page never computes or holds a face embedding itself. `identityEnabled` reflects
 * the deployment's process-wide `identity.enabled` kill switch (app.yaml/`.env`, restart
 * to change) - enrolment is disabled here, not hidden, when it's off, so an admin can see
 * *why* rather than wonder where the page went. */
export default function Persons({ identityEnabled }) {
  const [persons, setPersons] = useState([])
  const [name, setName] = useState('')
  const [photos, setPhotos] = useState({}) // { [poseKey]: { dataUrl, fileName } }
  const [enrolling, setEnrolling] = useState(false)
  const [busyId, setBusyId] = useState(null)
  const [inspectingId, setInspectingId] = useState(null)
  const fileInputRef = useRef(null)
  const toast = useToast()
  const allPosesCaptured = POSES.every((pose) => photos[pose.key])

  const load = useCallback(async () => {
    try {
      const body = await api.listPersons()
      setPersons(body.persons ?? [])
    } catch (err) {
      toast(err.message, 'error')
    }
  }, [toast])

  useEffect(() => {
    load()
  }, [load])

  // One picker, exactly 5 files at once - selection order maps onto POSES in order
  // (front/left/right/up/down), same as the 5 labelled slots this replaced.
  const onPickPhotos = async (e) => {
    const files = Array.from(e.target.files ?? [])
    if (!files.length) return
    if (files.length !== POSES.length) {
      toast(`Select exactly ${POSES.length} photos, in the order front/left/right/up/down.`, 'error')
      e.target.value = ''
      return
    }
    try {
      const dataUrls = await Promise.all(files.map(readAsDataUrl))
      const next = {}
      POSES.forEach((pose, i) => {
        next[pose.key] = { dataUrl: dataUrls[i], fileName: files[i].name }
      })
      setPhotos(next)
    } catch {
      toast('Could not read one of those photos.', 'error')
      e.target.value = ''
    }
  }

  const enrol = async () => {
    if (!name.trim() || !allPosesCaptured) return
    setEnrolling(true)
    try {
      const poses = Object.fromEntries(
        POSES.map((pose) => [pose.key, photos[pose.key].dataUrl]),
      )
      await api.createPerson({ name: name.trim(), poses })
      toast(`Enrolled ${name.trim()}.`, 'ok')
      setName('')
      setPhotos({})
      if (fileInputRef.current) fileInputRef.current.value = ''
      await load()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setEnrolling(false)
    }
  }

  const toggleActive = async (person) => {
    setBusyId(person.id)
    try {
      await api.updatePerson(person.id, { active: !person.active })
      await load()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setBusyId(null)
    }
  }

  const remove = async (person) => {
    if (!window.confirm(`Remove "${person.name}"? This also deletes their stored face data and cannot be undone.`)) return
    setBusyId(person.id)
    try {
      await api.deletePerson(person.id)
      toast(`${person.name} removed.`, 'ok')
      await load()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setBusyId(null)
    }
  }

  const inspected = persons.find((p) => p.id === inspectingId) ?? null

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <Card>
        <CardHead
          title="Enrolled people"
          aside={<Badge>{persons.length}</Badge>}
        />
        {persons.length === 0 ? (
          <EmptyState>No one is enrolled yet.</EmptyState>
        ) : (
          <ul className="grid grid-cols-1 gap-3 p-3 sm:grid-cols-2 2xl:grid-cols-3">
            {persons.map((person) => (
              <li key={person.id}>
                <button
                  type="button"
                  onClick={() => setInspectingId(person.id)}
                  className="flex w-full items-center gap-3 rounded-lg border border-ink-700 bg-ink-850 p-3 text-left transition-colors hover:border-ink-hover hover:bg-ink-800"
                >
                  <Avatar name={person.name} active={person.active} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] font-medium">{person.name}</span>
                    {person.external_id && (
                      <span className="block truncate font-mono text-[10.5px] text-ink-400">
                        {person.external_id}
                      </span>
                    )}
                    <span className="mt-1.5 block">
                      <Badge tone={person.active ? 'ok' : 'neutral'}>
                        {person.active ? 'Active' : 'Inactive'}
                      </Badge>
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card>
        <CardHead title="Enrol someone new" />
        <CardBody className="space-y-3.5">
          {!identityEnabled && (
            <p className="rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-[11.5px] leading-snug text-ink-400">
              Identity is currently off for this deployment (identity.enabled in
              config/app.yaml). Enrolment is disabled until it's turned on and the
              server is restarted — the rest of the product works without it.
            </p>
          )}
          <Field label="Name">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Full name"
              disabled={!identityEnabled}
            />
          </Field>
          <Field
            label="Photos"
            hint={`Select exactly ${POSES.length} photos at once, in this order: ${POSES.map((p) => p.label.toLowerCase()).join(' → ')}.`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={onPickPhotos}
              disabled={!identityEnabled}
              className="block w-full text-[12px] text-ink-300 file:mr-3 file:rounded-md file:border-0 file:bg-ink-700 file:px-3 file:py-1.5 file:text-[12px] file:text-ink-100"
            />
          </Field>
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
            {POSES.map((pose) => (
              <div key={pose.key} className="text-center">
                <div className="mx-auto h-16 w-16 overflow-hidden rounded-lg border border-ink-700 bg-ink-800">
                  {photos[pose.key] && (
                    <img
                      src={photos[pose.key].dataUrl}
                      alt={`${pose.label} preview`}
                      className="h-full w-full object-cover"
                    />
                  )}
                </div>
                <p className="mt-1 text-[11px] text-ink-300">
                  {pose.label} {photos[pose.key] && '✓'}
                </p>
              </div>
            ))}
          </div>
        </CardBody>
        <CardFoot>
          <Button
            variant="primary"
            className="w-full"
            onClick={enrol}
            disabled={!identityEnabled || enrolling || !name.trim() || !allPosesCaptured}
          >
            {enrolling
              ? 'Enrolling…'
              : allPosesCaptured
                ? 'Enrol'
                : `Enrol (${POSES.filter((p) => photos[p.key]).length}/${POSES.length} photos)`}
          </Button>
        </CardFoot>
      </Card>

      {inspected && (
        <Drawer title={inspected.name} onClose={() => setInspectingId(null)}>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Avatar name={inspected.name} active={inspected.active} large />
              <div className="min-w-0">
                <p className="truncate text-[15px] font-semibold">{inspected.name}</p>
                {inspected.external_id && (
                  <p className="truncate font-mono text-[11px] text-ink-400">{inspected.external_id}</p>
                )}
              </div>
              <Badge tone={inspected.active ? 'ok' : 'neutral'} className="ml-auto">
                {inspected.active ? 'Active' : 'Inactive'}
              </Badge>
            </div>
            <p className="text-[12px] leading-snug text-ink-400">
              An inactive person is no longer matched, so entering a restricted boundary raises an
              unauthorised-access alert. Removing someone also deletes their stored face data.
            </p>
            <div className="flex flex-wrap gap-2">
              <Button disabled={busyId === inspected.id} onClick={() => toggleActive(inspected)}>
                {inspected.active ? 'Deactivate' : 'Activate'}
              </Button>
              <Button
                variant="danger"
                disabled={busyId === inspected.id}
                onClick={async () => {
                  await remove(inspected)
                  setInspectingId(null)
                }}
              >
                Remove
              </Button>
            </div>
          </div>
        </Drawer>
      )}
    </div>
  )
}

function Avatar({ name, active, large = false }) {
  const initials = (name || '?')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join('')
  return (
    <span
      className={`grid shrink-0 place-items-center rounded-full border font-mono font-semibold ${
        large ? 'h-12 w-12 text-[15px]' : 'h-10 w-10 text-[12px]'
      } ${active ? 'border-brand-500/40 bg-brand-900 text-brand-400' : 'border-ink-600 bg-ink-800 text-ink-400'}`}
    >
      {initials}
    </span>
  )
}
