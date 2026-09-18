import { useEffect, useState } from 'react'
import {
  Button,
  Card,
  CardBody,
  CardFoot,
  CardHead,
  ChipGroup,
  Field,
  Input,
  Select,
  Toggle,
} from './ui'
import { api } from '../api'
import { DAYS, emitsEvents, typeOf } from '../lib/zones'

const DETECT_OPTIONS = [
  { value: 'person', label: 'People' },
  { value: 'fire', label: 'Fire' },
  { value: 'smoke', label: 'Smoke' },
]

const EVENT_OPTIONS = [
  { value: 'entry', label: 'Entry' },
  { value: 'exit', label: 'Exit' },
]

// Matches backend/src/perimeter/detect/ppe.py:PPE_ITEMS (Expansion Plan Phase H).
const PPE_OPTIONS = [
  { value: 'helmet', label: 'Helmet' },
  { value: 'vest', label: 'Vest' },
  { value: 'gloves', label: 'Gloves' },
  { value: 'shoes', label: 'Shoes' },
  { value: 'glasses', label: 'Glasses' },
]

export default function ZoneProperties({ zone, onChange, onDelete }) {
  const spec = typeOf(zone)
  const alerts = emitsEvents(zone)
  const set = (patch) => onChange({ ...zone, ...patch })

  // Restricted-zone allow-list (Expansion Plan Phase F.1). Fetched once per mount, not
  // per-zone-switch - the person list rarely changes mid-edit and a plain per-zone
  // fetch would just be a re-fetch of the same data on every click through the zone list.
  const [persons, setPersons] = useState([])
  useEffect(() => {
    api
      .listPersons()
      .then((body) => setPersons(body.persons ?? []))
      .catch(() => setPersons([]))
  }, [])

  const restrictedIds = zone.authorized_person_ids ?? []
  const toggleAuthorized = (personId) => {
    const next = restrictedIds.includes(personId)
      ? restrictedIds.filter((id) => id !== personId)
      : [...restrictedIds, personId]
    set({ authorized_person_ids: next })
  }

  const scheduled = Boolean(zone.schedule?.from)
  const days = zone.schedule?.days ?? DAYS

  const setSchedule = (patch) =>
    set({ schedule: { days, from: '18:00', to: '06:00', ...zone.schedule, ...patch } })

  return (
    <Card>
      <CardHead title="Properties" aside={<span className="text-[11px] text-ink-400">{spec.label}</span>} />
      <CardBody className="space-y-3.5">
        <Field label="Name">
          <Input
            id="f-name"
            value={zone.name || ''}
            placeholder={zone.id}
            onChange={(e) => set({ name: e.target.value })}
          />
        </Field>

        <Field label={alerts ? 'Detect' : 'Applies to'}>
          <ChipGroup
            options={DETECT_OPTIONS}
            value={(alerts ? zone.detect : zone.applies_to) ?? []}
            onChange={(value) => set(alerts ? { detect: value } : { applies_to: value })}
          />
        </Field>

        {alerts && (
          <Field label="Alert on">
            <ChipGroup
              options={EVENT_OPTIONS}
              value={zone.events ?? []}
              onChange={(events) => set({ events })}
            />
          </Field>
        )}

        {alerts && persons.length > 0 && (
          <Field
            label="Restrict entry to"
            hint="Leave empty to allow anyone. Otherwise, anyone else raises a critical, unauthorised-access alert on entry (Expansion Plan Phase F.1)."
          >
            <div className="max-h-40 space-y-1 overflow-y-auto rounded-lg border border-ink-700 p-2">
              {persons.map((person) => (
                <label
                  key={person.id}
                  className="flex items-center gap-2 rounded px-1.5 py-1 text-[12.5px] hover:bg-ink-800"
                >
                  <input
                    type="checkbox"
                    checked={restrictedIds.includes(person.id)}
                    onChange={() => toggleAuthorized(person.id)}
                  />
                  <span className="truncate">{person.name}</span>
                  {!person.active && <span className="text-ink-500">(inactive)</span>}
                </label>
              ))}
            </div>
          </Field>
        )}

        {zone.type === 'tripwire' && (
          <Field label="Direction" hint="Which way across the line counts as an entry.">
            <Select value={zone.direction || 'both'} onChange={(e) => set({ direction: e.target.value })}>
              <option value="both">Both ways</option>
              <option value="a_to_b">A → B only</option>
              <option value="b_to_a">B → A only</option>
            </Select>
          </Field>
        )}

        {zone.type === 'fire_roi' && (
          <Field
            label="Confidence delta"
            hint="Negative is more sensitive inside this area."
          >
            <Input
              type="number"
              step="0.01"
              min="-0.5"
              max="0.5"
              value={zone.conf_delta ?? -0.08}
              onChange={(e) => set({ conf_delta: Number(e.target.value) })}
            />
          </Field>
        )}

        {alerts && (
          <div className="grid grid-cols-2 gap-3">
            <Field label="Severity">
              <Select value={zone.severity || 'medium'} onChange={(e) => set({ severity: e.target.value })}>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </Select>
            </Field>
            <Field
              label="Hysteresis"
              hint="Detection frames, not camera frames."
            >
              <Input
                type="number"
                min="1"
                max="60"
                placeholder="4"
                value={zone.min_frames ?? ''}
                onChange={(e) =>
                  set({ min_frames: e.target.value ? Number(e.target.value) : undefined })
                }
              />
            </Field>
          </div>
        )}

        {zone.type === 'polygon' && (
          <fieldset className="rounded-lg border border-ink-700 p-3">
            <legend className="px-1.5 text-[11px] uppercase tracking-[0.06em] text-ink-400">
              Crowd formation
            </legend>
            <Toggle
              checked={zone.crowd_threshold != null}
              label="Alert when people cluster here"
              onChange={(on) =>
                set(
                  on
                    ? { crowd_threshold: 5 }
                    : { crowd_threshold: undefined, crowd_min_frames: undefined },
                )
              }
            />

            {zone.crowd_threshold != null && (
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Field
                  label="Threshold"
                  hint="People clustered together, not just anyone in the zone."
                >
                  <Input
                    type="number"
                    min="2"
                    max="100"
                    value={zone.crowd_threshold}
                    onChange={(e) =>
                      set({ crowd_threshold: Math.max(2, Number(e.target.value) || 2) })
                    }
                  />
                </Field>
                <Field label="Hysteresis" hint="Detection frames, not camera frames.">
                  <Input
                    type="number"
                    min="1"
                    max="60"
                    placeholder="4"
                    value={zone.crowd_min_frames ?? ''}
                    onChange={(e) =>
                      set({
                        crowd_min_frames: e.target.value ? Number(e.target.value) : undefined,
                      })
                    }
                  />
                </Field>
              </div>
            )}
          </fieldset>
        )}

        {zone.type === 'polygon' && (
          <fieldset className="rounded-lg border border-ink-700 p-3">
            <legend className="px-1.5 text-[11px] uppercase tracking-[0.06em] text-ink-400">
              PPE required
            </legend>
            <ChipGroup
              options={PPE_OPTIONS}
              value={zone.ppe_required ?? []}
              onChange={(ppe_required) => set({ ppe_required })}
            />
            <p className="mt-2 text-[11.5px] leading-snug text-ink-400">
              Anyone confirmed missing a selected item raises an alert. No trained model
              ships in this repo yet (Expansion Plan Phase H) — checking stays inactive
              until one is placed at <code>backend/models/ppe/model.onnx</code>.
            </p>
          </fieldset>
        )}

        <fieldset className="rounded-lg border border-ink-700 p-3">
          <legend className="px-1.5 text-[11px] uppercase tracking-[0.06em] text-ink-400">
            Active hours
          </legend>
          <Toggle
            checked={scheduled}
            label="Only alert during set hours"
            onChange={(on) =>
              on ? setSchedule({}) : set({ schedule: undefined })
            }
          />

          {scheduled && (
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <Field label="From">
                  <Input
                    type="time"
                    value={zone.schedule.from}
                    onChange={(e) => setSchedule({ from: e.target.value })}
                  />
                </Field>
                <Field label="To">
                  <Input
                    type="time"
                    value={zone.schedule.to}
                    onChange={(e) => setSchedule({ to: e.target.value })}
                  />
                </Field>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {DAYS.map((day) => {
                  const on = days.includes(day)
                  return (
                    <button
                      key={day}
                      type="button"
                      onClick={() =>
                        setSchedule({
                          days: on ? days.filter((d) => d !== day) : [...days, day],
                        })
                      }
                      className={`rounded-md border px-2 py-1 text-[11.5px] capitalize transition-colors ${
                        on
                          ? 'border-brand-500 bg-brand-500 font-semibold text-[#04122a]'
                          : 'border-ink-700 bg-ink-800 text-ink-300 hover:border-ink-600'
                      }`}
                    >
                      {day}
                    </button>
                  )
                })}
              </div>

              {zone.schedule.from > zone.schedule.to && (
                <p className="text-[11.5px] text-ink-400">
                  Overnight window — the morning half belongs to the previous day.
                </p>
              )}
            </div>
          )}
        </fieldset>

        <p className="text-[11.5px] leading-snug text-ink-400">{spec.description}</p>
      </CardBody>

      <CardFoot>
        <Button variant="danger" onClick={onDelete} className="w-full">
          Delete boundary
        </Button>
      </CardFoot>
    </Card>
  )
}
