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

export default function ZoneProperties({ zone, onChange, onDelete }) {
  const spec = typeOf(zone)
  const alerts = emitsEvents(zone)
  const set = (patch) => onChange({ ...zone, ...patch })

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
