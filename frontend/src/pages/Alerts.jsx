import { useCallback, useEffect, useState } from 'react'
import {
  Button,
  Card,
  CardBody,
  CardFoot,
  CardHead,
  ChipGroup,
  EmptyState,
  Field,
  Input,
  Select,
  Toggle,
} from '../components/ui'
import { api } from '../api'
import { useToast } from '../components/Toast'

const SEVERITIES = ['low', 'medium', 'high', 'critical']

// Mirrors the "kind" values pipeline.py actually publishes (alerts/router.py matches
// against these verbatim). "phone" isn't wired to a detector yet but is included so a
// rule can be pre-configured before that module ships.
const KIND_OPTIONS = [
  { value: 'boundary', label: 'Boundary entry/exit' },
  { value: 'crowd', label: 'Crowd formation' },
  { value: 'fire', label: 'Fire' },
  { value: 'smoke', label: 'Smoke' },
  { value: 'ppe', label: 'PPE violation' },
  { value: 'access', label: 'Unauthorized access' },
  { value: 'phone', label: 'Phone near ear' },
]

const BLANK_RULE = {
  id: '',
  type: 'smtp',
  to: '',
  min_severity: 'medium',
  cooldown_seconds: 300,
  enabled: true,
  kinds: [],
}

function ruleToForm(rule) {
  return { ...rule, to: rule.to.join(', '), kinds: rule.kinds ?? [] }
}

function formToRule(form) {
  return {
    id: form.id,
    type: form.type,
    to: form.to.split(',').map((addr) => addr.trim()).filter(Boolean),
    min_severity: form.min_severity,
    cooldown_seconds: Number(form.cooldown_seconds) || 0,
    enabled: form.enabled,
    kinds: form.kinds,
  }
}

function kindsSummary(kinds) {
  if (!kinds || kinds.length === 0) return 'All alert types'
  return kinds.join(', ')
}

/* Alert routing rules + the browser "system sound" setting (Expansion Plan Phase D).
 * One document, not per-item CRUD - mirrors how Boundaries.jsx holds a camera's full
 * zone list in memory and PUTs it whole, rather than the per-item CRUD Cameras/Accounts
 * use. Edits here are local until "Save all" - the backend rejects the whole document on
 * one bad rule, so a half-finished edit never reaches Mongo. */
export default function Alerts() {
  const [config, setConfig] = useState(null)
  const [selectedId, setSelectedId] = useState(null) // null | '_new' | an existing rule id
  const [form, setForm] = useState(null)
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  const load = useCallback(async () => {
    try {
      setConfig(await api.getAlertConfig())
    } catch (err) {
      toast(err.message, 'error')
    }
  }, [toast])

  useEffect(() => {
    load()
  }, [load])

  const selectExisting = (rule) => {
    setSelectedId(rule.id)
    setForm(ruleToForm(rule))
  }

  const startNew = () => {
    setSelectedId('_new')
    setForm({ ...BLANK_RULE })
  }

  const set = (patch) => setForm((f) => ({ ...f, ...patch }))

  const saveAll = async (nextConfig) => {
    setSaving(true)
    try {
      const saved = await api.saveAlertConfig(nextConfig)
      setConfig(saved)
      toast('Alert settings saved.', 'ok')
      return true
    } catch (err) {
      toast(err.message, 'error')
      return false
    } finally {
      setSaving(false)
    }
  }

  const saveSoundSettings = (patch) => saveAll({ ...config, ...patch })

  const saveRule = async () => {
    const rule = formToRule(form)
    const others = (config.sinks || []).filter((r) => r.id !== selectedId)
    const ok = await saveAll({ ...config, sinks: [...others, rule] })
    if (ok) {
      setSelectedId(null)
      setForm(null)
    }
  }

  const removeRule = async () => {
    if (selectedId === '_new' || !selectedId) return
    if (!window.confirm('Remove this alert rule? This cannot be undone.')) return
    const remaining = (config.sinks || []).filter((r) => r.id !== selectedId)
    const ok = await saveAll({ ...config, sinks: remaining })
    if (ok) {
      setSelectedId(null)
      setForm(null)
    }
  }

  if (!config) return null

  return (
    <div className="space-y-4">
      <Card>
        <CardHead title="System sound" />
        <CardBody className="space-y-3.5">
          <Toggle
            checked={config.sound_enabled}
            onChange={(sound_enabled) => saveSoundSettings({ sound_enabled })}
            label="Play a sound in the browser when a new alert arrives"
          />
          <Field label="Minimum severity to play a sound" className="max-w-xs">
            <Select
              value={config.sound_min_severity}
              onChange={(e) => saveSoundSettings({ sound_min_severity: e.target.value })}
            >
              {SEVERITIES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </Select>
          </Field>
          <p className="text-[11.5px] leading-snug text-ink-400">
            Browsers require a click somewhere on the page before they'll allow sound to
            play automatically - keep a dashboard tab open and interacted with. Email
            (below) doesn't have this limitation and is the reliable backstop.
          </p>
        </CardBody>
      </Card>

      <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
        <Card>
          <CardHead
            title="Email rules"
            aside={
              <span className="rounded-full bg-ink-700 px-2 py-0.5 text-[11px] text-ink-200">
                {(config.sinks || []).length}
              </span>
            }
          />
          {(config.sinks || []).length === 0 ? (
            <EmptyState>No email alert rules yet.</EmptyState>
          ) : (
            <ul className="p-1.5">
              {config.sinks.map((rule) => (
                <li key={rule.id}>
                  <button
                    type="button"
                    onClick={() => selectExisting(rule)}
                    className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors ${
                      selectedId === rule.id
                        ? 'border-brand-600 bg-brand-900'
                        : 'border-transparent hover:bg-ink-800'
                    }`}
                  >
                    <span
                      className={`h-2 w-2 shrink-0 rounded-full ${rule.enabled ? 'bg-ok-400' : 'bg-ink-600'}`}
                    />
                    <span className="min-w-0 flex-1 truncate text-left">
                      <span className="block truncate text-[13px]">
                        {rule.to.join(', ') || '(no recipients)'}
                      </span>
                      <span className="block truncate text-[10.5px] text-ink-400">
                        {kindsSummary(rule.kinds)}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          <CardFoot>
            <Button className="w-full" onClick={startNew}>
              + Add rule
            </Button>
          </CardFoot>
        </Card>

        {!form ? (
          <Card>
            <EmptyState>Add a rule to get started.</EmptyState>
          </Card>
        ) : (
          <Card>
            <CardHead title={selectedId === '_new' ? 'New email rule' : 'Edit rule'} />
            <CardBody className="space-y-3.5">
              <Field
                label="Recipients"
                hint="Comma-separated. All of them get the same email at the same time."
              >
                <Input
                  value={form.to}
                  onChange={(e) => set({ to: e.target.value })}
                  placeholder="ops@example.com, security@example.com"
                />
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Minimum severity">
                  <Select
                    value={form.min_severity}
                    onChange={(e) => set({ min_severity: e.target.value })}
                  >
                    {SEVERITIES.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </Select>
                </Field>
                <Field label="Cooldown (seconds)" hint="0 = never throttle">
                  <Input
                    type="number"
                    min="0"
                    value={form.cooldown_seconds}
                    onChange={(e) => set({ cooldown_seconds: e.target.value })}
                  />
                </Field>
              </div>
              <Field
                label="Alert types"
                hint="Leave all unselected to match every alert type. Select specific types to route only those to these recipients (e.g. fire/smoke to a safety team, boundary/crowd to security)."
              >
                <ChipGroup
                  options={KIND_OPTIONS}
                  value={form.kinds}
                  onChange={(kinds) => set({ kinds })}
                />
              </Field>
              <Toggle checked={form.enabled} onChange={(enabled) => set({ enabled })} label="Enabled" />
            </CardBody>
            <CardFoot className="flex items-center gap-2">
              <Button
                variant="primary"
                onClick={saveRule}
                disabled={saving || !form.to.trim()}
              >
                {saving ? 'Saving…' : 'Save rule'}
              </Button>
              {selectedId !== '_new' && (
                <Button variant="danger" className="ml-auto" onClick={removeRule} disabled={saving}>
                  Remove
                </Button>
              )}
            </CardFoot>
          </Card>
        )}
      </div>
    </div>
  )
}
