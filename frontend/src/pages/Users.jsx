import { useCallback, useEffect, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  CardBody,
  CardFoot,
  CardHead,
  EmptyState,
  Field,
  Input,
  Select,
  Toggle,
} from '../components/ui'
import { api } from '../api'
import { useToast } from '../components/Toast'

const BLANK_FORM = { email: '', password: '', role: 'operator', active: true }

function formFromUser(user) {
  return { email: user.email, password: '', role: user.role, active: user.active }
}

/* Admin-only account management (Expansion Plan Phase B). Mirrors Cameras.jsx's
 * list + detail-panel shape. Email is immutable once created - same rule the backend
 * enforces (a camera's id is immutable via its update route too) - so existing rows
 * show it read-only. The password field is never pre-filled: on an existing account it
 * is an opt-in "set a new password", the same write-only pattern the Cameras page uses
 * for the RTSP URL. */
export default function Users({ currentUser }) {
  const [users, setUsers] = useState([])
  const [selectedId, setSelectedId] = useState(null) // null | '_new' | an existing email
  const [form, setForm] = useState(null)
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const toast = useToast()

  const load = useCallback(async () => {
    try {
      const body = await api.listUsers()
      setUsers(body.users ?? [])
    } catch (err) {
      toast(err.message, 'error')
    }
  }, [toast])

  useEffect(() => {
    load()
  }, [load])

  const selectExisting = (user) => {
    setSelectedId(user.id)
    setForm(formFromUser(user))
  }

  const startNew = () => {
    setSelectedId('_new')
    setForm({ ...BLANK_FORM })
  }

  if (!form && users.length > 0 && selectedId === null) {
    selectExisting(users[0])
  }

  const set = (patch) => setForm((f) => ({ ...f, ...patch }))

  const save = async () => {
    setSaving(true)
    try {
      if (selectedId === '_new') {
        await api.createUser({ email: form.email, password: form.password, role: form.role })
        toast(`Account created for ${form.email}.`, 'ok')
      } else {
        const patch = { role: form.role, active: form.active }
        if (form.password.trim()) patch.password = form.password.trim()
        await api.updateUser(selectedId, patch)
        toast('Account updated.', 'ok')
      }
      await load()
      setSelectedId(null)
      setForm(null)
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setSaving(false)
    }
  }

  const remove = async () => {
    if (selectedId === '_new' || !selectedId) return
    if (!window.confirm(`Remove the account for "${form.email}"? This cannot be undone.`)) return
    setDeleting(true)
    try {
      await api.deleteUser(selectedId)
      toast('Account removed.', 'ok')
      setSelectedId(null)
      setForm(null)
      await load()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <div className="grid grid-cols-1 items-start gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
      <Card>
        <CardHead
          title="Accounts"
          aside={
            <span className="rounded-full bg-ink-700 px-2 py-0.5 text-[11px] text-ink-200">
              {users.length}
            </span>
          }
        />
        {users.length === 0 ? (
          <EmptyState>No accounts yet.</EmptyState>
        ) : (
          <ul className="p-1.5">
            {users.map((user) => (
              <li key={user.id}>
                <button
                  type="button"
                  onClick={() => selectExisting(user)}
                  className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors ${
                    selectedId === user.id
                      ? 'border-brand-600 bg-brand-900'
                      : 'border-transparent hover:bg-ink-800'
                  }`}
                >
                  <span
                    className={`h-2 w-2 shrink-0 rounded-full ${user.active ? 'bg-ok-400' : 'bg-ink-600'}`}
                  />
                  <span className="min-w-0 flex-1 truncate text-[13px]">{user.email}</span>
                  <Badge tone={user.role === 'admin' ? 'info' : 'neutral'}>{user.role}</Badge>
                </button>
              </li>
            ))}
          </ul>
        )}
        <CardFoot>
          <Button className="w-full" onClick={startNew}>
            + Add account
          </Button>
        </CardFoot>
      </Card>

      {!form ? (
        <Card>
          <EmptyState>Add an account to get started.</EmptyState>
        </Card>
      ) : (
        <Card>
          <CardHead title={selectedId === '_new' ? 'New account' : form.email} />
          <CardBody className="space-y-3.5">
            <Field label="Email" hint={selectedId === '_new' ? undefined : 'Cannot be changed once created.'}>
              <Input
                type="email"
                value={form.email}
                onChange={(e) => set({ email: e.target.value })}
                disabled={selectedId !== '_new'}
                placeholder="name@example.com"
              />
            </Field>

            <Field
              label={selectedId === '_new' ? 'Password' : 'Set new password'}
              hint={selectedId === '_new' ? '8–72 characters.' : 'Leave blank to keep the current password.'}
            >
              <Input
                type="password"
                autoComplete="new-password"
                value={form.password}
                onChange={(e) => set({ password: e.target.value })}
                placeholder={selectedId === '_new' ? '' : '••••••••'}
              />
            </Field>

            <Field label="Role" hint="Operators can view everything but cannot change configuration.">
              <Select value={form.role} onChange={(e) => set({ role: e.target.value })}>
                <option value="operator">Operator</option>
                <option value="admin">Admin</option>
              </Select>
            </Field>

            {selectedId !== '_new' && (
              <Toggle
                checked={form.active}
                onChange={(active) => set({ active })}
                label={form.active ? 'Active' : 'Deactivated — cannot log in'}
              />
            )}

            {selectedId === form.email && selectedId === currentUser?.id && (
              <p className="text-[11.5px] leading-snug text-ink-400">
                This is the account you are signed in as.
              </p>
            )}
          </CardBody>
          <CardFoot className="flex items-center gap-2">
            <Button variant="primary" onClick={save} disabled={saving || !form.email || (selectedId === '_new' && !form.password)}>
              {saving ? 'Saving…' : 'Save'}
            </Button>
            {selectedId !== '_new' && (
              <Button variant="danger" className="ml-auto" onClick={remove} disabled={deleting}>
                {deleting ? 'Removing…' : 'Remove account'}
              </Button>
            )}
          </CardFoot>
        </Card>
      )}
    </div>
  )
}
