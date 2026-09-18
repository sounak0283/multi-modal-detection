import { useState } from 'react'
import { Card, CardBody, CardHead, Field, Input, Button } from '../components/ui'
import { api } from '../api'

/* Shown instead of the dashboard whenever there is no valid session (Expansion Plan
 * Phase B). Errors render inline rather than as a Toast - there is no ToastProvider
 * wrapping this page's parent decision point (App.jsx renders Login OR Dashboard, never
 * both), and a login failure is exactly the kind of message a user should have to
 * dismiss by trying again, not watch fade after a few seconds. */
export default function Login({ onSuccess }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const user = await api.login(email, password)
      onSuccess(user)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center bg-ink-950 px-4 py-16">
      <Card className="w-full max-w-[360px]">
        <CardHead
          title="Sign in"
          aside={
            <span className="flex items-center gap-2">
              <span className="h-[22px] w-[22px] shrink-0 rounded-lg bg-gradient-to-br from-brand-500 to-[#7c5cff]" />
              <strong className="text-[13.5px] tracking-tight">Perimeter</strong>
            </span>
          }
        />
        <CardBody>
          <form className="space-y-3.5" onSubmit={submit}>
            <Field label="Email">
              <Input
                type="email"
                autoComplete="username"
                autoFocus
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </Field>
            <Field label="Password">
              <Input
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </Field>
            {error && (
              <p role="alert" className="text-[12.5px] text-alarm-400">
                {error}
              </p>
            )}
            <Button type="submit" variant="primary" className="w-full justify-center" disabled={busy}>
              {busy ? 'Signing in…' : 'Sign in'}
            </Button>
          </form>
        </CardBody>
      </Card>
    </div>
  )
}
