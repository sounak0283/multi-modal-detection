import { useState } from 'react'
import { Card, CardBody, Field, Input, Button } from '../components/ui'
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
    <div className="relative flex min-h-full items-center justify-center overflow-hidden bg-ink-950 px-4 py-16">
      {/* Faint grid + indigo glow: atmosphere only, no data. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.35]"
        style={{
          backgroundImage:
            'linear-gradient(var(--color-ink-700) 1px, transparent 1px), linear-gradient(90deg, var(--color-ink-700) 1px, transparent 1px)',
          backgroundSize: '40px 40px',
          maskImage: 'radial-gradient(ellipse at center, #000 20%, transparent 70%)',
          WebkitMaskImage: 'radial-gradient(ellipse at center, #000 20%, transparent 70%)',
        }}
      />
      <div className="relative w-full max-w-[380px]">
        <div className="mb-6 flex flex-col items-center text-center">
          <span className="mb-3 grid h-10 w-10 place-items-center rounded-lg bg-brand-500 text-white">
            <svg viewBox="0 0 16 16" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round">
              <path d="M8 1.8 13.5 4v4c0 3.2-2.3 5.2-5.5 6.2C4.8 13.2 2.5 11.2 2.5 8V4z" />
            </svg>
          </span>
          <strong className="text-[15px] font-semibold tracking-[0.08em]">PERIMETER</strong>
          <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.1em] text-ink-400">
            Every boundary, watched.
          </p>
        </div>
        <Card className="shadow-[var(--level2-shadow)]">
          <CardBody className="p-6">
            <h1 className="text-[18px] font-semibold tracking-tight">Sign in to Perimeter</h1>
            <p className="mb-5 mt-1 text-[12.5px] text-ink-400">
              Enter your credentials to access the safety console.
            </p>
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
    </div>
  )
}
