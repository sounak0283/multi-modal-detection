import { useState } from 'react'
import { Field, Input, Button } from '../components/ui'
import { api } from '../api'

/* Shown instead of the dashboard whenever there is no valid session (Expansion Plan
 * Phase B). Errors render inline rather than as a Toast - there is no ToastProvider
 * wrapping this page's parent decision point (App.jsx renders Login OR Dashboard, never
 * both), and a login failure is exactly the kind of message a user should have to
 * dismiss by trying again, not watch fade after a few seconds.
 *
 * Layout follows the Stitch sign-in screen: a brand panel on the left (illustration only,
 * no live data) and the form card on the right. */
export default function Login({ onSuccess }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
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
    <div className="grid min-h-full bg-ink-950 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
      <BrandPanel />

      <div className="flex min-h-full flex-col px-6 py-8 sm:px-12 lg:border-l lg:border-ink-700">
        <div className="flex items-center gap-2 lg:hidden">
          <Logo />
          <strong className="text-[15px] font-semibold tracking-tight">PERIMETER</strong>
        </div>

        <div className="flex flex-1 items-center justify-center py-10">
          <div className="w-full max-w-[400px] rounded-xl border border-ink-700 bg-ink-900 p-7 shadow-[var(--level2-shadow)]">
            <p className="font-mono text-[11px] font-medium uppercase tracking-[0.1em] text-brand-400">
              Sign in
            </p>
            <h1 className="mt-2 text-[22px] font-semibold tracking-tight">Sign in to Perimeter</h1>
            <p className="mb-6 mt-1 text-[12.5px] text-ink-400">
              Enter your credentials to access the safety console.
            </p>

            <form className="space-y-4" onSubmit={submit}>
              <Field label="Email">
                <Input
                  type="email"
                  autoComplete="username"
                  autoFocus
                  required
                  placeholder="name@company"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="h-10"
                />
              </Field>
              <Field label="Password">
                <div className="relative">
                  <Input
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="h-10 pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((on) => !on)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                    className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-ink-400 transition-colors hover:text-ink-100"
                  >
                    <EyeIcon off={showPassword} />
                  </button>
                </div>
              </Field>
              {error && (
                <p role="alert" className="text-[12.5px] text-alarm-400">
                  {error}
                </p>
              )}
              <Button
                type="submit"
                variant="primary"
                className="h-10 w-full justify-center text-[14px]"
                disabled={busy}
              >
                {busy ? 'Signing in…' : 'Sign in →'}
              </Button>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}

function Logo() {
  return (
    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-brand-500/40 bg-brand-900 text-brand-400">
      <svg
        viewBox="0 0 16 16"
        width="19"
        height="19"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      >
        <path d="M8 1.8 13.5 4v4c0 3.2-2.3 5.2-5.5 6.2C4.8 13.2 2.5 11.2 2.5 8V4z" />
        <circle cx="8" cy="7.6" r="1.6" />
      </svg>
    </span>
  )
}

function EyeIcon({ off }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8Z" />
      <circle cx="8" cy="8" r="2" />
      {off && <path d="M2.5 13.5 13.5 2.5" />}
    </svg>
  )
}

/* Illustration only: a boundary polygon and a tripwire drawn the way the product draws
 * them. No cameras, counts or readings - nothing here claims to be live. */
function BrandPanel() {
  return (
    <div className="relative hidden overflow-hidden lg:block">
      <div
        aria-hidden
        className="absolute inset-0 opacity-40"
        style={{
          backgroundImage:
            'linear-gradient(var(--color-ink-700) 1px, transparent 1px), linear-gradient(90deg, var(--color-ink-700) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
          maskImage: 'radial-gradient(ellipse at 45% 55%, #000 10%, transparent 72%)',
          WebkitMaskImage: 'radial-gradient(ellipse at 45% 55%, #000 10%, transparent 72%)',
        }}
      />
      <div className="relative flex h-full flex-col p-10">
        <div className="flex items-center gap-3">
          <Logo />
          <strong className="text-[18px] font-semibold tracking-tight">PERIMETER</strong>
        </div>
        <h2 className="mt-8 text-[34px] font-semibold leading-tight tracking-tight">
          Every boundary, watched.
        </h2>
        <p className="mt-2 font-mono text-[12px] uppercase tracking-[0.08em] text-ink-400">
          Fire · smoke · boundary · PPE · crowd
        </p>

        <svg
          viewBox="0 0 520 380"
          className="mx-auto my-auto w-full max-w-[560px]"
          fill="none"
          aria-hidden
        >
          <polygon
            points="70,90 250,50 380,150 330,300 120,330 50,200"
            fill="rgba(16,185,129,0.07)"
            stroke="var(--color-zone-area)"
            strokeWidth="1.5"
            strokeDasharray="6 5"
          />
          {[
            [70, 90],
            [250, 50],
            [380, 150],
            [330, 300],
            [120, 330],
            [50, 200],
          ].map(([x, y]) => (
            <circle key={`${x}-${y}`} cx={x} cy={y} r="4" fill="var(--color-zone-area)" />
          ))}
          <line
            x1="400"
            y1="60"
            x2="470"
            y2="320"
            stroke="var(--color-info-400)"
            strokeWidth="1.5"
            strokeDasharray="6 5"
          />
          <circle cx="400" cy="60" r="4" fill="var(--color-info-400)" />
          <circle cx="470" cy="320" r="4" fill="var(--color-info-400)" />
          <g fontFamily="var(--font-mono)" fontSize="11" letterSpacing="0.8">
            <rect x="82" y="68" width="92" height="22" rx="3" fill="var(--color-ink-900)" stroke="var(--color-zone-area)" strokeOpacity="0.5" />
            <text x="92" y="83" fill="var(--color-zone-area)">BOUNDARY</text>
            <rect x="412" y="150" width="86" height="22" rx="3" fill="var(--color-ink-900)" stroke="var(--color-info-400)" strokeOpacity="0.5" />
            <text x="422" y="165" fill="var(--color-info-400)">TRIPWIRE</text>
          </g>
        </svg>
      </div>
    </div>
  )
}
