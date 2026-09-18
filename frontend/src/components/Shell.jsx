import { Dot } from './ui'

/* Application shell: sidebar navigation, page header, and a persistent status rail. */

const NAV = [
  { id: 'live', label: 'Live view', icon: PlayIcon },
  { id: 'boundaries', label: 'Boundaries', icon: ShapeIcon },
  { id: 'cameras', label: 'Cameras', icon: CameraIcon },
  { id: 'history', label: 'Alert history', icon: ListIcon },
  { id: 'recognition', label: 'Recognition log', icon: ScanIcon },
]

const ADMIN_NAV = [
  { id: 'alerts', label: 'Alerts', icon: BellIcon },
  { id: 'persons', label: 'People', icon: IdIcon },
  { id: 'users', label: 'Accounts', icon: UserIcon },
]

export default function Shell({
  view,
  onNavigate,
  health,
  title,
  subtitle,
  actions,
  user,
  onLogout,
  children,
}) {
  const nav = user?.role === 'admin' ? [...NAV, ...ADMIN_NAV] : NAV
  return (
    <div className="grid min-h-full grid-cols-1 md:grid-cols-[232px_minmax(0,1fr)]">
      <aside className="flex flex-col border-r border-ink-700 bg-ink-900 md:sticky md:top-0 md:h-screen">
        <div className="flex items-center gap-2.5 border-b border-ink-800 px-4 py-4">
          <span className="h-[26px] w-[26px] shrink-0 rounded-lg bg-gradient-to-br from-brand-500 to-[#7c5cff]" />
          <span className="min-w-0 leading-tight">
            <strong className="block text-[14px] tracking-tight">Perimeter</strong>
            <small className="block text-[10.5px] text-ink-400">Fire, smoke &amp; boundary</small>
          </span>
        </div>

        <nav className="flex flex-1 flex-row gap-0.5 overflow-x-auto p-2.5 md:flex-col md:overflow-visible">
          {nav.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              data-view={id}
              onClick={() => onNavigate(id)}
              aria-current={view === id ? 'page' : undefined}
              className={`flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-left text-[13.5px] transition-colors ${
                view === id
                  ? 'bg-brand-900 font-medium text-[#cfe2ff]'
                  : 'text-ink-300 hover:bg-ink-800 hover:text-ink-100'
              }`}
            >
              <Icon />
              {label}
            </button>
          ))}
        </nav>

        <div className="hidden space-y-2 p-2.5 md:block">
          <StatusRail health={health} />
          {user && <AccountRow user={user} onLogout={onLogout} />}
        </div>
      </aside>

      <main className="flex min-w-0 flex-col">
        <header className="flex items-center gap-4 border-b border-ink-700 bg-ink-900 px-6 py-4">
          <div className="min-w-0">
            <h1 className="truncate text-[17px] font-semibold tracking-tight">{title}</h1>
            <p className="mt-0.5 text-[12.5px] text-ink-400">{subtitle}</p>
          </div>
          <div className="ml-auto flex shrink-0 gap-2">{actions}</div>
        </header>
        <div className="min-w-0 flex-1 p-6">{children}</div>
      </main>
    </div>
  )
}

/* Cameras and storage are separate indicators on purpose. They fail independently and
 * the response differs: a camera down means one site is unwatched, a dead database
 * means every alert is firing but not being kept. One combined light would hide that.
 * Per-camera detail (fps, people tracked, feed state) lives on the Live view /
 * Cameras page instead of here - the sidebar is a system-wide summary across any
 * number of cameras, not one camera's dashboard. */
function StatusRail({ health }) {
  const cameras = health?.cameras ?? []
  const db = health?.database ?? {}

  const running = cameras.filter((c) => c.running).length
  const enabled = cameras.filter((c) => c.enabled).length
  const camerasTone = enabled === 0 ? null : running === enabled ? 'ok' : 'alarm'
  const dbTone = db.connected ? 'ok' : db.configured ? 'alarm' : null

  return (
    <div className="rounded-lg border border-ink-700 bg-ink-800 px-3 py-2.5">
      <Row tone={camerasTone} label="Cameras" value={`${running}/${enabled} running`} />
      <Row
        tone={dbTone}
        label="Recording"
        value={db.connected ? 'recording' : db.configured ? 'db down' : 'not recording'}
      />
    </div>
  )
}

function AccountRow({ user, onLogout }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-ink-700 bg-ink-800 px-3 py-2.5">
      <div className="min-w-0 flex-1">
        <p className="truncate text-[12px] text-ink-100">{user.email}</p>
        <p className="text-[10.5px] capitalize text-ink-400">{user.role}</p>
      </div>
      <button
        type="button"
        onClick={onLogout}
        className="shrink-0 rounded-md px-2 py-1 text-[11px] text-ink-300 transition-colors hover:bg-ink-700 hover:text-ink-100"
      >
        Log out
      </button>
    </div>
  )
}

function Row({ tone, label, value }) {
  return (
    <div className="flex items-center gap-2 py-[3px] text-[12px]">
      <Dot tone={tone} />
      <span className="flex-1 text-ink-400">{label}</span>
      <span className="tabular-nums text-ink-100">{value}</span>
    </div>
  )
}

/* Inline SVG rather than an icon package: six glyphs is not worth a dependency, a
 * licence row in NOTICE.md, or the bundle weight. */
const iconProps = {
  width: 15,
  height: 15,
  viewBox: '0 0 16 16',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.5,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  className: 'shrink-0 opacity-80',
}

function PlayIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4.5 3.2v9.6l7.5-4.8z" />
    </svg>
  )
}

function ShapeIcon() {
  return (
    <svg {...iconProps}>
      <path d="M2.5 5.5 8 2l5.5 3.5v5L8 14 2.5 10.5z" />
    </svg>
  )
}

function CameraIcon() {
  return (
    <svg {...iconProps}>
      <rect x="1.8" y="4" width="12.4" height="8" rx="1.6" />
      <circle cx="8" cy="8" r="2.2" />
    </svg>
  )
}

function ListIcon() {
  return (
    <svg {...iconProps}>
      <path d="M2.5 4h11M2.5 8h11M2.5 12h7" />
    </svg>
  )
}

function UserIcon() {
  return (
    <svg {...iconProps}>
      <circle cx="8" cy="5.3" r="2.6" />
      <path d="M2.8 13.4c0-2.7 2.3-4.4 5.2-4.4s5.2 1.7 5.2 4.4" />
    </svg>
  )
}

function IdIcon() {
  return (
    <svg {...iconProps}>
      <rect x="1.8" y="3" width="12.4" height="10" rx="1.4" />
      <circle cx="5.6" cy="7.4" r="1.6" />
      <path d="M3.4 11.2c0-1.4 1-2.2 2.2-2.2s2.2.8 2.2 2.2" />
      <path d="M9.6 6.4h3M9.6 8.8h3" />
    </svg>
  )
}

function ScanIcon() {
  return (
    <svg {...iconProps}>
      <path d="M2 5V3a1 1 0 0 1 1-1h2M14 2h2a1 1 0 0 1 1 1v2M14 14h2a1 1 0 0 1-1 1h-2M2 11v2a1 1 0 0 0 1 1h2" />
      <circle cx="8" cy="8" r="2.4" />
    </svg>
  )
}

function BellIcon() {
  return (
    <svg {...iconProps}>
      <path d="M4 6.5a4 4 0 0 1 8 0c0 3.5 1.2 4.5 1.2 4.5H2.8S4 10 4 6.5Z" />
      <path d="M6.5 13.2a1.5 1.5 0 0 0 3 0" />
    </svg>
  )
}
