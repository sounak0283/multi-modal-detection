import { Dot } from './ui'

/* Application shell: sidebar navigation, page header, and a persistent status rail. */

const NAV = [
  { id: 'live', label: 'Live view', icon: PlayIcon },
  { id: 'boundaries', label: 'Boundaries', icon: ShapeIcon },
  { id: 'camera', label: 'Camera', icon: CameraIcon },
  { id: 'history', label: 'Alert history', icon: ListIcon },
]

export default function Shell({ view, onNavigate, health, title, subtitle, actions, children }) {
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
          {NAV.map(({ id, label, icon: Icon }) => (
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

        <div className="hidden p-2.5 md:block">
          <StatusRail health={health} />
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

/* Camera and storage are separate indicators on purpose. They fail independently and
 * the response differs: a dead camera means the site is unwatched, a dead database
 * means alerts are firing but not being kept. One combined light would hide that. */
function StatusRail({ health }) {
  const feed = health?.feed_state
  const db = health?.database ?? {}

  const feedTone = feed === 'live' ? 'ok' : feed === 'lost' ? 'alarm' : null
  const dbTone = db.connected ? 'ok' : db.configured ? 'alarm' : null

  return (
    <div className="rounded-lg border border-ink-700 bg-ink-800 px-3 py-2.5">
      <Row tone={feedTone} label="Camera" value={feed || 'unknown'} />
      <Row
        tone={dbTone}
        label="Recording"
        value={db.connected ? 'recording' : db.configured ? 'db down' : 'not recording'}
      />
      <Row label="Boundaries" value={health?.zones ?? 0} />
      <p className="mt-2 border-t border-ink-700 pt-2 text-[11px] tabular-nums text-ink-400">
        {health?.inference_ms != null
          ? `${health.render_fps ?? 0} fps · ${health.inference_ms} ms · ${health.people_tracked ?? 0} tracked`
          : 'pipeline not running'}
      </p>
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
