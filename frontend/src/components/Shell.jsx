import { useState } from 'react'
import { Dot, Icon, IconTile, TONES, cx } from './ui'
import { getTheme, setTheme } from '../lib/theme'

/* Application shell: grouped sidebar navigation (240px, collapsible to 56px), top bar,
 * and a persistent system-health + signed-in-user footer. */

// Icons and section colours are the ones used by the Stitch light design.
const GROUPS = [
  {
    label: 'Monitor',
    items: [
      { id: 'live', label: 'Live view', icon: 'grid_view', tone: 'indigo' },
      { id: 'boundaries', label: 'Boundaries', icon: 'polyline', tone: 'purple' },
      { id: 'cameras', label: 'Cameras', icon: 'videocam', tone: 'sky' },
    ],
  },
  {
    label: 'Review',
    items: [
      { id: 'history', label: 'Alert history', icon: 'notifications_active', tone: 'rose' },
      { id: 'evidence', label: 'Evidence', icon: 'folder_special', tone: 'amber' },
      { id: 'recognition', label: 'Recognition log', icon: 'badge', tone: 'teal' },
    ],
  },
  {
    label: 'Configure',
    items: [
      { id: 'videotest', label: 'Video test', icon: 'movie', tone: 'pink', feature: 'video_test', admin: true },
      { id: 'alerts', label: 'Alert rules', icon: 'rule', tone: 'violet', admin: true },
      { id: 'persons', label: 'People', icon: 'group', tone: 'emerald', admin: true },
      { id: 'users', label: 'Accounts', icon: 'manage_accounts', tone: 'blue', admin: true },
      { id: 'settings', label: 'Settings', icon: 'settings', tone: 'slate', admin: true },
    ],
  },
]

const VIEW_META = Object.fromEntries(GROUPS.flatMap((g) => g.items).map((i) => [i.id, i]))

const COLLAPSE_KEY = 'perimeter.sidebar.collapsed'
const readCollapsed = () => {
  try {
    return localStorage.getItem(COLLAPSE_KEY) === '1'
  } catch {
    return false
  }
}

export default function Shell({
  view,
  onNavigate,
  health,
  title,
  subtitle,
  actions,
  features,
  user,
  onLogout,
  children,
}) {
  const [collapsed, setCollapsed] = useState(readCollapsed)
  const [theme, setThemeState] = useState(getTheme)
  const toggleCollapsed = () =>
    setCollapsed((current) => {
      try {
        localStorage.setItem(COLLAPSE_KEY, current ? '0' : '1')
      } catch {
        /* storage blocked - applies for this page view only */
      }
      return !current
    })

  const isAdmin = user?.role === 'admin'
  const groups = GROUPS.map((group) => ({
    ...group,
    items: group.items.filter(
      (item) => (!item.admin || isAdmin) && (!item.feature || features?.[item.feature] !== false),
    ),
  })).filter((group) => group.items.length > 0)

  return (
    <div
      className={cx(
        'grid min-h-full grid-cols-1',
        collapsed ? 'md:grid-cols-[56px_minmax(0,1fr)]' : 'md:grid-cols-[240px_minmax(0,1fr)]',
      )}
    >
      <aside className="flex flex-col border-r border-ink-700 bg-ink-900 md:sticky md:top-0 md:h-screen">
        <div
          className={cx(
            'flex items-center gap-2.5 border-b border-ink-700 py-3.5',
            collapsed ? 'px-4 md:justify-center md:px-0' : 'px-4',
          )}
        >
          <Logo />
          <span className={cx('min-w-0 leading-tight', collapsed && 'md:hidden')}>
            <strong className="block text-[14px] font-semibold tracking-[0.04em]">PERIMETER</strong>
            <small className="block font-mono text-[10px] uppercase tracking-[0.08em] text-ink-400">
              Fire, smoke &amp; boundary
            </small>
          </span>
        </div>

        <nav className="flex flex-1 flex-row gap-1 overflow-x-auto p-2 md:flex-col md:gap-0 md:overflow-y-auto md:overflow-x-hidden">
          {groups.map((group) => (
            <div key={group.label} className="flex flex-row gap-0.5 md:mb-3 md:flex-col">
              <p
                className={cx(
                  'hidden px-3 pb-1 pt-2 font-mono text-[10px] font-medium uppercase tracking-[0.1em] text-ink-500 md:block',
                  collapsed && 'md:invisible md:h-2 md:overflow-hidden md:p-0',
                )}
              >
                {group.label}
              </p>
              {group.items.map(({ id, label, icon, tone }) => {
                const on = view === id
                return (
                  <button
                    key={id}
                    type="button"
                    data-view={id}
                    title={collapsed ? label : undefined}
                    onClick={() => onNavigate(id)}
                    aria-current={on ? 'page' : undefined}
                    className={cx(
                      'flex h-9 shrink-0 items-center gap-2.5 rounded-lg border px-2 text-left text-[13px] transition-colors',
                      collapsed && 'md:justify-center md:px-0',
                      on
                        ? cx('border-brand-500/30 bg-brand-900 font-medium text-brand-100 light:shadow-xs', TONES[tone].row)
                        : 'border-transparent text-ink-300 hover:bg-ink-800 hover:text-ink-100 light:hover:bg-slate-50',
                    )}
                  >
                    <IconTile icon={icon} tone={tone} size="sm" active={on} className="bg-transparent" />
                    <span className={cx(collapsed && 'md:hidden')}>{label}</span>
                  </button>
                )
              })}
            </div>
          ))}
        </nav>

        <div className="hidden space-y-2 border-t border-ink-700 p-2 md:block">
          {!collapsed && <StatusRail health={health} />}
          {!collapsed && user && <AccountRow user={user} onLogout={onLogout} />}
          {collapsed && user && (
            <button
              type="button"
              onClick={onLogout}
              title={`Log out (${user.email})`}
              aria-label="Log out"
              className="flex h-8 w-full items-center justify-center rounded-lg text-[13px] text-ink-300 hover:bg-ink-800 hover:text-ink-100"
            >
              ⎋
            </button>
          )}
          <button
            type="button"
            onClick={toggleCollapsed}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            className="flex h-7 w-full items-center justify-center rounded-lg text-[11px] text-ink-400 transition-colors hover:bg-ink-800 hover:text-ink-100"
          >
            {collapsed ? '»' : '« Collapse'}
          </button>
        </div>
      </aside>

      <main className="flex min-w-0 flex-col">
        <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-ink-700 bg-ink-900 px-6 py-3 light:shadow-[0_1px_2px_rgba(16,24,40,0.04)]">
          {VIEW_META[view] && (
            <span className="hidden h-10 w-10 shrink-0 place-items-center rounded-xl text-white shadow-sm light:grid light:bg-gradient-to-br light:from-indigo-500 light:to-violet-600">
              <Icon name={VIEW_META[view].icon} size={22} />
            </span>
          )}
          <div className="min-w-0">
            <h1 className="truncate text-[17px] font-semibold tracking-tight light:text-[20px] light:font-bold">{title}</h1>
            <p className="mt-0.5 text-[12px] text-ink-400">{subtitle}</p>
          </div>
          <div className="ml-auto flex shrink-0 items-center gap-2">
            {actions}
            <button
              type="button"
              onClick={() => setThemeState(setTheme(theme === 'light' ? 'dark' : 'light'))}
              aria-label={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
              title={theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'}
              className="grid h-8 w-8 place-items-center rounded-lg border border-ink-700 bg-ink-900 text-ink-300 transition-colors hover:border-ink-hover hover:text-ink-100"
            >
              <Icon name={theme === 'light' ? 'dark_mode' : 'light_mode'} size={18} />
            </button>
          </div>
        </header>
        <div className="min-w-0 flex-1 p-6 pb-20">{children}</div>
      </main>
    </div>
  )
}

function Logo() {
  return (
    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-brand-500/30 bg-brand-900 text-brand-400 light:border-transparent light:bg-[image:var(--aura)] light:text-white light:shadow-sm">
      <Icon name="security" size={19} fill />
    </span>
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
    <div className="rounded-xl border border-ink-700 bg-ink-850 px-3 py-2 light:bg-slate-50/80">
      <p className="pb-1 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-500">Status</p>
      <Row
        tone={camerasTone}
        live={camerasTone === 'ok'}
        label="Cameras"
        value={`${running}/${enabled} running`}
      />
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
    <div className="flex items-center gap-2 rounded-lg border border-ink-700 bg-ink-850 px-3 py-2">
      <div className="min-w-0 flex-1">
        <p className="truncate text-[12px] text-ink-100">{user.email}</p>
        <p className="font-mono text-[10px] uppercase tracking-[0.08em] text-ink-400">
          {user.role}
        </p>
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

function Row({ tone, live, label, value }) {
  return (
    <div className="flex items-center gap-2 py-[3px] text-[12px]">
      <Dot tone={tone} live={live} />
      <span className="flex-1 text-ink-400">{label}</span>
      <span className="font-mono text-[11px] tabular-nums text-ink-100">{value}</span>
    </div>
  )
}
