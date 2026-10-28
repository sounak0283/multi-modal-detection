/* Design primitives.
 *
 * Every surface in the app is a Card, every action is a Button, every state is a Badge.
 * Keeping them here rather than repeating utility strings is what stops a dashboard
 * drifting into six slightly different greys and four button heights.
 *
 * Sizes follow design/DESIGN.md: buttons and inputs 32px, table rows 36px, pills and
 * telemetry chips 20px, cards 8px radius.
 */
import { useEffect } from 'react'
import { severityOf, eventTone } from '../lib/status'

export const cx = (...parts) => parts.filter(Boolean).join(' ')

export function Card({ children, className = '' }) {
  return (
    <section
      className={cx(
        'perimeter-card rounded-xl border border-ink-700 bg-ink-900 overflow-hidden',
        className,
      )}
    >
      {children}
    </section>
  )
}

/** Card header. `title` renders as the uppercase section label. */
export function CardHead({ title, aside, children }) {
  return (
    <header className="flex min-h-11 items-center gap-3 border-b border-ink-700 px-4 py-2.5">
      <h2 className="flex-1 font-mono text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-400">
        {title}
      </h2>
      {aside}
      {children}
    </header>
  )
}

/** Alias: a Card with an uppercase label, as named in the design inventory. */
export const SectionCard = Card

export function CardBody({ children, className = '' }) {
  return <div className={cx('p-4', className)}>{children}</div>
}

export function CardFoot({ children, className = '' }) {
  return (
    <footer className={cx('border-t border-ink-700 px-4 py-3', className)}>{children}</footer>
  )
}

const BUTTON_VARIANTS = {
  default:
    'bg-ink-900 border-ink-700 text-ink-300 hover:bg-ink-800 hover:border-ink-hover hover:text-ink-100',
  primary:
    'bg-brand-500 border-brand-500 text-on-brand font-semibold hover:bg-brand-600 hover:border-brand-600',
  danger: 'bg-transparent border-alarm-400/50 text-alarm-400 hover:bg-alarm-900',
  ghost: 'bg-transparent border-transparent text-ink-300 hover:bg-ink-800 hover:text-ink-100',
}

export function Button({
  variant = 'default',
  className = '',
  type = 'button',
  children,
  ...rest
}) {
  return (
    <button
      type={type}
      className={cx(
        'inline-flex h-8 items-center justify-center gap-2 rounded-lg border px-3 text-[13px] font-medium',
        'transition-[color,background-color,border-color,transform] active:scale-[0.98]',
        'disabled:cursor-not-allowed disabled:opacity-40 disabled:active:scale-100',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500',
        BUTTON_VARIANTS[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

const BADGE_TONES = {
  neutral: 'border-ink-600 bg-ink-800 text-ink-300',
  ok: 'border-ok-400/30 bg-ok-900 text-ok-400',
  warn: 'border-warn-400/30 bg-warn-900 text-warn-400',
  alarm: 'border-alarm-400/30 bg-alarm-900 text-alarm-400',
  info: 'border-info-400/30 bg-info-900 text-info-400',
  brand: 'border-brand-500/30 bg-brand-900 text-brand-400',
}

/** Status pill: 20px, mono, uppercase, tinted base + low-opacity border. */
export function Badge({ tone = 'neutral', children, className = '' }) {
  return (
    <span
      className={cx(
        'inline-flex h-5 items-center whitespace-nowrap rounded-full border px-2 font-mono text-[11px] font-medium uppercase tracking-[0.04em]',
        BADGE_TONES[tone] ?? BADGE_TONES.neutral,
        className,
      )}
    >
      {children}
    </span>
  )
}

/** Severity chip - colour comes from the shared map in lib/status.js. */
export function SeverityChip({ severity, className = '' }) {
  if (!severity) return <span className="text-ink-500">—</span>
  const { tone, label } = severityOf(severity)
  return (
    <Badge tone={tone} className={className}>
      {label}
    </Badge>
  )
}

/** Event pill (Entry, Exit, Helmet, Fire, ...), coloured from the same shared map. */
export function EventPill({ event, children, className = '' }) {
  return (
    <Badge tone={eventTone(event)} className={className}>
      {children}
    </Badge>
  )
}

/** Status dot. `tone` null renders the inactive/unknown state rather than hiding.
 * `live` adds the expanding ping ring used for live feeds and active detections. */
export function Dot({ tone, live = false, className = '' }) {
  const fill = {
    ok: 'bg-ok-400',
    alarm: 'bg-alarm-400',
    warn: 'bg-warn-400',
    info: 'bg-info-400',
  }[tone]
  return (
    <span className={cx('relative inline-flex h-1.5 w-1.5 shrink-0', className)}>
      {live && fill && (
        <span className={cx('absolute inset-0 animate-live-ping rounded-full', fill)} />
      )}
      <span className={cx('relative h-1.5 w-1.5 rounded-full', fill || 'bg-ink-500')} />
    </span>
  )
}
export const StatusDot = Dot

/** 20px mono label/value pair used on video tiles and headers. */
export function TelemetryChip({ label, children, className = '' }) {
  return (
    <span
      className={cx(
        'inline-flex h-5 items-center gap-1.5 rounded-sm border border-ink-700 bg-ink-800/80 px-1.5 font-mono text-[11px]',
        className,
      )}
    >
      {label && <span className="uppercase tracking-[0.04em] text-ink-400">{label}</span>}
      <span className="text-ink-100">{children}</span>
    </span>
  )
}

export function Field({ label, hint, children, className = '' }) {
  return (
    <label className={cx('block', className)}>
      {label && <span className="mb-1.5 block text-[12px] font-medium text-ink-300">{label}</span>}
      {children}
      {hint && <span className="mt-1 block text-[11.5px] leading-snug text-ink-400">{hint}</span>}
    </label>
  )
}

const CONTROL =
  'h-8 w-full rounded-lg border border-ink-700 bg-ink-950 px-3 text-[13px] text-ink-100 '
  + 'placeholder:text-ink-500 transition-colors hover:border-ink-hover '
  + 'focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500 '
  + 'disabled:cursor-not-allowed disabled:opacity-60'

export function Input({ className = '', ...rest }) {
  return <input className={cx(CONTROL, className)} {...rest} />
}

/** Input with the "locked" look for values fixed after creation (e.g. camera ID). */
export function TextField({ locked = false, className = '', ...rest }) {
  return (
    <div className="relative">
      <input
        readOnly={locked || rest.readOnly}
        className={cx(
          CONTROL,
          locked && 'cursor-not-allowed bg-ink-800 pr-8 font-mono text-ink-300',
          className,
        )}
        {...rest}
      />
      {locked && (
        <svg
          viewBox="0 0 16 16"
          width="13"
          height="13"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-ink-400"
          aria-label="Locked"
        >
          <rect x="3.5" y="7" width="9" height="6.5" rx="1.5" />
          <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" strokeLinecap="round" />
        </svg>
      )}
    </div>
  )
}

export function Select({ className = '', children, ...rest }) {
  return (
    <select className={cx(CONTROL, className)} {...rest}>
      {children}
    </select>
  )
}

/** Multi-select chips. Reads faster than a column of checkboxes at this size. */
export function ChipGroup({ options, value, onChange }) {
  const toggle = (option) =>
    onChange(value.includes(option) ? value.filter((v) => v !== option) : [...value, option])

  return (
    <div className="flex flex-wrap gap-2">
      {options.map(({ value: option, label }) => {
        const on = value.includes(option)
        return (
          <button
            key={option}
            type="button"
            onClick={() => toggle(option)}
            aria-pressed={on}
            className={cx(
              'h-7 rounded-md border px-2.5 text-[12px] transition-colors',
              on
                ? 'border-brand-500 bg-brand-900 text-brand-400'
                : 'border-ink-700 bg-ink-900 text-ink-300 hover:border-ink-hover hover:text-ink-100',
            )}
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}

/** Single-choice segmented switch. `options` is [{ value, label }]. */
export function SegmentedControl({ options, value, onChange, className = '' }) {
  return (
    <div
      role="group"
      className={cx('inline-flex rounded-lg border border-ink-700 bg-ink-950 p-0.5', className)}
    >
      {options.map((option) => {
        const on = option.value === value
        return (
          <button
            key={String(option.value)}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={on}
            className={cx(
              'h-6 rounded-md border px-3 text-[12px] transition-colors',
              on
                ? 'border-ink-600 bg-ink-800 font-medium text-ink-100'
                : 'border-transparent text-ink-400 hover:text-ink-100',
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}

export function Toggle({ checked, onChange, label }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center gap-3 text-left text-[13px] text-ink-100"
    >
      <span
        className={cx(
          'relative h-5 w-9 shrink-0 rounded-full border transition-colors',
          checked ? 'border-brand-500 bg-brand-500' : 'border-ink-600 bg-ink-700',
        )}
      >
        <span
          className={cx(
            'absolute top-0.5 h-3.5 w-3.5 rounded-full bg-white transition-all',
            checked ? 'left-[18px]' : 'left-0.5',
          )}
        />
      </span>
      {label}
    </button>
  )
}

/** KPI tile. `aside` (e.g. a hint or sparkline) renders top-right. */
export function Stat({ value, label, tone, aside }) {
  const tones = {
    alarm: 'text-alarm-400',
    warn: 'text-warn-400',
    ok: 'text-ok-400',
    info: 'text-info-400',
  }
  return (
    <Card className="px-4 py-3">
      <div className="flex items-start justify-between gap-2">
        <div className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-ink-400">
          {label}
        </div>
        {aside}
      </div>
      <div
        className={cx(
          'mt-1 font-mono text-[26px] font-semibold leading-tight tabular-nums',
          tones[tone],
        )}
      >
        {value}
      </div>
    </Card>
  )
}
export const KPI = Stat

export function EmptyState({ children, action }) {
  return (
    <div className="px-4 py-10 text-center">
      <p className="text-[12.5px] text-ink-400">{children}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}

export function Skeleton({ className = '' }) {
  return <div className={cx('animate-pulse rounded-md bg-ink-800', className)} />
}

/* ---- Tables: sticky mono header, 36px rows, hover tint --------------------------- */

/** Scroll container + table. Wide content scrolls inside its own box. */
export function DataTable({ children, className = '', maxHeight }) {
  return (
    <div className={cx('overflow-auto', className)} style={maxHeight ? { maxHeight } : undefined}>
      <table className="w-full border-collapse text-[13px]">{children}</table>
    </div>
  )
}

export function Th({ children, align = 'left', className = '' }) {
  return (
    <th
      className={cx(
        'sticky top-0 z-[1] h-9 whitespace-nowrap bg-ink-800 px-4 font-mono text-[11px] font-medium uppercase tracking-[0.08em] text-ink-400',
        align === 'right' ? 'text-right' : 'text-left',
        className,
      )}
    >
      {children}
    </th>
  )
}

export function Tr({ children, onClick, className = '', ...rest }) {
  return (
    <tr
      onClick={onClick}
      className={cx(
        'h-9 border-t border-ink-800 transition-colors hover:bg-ink-800/70',
        onClick && 'cursor-pointer',
        className,
      )}
      {...rest}
    >
      {children}
    </tr>
  )
}

/** `mono` for telemetry cells (timestamps, IDs, numbers); numbers right-align. */
export function Td({ children, mono = false, align = 'left', className = '', ...rest }) {
  return (
    <td
      className={cx(
        'whitespace-nowrap px-4 py-1.5',
        mono && 'font-mono tabular-nums text-ink-300',
        align === 'right' && 'text-right',
        className,
      )}
      {...rest}
    >
      {children}
    </td>
  )
}

/* ---- Overlays -------------------------------------------------------------------- */

function useEscape(onClose) {
  useEffect(() => {
    const onKey = (event) => event.key === 'Escape' && onClose?.()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
}

function CloseButton({ onClose }) {
  return (
    <button
      type="button"
      onClick={onClose}
      aria-label="Close"
      className="rounded-md px-2 py-1 text-ink-300 transition-colors hover:bg-ink-700 hover:text-ink-100"
    >
      ✕
    </button>
  )
}

/** Centred dialog. Click the scrim or press Esc to close. */
export function Modal({ title, onClose, children, footer, wide = false }) {
  useEscape(onClose)
  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-4"
      onMouseDown={(event) => event.target === event.currentTarget && onClose?.()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cx(
          'flex max-h-full w-full flex-col overflow-hidden rounded-xl border border-ink-600 bg-ink-800 shadow-[var(--level2-shadow)]',
          wide ? 'max-w-4xl' : 'max-w-lg',
        )}
      >
        <header className="flex items-center gap-3 border-b border-ink-600 px-4 py-3">
          <h2 className="flex-1 text-[14px] font-semibold text-ink-100">{title}</h2>
          <CloseButton onClose={onClose} />
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
        {footer && <footer className="border-t border-ink-600 px-4 py-3">{footer}</footer>}
      </div>
    </div>
  )
}

/** Right-side inspector, 360px by default. */
export function Drawer({ title, onClose, children, footer, width = 360 }) {
  useEscape(onClose)
  return (
    <div
      className="fixed inset-0 z-40 flex justify-end bg-black/40"
      onMouseDown={(event) => event.target === event.currentTarget && onClose?.()}
    >
      <aside
        role="dialog"
        aria-label={title}
        style={{ width }}
        className="flex h-full max-w-full flex-col border-l border-ink-600 bg-ink-800 shadow-[var(--level2-shadow)]"
      >
        <header className="flex items-center gap-3 border-b border-ink-600 px-4 py-3">
          <h2 className="flex-1 truncate text-[14px] font-semibold text-ink-100">{title}</h2>
          <CloseButton onClose={onClose} />
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{children}</div>
        {footer && <footer className="border-t border-ink-600 px-4 py-3">{footer}</footer>}
      </aside>
    </div>
  )
}

/** In-page heading row, for pages that want a title block below the top bar. */
export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="mb-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0 flex-1">
        <h2 className="text-[18px] font-semibold tracking-tight text-ink-100">{title}</h2>
        {subtitle && <p className="mt-0.5 text-[12.5px] text-ink-400">{subtitle}</p>}
      </div>
      {actions}
    </div>
  )
}
