/* Design primitives.
 *
 * Every surface in the app is a Card, every action is a Button, every state is a Badge.
 * Keeping them here rather than repeating utility strings is what stops a dashboard
 * drifting into six slightly different greys and four button heights.
 */

const cx = (...parts) => parts.filter(Boolean).join(' ')

export function Card({ children, className = '' }) {
  return (
    <section
      className={cx(
        'rounded-xl border border-ink-700 bg-ink-900 overflow-hidden',
        className,
      )}
    >
      {children}
    </section>
  )
}

export function CardHead({ title, aside, children }) {
  return (
    <header className="flex items-center gap-3 border-b border-ink-800 px-4 py-3">
      <h2 className="flex-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-400">
        {title}
      </h2>
      {aside}
      {children}
    </header>
  )
}

export function CardBody({ children, className = '' }) {
  return <div className={cx('p-4', className)}>{children}</div>
}

export function CardFoot({ children, className = '' }) {
  return (
    <footer className={cx('border-t border-ink-800 px-4 py-3', className)}>{children}</footer>
  )
}

const BUTTON_VARIANTS = {
  default: 'bg-ink-800 border-ink-700 text-ink-100 hover:bg-ink-700 hover:border-ink-600',
  primary:
    'bg-brand-500 border-brand-500 text-[#04122a] font-semibold hover:bg-brand-600 hover:border-brand-600',
  danger: 'bg-transparent border-alarm-900 text-alarm-400 hover:bg-alarm-900',
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
        'inline-flex items-center justify-center gap-2 rounded-lg border px-3 py-1.5 text-[13px]',
        'transition-colors disabled:opacity-40 disabled:cursor-not-allowed',
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
  neutral: 'border-ink-700 bg-ink-800 text-ink-300',
  ok: 'border-ok-900 bg-ok-900 text-ok-400',
  warn: 'border-warn-900 bg-warn-900 text-warn-400',
  alarm: 'border-alarm-900 bg-alarm-900 text-alarm-400',
  info: 'border-brand-900 bg-brand-900 text-brand-500',
}

export function Badge({ tone = 'neutral', children, className = '' }) {
  return (
    <span
      className={cx(
        'inline-block rounded-full border px-2 py-0.5 text-[11px] font-semibold capitalize',
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/** Status dot. `tone` null renders the inactive/unknown state rather than hiding. */
export function Dot({ tone, className = '' }) {
  const tones = {
    ok: 'bg-ok-400 shadow-[0_0_0_3px_rgba(52,211,153,0.16)]',
    alarm: 'bg-alarm-400 shadow-[0_0_0_3px_rgba(248,113,113,0.16)]',
    warn: 'bg-warn-400 shadow-[0_0_0_3px_rgba(251,191,36,0.16)]',
  }
  return (
    <span
      className={cx('h-[7px] w-[7px] shrink-0 rounded-full', tones[tone] || 'bg-ink-500', className)}
    />
  )
}

export function Field({ label, hint, children, className = '' }) {
  return (
    <label className={cx('block', className)}>
      {label && <span className="mb-1.5 block text-[12.5px] text-ink-300">{label}</span>}
      {children}
      {hint && <span className="mt-1 block text-[11.5px] leading-snug text-ink-400">{hint}</span>}
    </label>
  )
}

const CONTROL =
  'w-full rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-[13px] text-ink-100 '
  + 'placeholder:text-ink-500 focus:border-brand-500 focus:outline-none'

export function Input({ className = '', ...rest }) {
  return <input className={cx(CONTROL, className)} {...rest} />
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
              'rounded-full border px-3 py-1 text-[12px] transition-colors',
              on
                ? 'border-brand-600 bg-brand-900 text-brand-500'
                : 'border-ink-700 bg-ink-800 text-ink-300 hover:border-ink-600',
            )}
          >
            {label}
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

export function Stat({ value, label, tone }) {
  const tones = { alarm: 'text-alarm-400', warn: 'text-warn-400', ok: 'text-ok-400' }
  return (
    <Card className="px-4 py-3.5">
      <div className={cx('text-[26px] font-semibold leading-tight tabular-nums', tones[tone])}>
        {value}
      </div>
      <div className="mt-0.5 text-[10.5px] uppercase tracking-[0.08em] text-ink-400">{label}</div>
    </Card>
  )
}

export function EmptyState({ children }) {
  return <p className="px-4 py-8 text-center text-[12.5px] text-ink-400">{children}</p>
}
