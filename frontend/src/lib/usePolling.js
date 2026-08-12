import { useEffect, useRef, useState } from 'react'

/** Poll `fn` every `intervalMs`, skipping while `enabled` is false.
 *
 * Errors are swallowed into `error` rather than thrown: a dashboard that unmounts
 * itself because one health poll failed during a backend restart is worse than one
 * showing a stale number for two seconds.
 */
export function usePolling(fn, intervalMs, enabled = true) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const saved = useRef(fn)
  saved.current = fn

  useEffect(() => {
    if (!enabled) return undefined
    let cancelled = false

    const tick = async () => {
      try {
        const result = await saved.current()
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) setError(err)
      }
    }

    tick()
    const id = setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [intervalMs, enabled])

  return { data, error }
}
