import { useEffect, useRef } from 'react'
import { api } from '../api'

const EVENTS_POLL_MS = 2500
const CONFIG_POLL_MS = 10000
const SEVERITY_RANK = { low: 0, medium: 1, high: 2, critical: 3 }
// See FireAlertOverlay's identical cap - same reasoning, same shape of `seen` Set.
const SEEN_CAP = 500

/* Plays a short beep in the browser for a new alert meeting the configured severity
 * floor (Expansion Plan Phase D). Always mounted, renders nothing - same shape as
 * FireAlertOverlay (dedup via a `seen` Set, independent poll interval) but for sound
 * rather than a modal, and driven by the admin-configurable sound_enabled/
 * sound_min_severity from /api/alert-config rather than a fixed kind filter.
 *
 * Browsers require a prior user gesture before audio can play - a known limitation
 * (PLATFORM_EXPANSION_PLAN.md's risk register), not something this component works
 * around. Email is the reliable, always-on backstop channel; this is a convenience for
 * whoever already has the dashboard open and interacted with. */
export default function AlertSoundPlayer() {
  const seen = useRef(new Set())
  const primed = useRef(false)
  const configRef = useRef({ sound_enabled: true, sound_min_severity: 'high' })
  const audioCtxRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    const pollConfig = async () => {
      try {
        const config = await api.getAlertConfig()
        if (!cancelled) configRef.current = config
      } catch {
        /* keep the last known config; the status rail already shows backend issues */
      }
    }

    pollConfig()
    const id = setInterval(pollConfig, CONFIG_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const pollEvents = async () => {
      let page
      try {
        page = await api.liveEvents(20)
      } catch {
        return
      }
      if (cancelled) return

      const { sound_enabled, sound_min_severity } = configRef.current
      const floor = SEVERITY_RANK[sound_min_severity] ?? SEVERITY_RANK.high

      // The first poll after a page load only records what already happened: the server
      // keeps recent alerts in memory, and a reload must not replay a beep for each one.
      const priming = !primed.current
      primed.current = true

      for (const event of page.events || []) {
        const key = `${event.ts}|${event.kind}|${event.message}`
        if (seen.current.has(key)) continue
        seen.current.add(key)
        if (seen.current.size > SEEN_CAP) {
          seen.current.delete(seen.current.values().next().value)
        }

        const rank = SEVERITY_RANK[event.severity] ?? SEVERITY_RANK.medium
        if (!priming && sound_enabled && rank >= floor) beep()
      }
    }

    pollEvents()
    const id = setInterval(pollEvents, EVENTS_POLL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const beep = () => {
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext
      if (!Ctx) return
      const ctx = audioCtxRef.current || new Ctx()
      audioCtxRef.current = ctx

      const oscillator = ctx.createOscillator()
      const gain = ctx.createGain()
      oscillator.type = 'sine'
      oscillator.frequency.value = 880
      gain.gain.setValueAtTime(0.001, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.02)
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5)

      oscillator.connect(gain)
      gain.connect(ctx.destination)
      oscillator.start()
      oscillator.stop(ctx.currentTime + 0.5)
    } catch {
      /* autoplay blocked (no prior user gesture) or Web Audio unavailable - silent */
    }
  }

  return null
}
