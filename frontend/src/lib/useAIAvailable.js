import { useEffect, useState } from 'react'
import { api } from '../api'

/* Whether the server has the AI features on. Asked once per page load and shared by every
 * component that needs it (the answer only changes with a server restart). */
let cached = null

export function useAIAvailable() {
  const [available, setAvailable] = useState(false)
  useEffect(() => {
    let live = true
    cached ??= api
      .aiStatus()
      .then((s) => Boolean(s.available))
      .catch(() => {
        cached = null // try again next time rather than hiding AI for the whole session
        return false
      })
    cached.then((value) => live && setAvailable(value))
    return () => {
      live = false
    }
  }, [])
  return available
}
