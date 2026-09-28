import { useCallback, useEffect, useRef, useState } from 'react'
import { streamUrl } from '../api'

const RETRY_MS = 3000

/** A camera's MJPEG stream URL that re-requests itself after a load error.
 *
 * An <img> never retries on its own, so a stream opened while the camera was still
 * starting (a webcam can take ~40 s, and the server answers 503 until then) otherwise
 * stays broken until the page is refreshed - even after the camera goes live.
 */
export function useRetryingStream(cameraId) {
  const [bust, setBust] = useState(() => Date.now())
  const timer = useRef(null)

  useEffect(() => () => clearTimeout(timer.current), [])

  const onError = useCallback(() => {
    clearTimeout(timer.current)
    timer.current = setTimeout(() => setBust(Date.now()), RETRY_MS)
  }, [])

  return { src: streamUrl(cameraId, bust), onError }
}
