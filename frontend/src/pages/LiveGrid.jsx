import { useMemo } from 'react'
import { Card, CardHead, Dot, EmptyState } from '../components/ui'
import { streamUrl } from '../api'

/* All-cameras grid for Live view.
 *
 * A lighter sibling of LiveView: no zone overlay, no per-camera stat polling - just
 * every camera's live MJPEG stream (already annotated server-side with track boxes and
 * the status bar, see pipeline.py:_render) in one glance, so an operator watching
 * several cameras does not have to flip through the single-camera dropdown one at a
 * time. Clicking a tile jumps to that camera's full single-camera Live view, where the
 * zone overlay and stat tiles live.
 */
export default function LiveGrid({ cameras, onSelect }) {
  // One shared cache-buster for the whole grid: it only needs to change on mount (a
  // fresh grid should not reuse a stale stream URL from before), not per camera.
  const bust = useMemo(() => Date.now(), [])

  if (cameras.length === 0) {
    return (
      <Card>
        <EmptyState>No cameras yet.</EmptyState>
      </Card>
    )
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {cameras.map((camera) => (
        <Card key={camera.id} className="p-0">
          <CardHead
            title={camera.name || camera.id}
            aside={
              <span className="flex items-center gap-1.5 text-[11px] text-ink-400">
                <Dot tone={camera.running ? 'ok' : camera.enabled ? 'warn' : undefined} />
                {camera.running ? 'Running' : camera.enabled ? 'Enabled, not started' : 'Disabled'}
              </span>
            }
          />
          <button
            type="button"
            onClick={() => onSelect(camera.id)}
            className="block w-full bg-[#05070a]"
            title={`Open ${camera.name || camera.id} in single view`}
          >
            {camera.running ? (
              <img
                src={streamUrl(camera.id, bust)}
                alt={`${camera.name || camera.id} live view`}
                className="block aspect-video w-full object-contain"
              />
            ) : (
              <div className="flex aspect-video w-full items-center justify-center text-[12.5px] text-ink-500">
                Not streaming
              </div>
            )}
          </button>
        </Card>
      ))}
    </div>
  )
}
