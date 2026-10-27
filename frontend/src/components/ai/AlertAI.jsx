import { useCallback, useEffect, useState } from 'react'
import { api } from '../../api'
import { useAIAvailable } from '../../lib/useAIAvailable'
import ChatPanel from './ChatPanel'

/* Under an alert's video or snapshot (Evidence page, Alert history viewer): what the AI
 * saw in the clip, and a chat about this one alert. The chat picks up the latest earlier
 * conversation about the same alert. Renders nothing when the server has no AI. */
export default function AlertAI({ event, chatHeight = 'h-[360px]' }) {
  const available = useAIAvailable()
  if (!available || !event?.id) return null
  return (
    <div className="perimeter-card overflow-hidden rounded-xl border border-ink-700 bg-ink-900">
      {event.clip_path && <Analysis key={`a-${event.id}`} eventId={event.id} />}
      <AlertChat key={`c-${event.id}`} eventId={event.id} height={chatHeight} />
    </div>
  )
}

const POLL_MS = 3000

function Analysis({ eventId }) {
  const [state, setState] = useState({ loading: true })

  const load = useCallback(async () => {
    try {
      setState({ data: await api.eventAnalysis(eventId) })
    } catch (err) {
      setState(err.message.includes('not been analysed') ? { data: null } : { error: err.message })
    }
  }, [eventId])

  useEffect(() => {
    load()
  }, [load])

  const status = state.data?.status
  useEffect(() => {
    if (status !== 'queued' && status !== 'running') return undefined
    const id = setInterval(load, POLL_MS)
    return () => clearInterval(id)
  }, [status, load])

  const start = async () => {
    try {
      setState({ data: await api.analyzeEvent(eventId) })
    } catch (err) {
      setState({ error: err.message })
    }
  }

  const analysis = state.data?.analysis
  return (
    <section className="border-b border-ink-800 p-4">
      <div className="mb-2 flex items-center gap-2">
        <h2 className="flex-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-400">
          What the AI saw in the video
        </h2>
        {(status === 'done' || status === 'failed') && (
          <button
            type="button"
            onClick={start}
            className="rounded-md px-2 py-0.5 text-[11.5px] text-ink-400 hover:bg-ink-800 hover:text-ink-100"
          >
            Analyse again
          </button>
        )}
      </div>
      {state.loading && <p className="text-[12.5px] text-ink-400">Loading…</p>}
      {state.error && <p className="text-[12.5px] text-alarm-400">{state.error}</p>}
      {!state.loading && !state.error && state.data === null && (
        <div className="flex items-center gap-3">
          <p className="flex-1 text-[12.5px] text-ink-400">This video has not been analysed yet.</p>
          <button
            type="button"
            onClick={start}
            className="rounded-lg border border-brand-500 bg-brand-500 px-3 py-1.5 text-[12.5px] font-semibold text-on-brand hover:bg-brand-600"
          >
            Analyse video
          </button>
        </div>
      )}
      {(status === 'queued' || status === 'running') && (
        <p className="text-[12.5px] text-ink-300">
          Watching the video… this usually takes 15–20 seconds.
        </p>
      )}
      {status === 'failed' && (
        <p className="text-[12.5px] text-alarm-400">
          The analysis did not finish{state.data.error ? `: ${state.data.error}` : '.'}
        </p>
      )}
      {status === 'done' && analysis && (
        <div className="space-y-2 text-[13px] leading-relaxed text-ink-200">
          <p>{analysis.summary}</p>
          {analysis.findings && analysis.findings !== analysis.summary && (
            <p className="text-ink-300">{analysis.findings}</p>
          )}
          <div className="flex flex-wrap gap-1.5 pt-0.5">
            {analysis.people_count != null && (
              <Chip>
                {analysis.people_count} {analysis.people_count === 1 ? 'person' : 'people'}
              </Chip>
            )}
            {(analysis.activities ?? []).map((a) => (
              <Chip key={a}>{a}</Chip>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

function AlertChat({ eventId, height }) {
  const [conversationId, setConversationId] = useState(undefined) // undefined = looking up

  useEffect(() => {
    let live = true
    api
      .conversations(eventId)
      .then((r) => live && setConversationId(r.conversations?.[0]?.id ?? null))
      .catch(() => live && setConversationId(null))
    return () => {
      live = false
    }
  }, [eventId])

  return (
    <section>
      <div className="flex items-center gap-2 px-4 pt-3">
        <h2 className="flex-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-400">
          Ask about this alert
        </h2>
        {conversationId && (
          <button
            type="button"
            onClick={() => setConversationId(null)}
            className="rounded-md px-2 py-0.5 text-[11.5px] text-ink-400 hover:bg-ink-800 hover:text-ink-100"
          >
            New chat
          </button>
        )}
      </div>
      {conversationId !== undefined && (
        <ChatPanel
          className={height}
          eventId={eventId}
          conversationId={conversationId}
          onConversation={setConversationId}
          placeholder="Ask about this video or snapshot…"
        />
      )}
    </section>
  )
}

function Chip({ children }) {
  return (
    <span className="rounded-full border border-ink-700 bg-ink-800 px-2 py-0.5 text-[11.5px] text-ink-300">
      {children}
    </span>
  )
}
