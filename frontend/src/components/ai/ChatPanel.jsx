import { useCallback, useEffect, useRef, useState } from 'react'
import { api, streamChat } from '../../api'
import { Markdown } from '../../lib/markdown'

/* One conversation with the assistant, streamed as it is written.
 *
 * Used twice: in the assistant drawer (any question, `view` picks the suggested questions)
 * and under an alert's video (`eventId` set - the assistant then knows which alert "this"
 * is). `conversationId` continues an earlier conversation; `onConversation` hears the id
 * of a new one. */
export default function ChatPanel({
  eventId = '',
  view = '',
  conversationId = null,
  onConversation,
  placeholder = 'Ask about alerts, cameras or footage…',
  className = '',
}) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [suggestions, setSuggestions] = useState([])
  const idRef = useRef(conversationId)
  // The conversation already on screen. A new conversation's id coming back down from the
  // parent (via onConversation) must not abort the reply that is still streaming into it.
  const shownRef = useRef(undefined)
  const abortRef = useRef(null)
  const scrollRef = useRef(null)

  // Load an earlier conversation; a new one starts empty.
  useEffect(() => {
    if (conversationId === shownRef.current) return undefined
    shownRef.current = conversationId
    idRef.current = conversationId
    abortRef.current?.abort()
    setBusy(false)
    if (!conversationId) {
      setMessages([])
      return undefined
    }
    let live = true
    api
      .conversation(conversationId)
      .then((c) => live && setMessages(c.messages.map((m) => ({ role: m.role, content: m.content }))))
      .catch(() => live && setMessages([]))
    return () => {
      live = false
    }
  }, [conversationId])

  useEffect(() => {
    let live = true
    api
      .aiSuggestions({ view, eventId })
      .then((r) => live && setSuggestions(r.suggestions ?? []))
      .catch(() => {})
    return () => {
      live = false
    }
  }, [view, eventId])

  useEffect(() => () => abortRef.current?.abort(), [])

  useEffect(() => {
    const box = scrollRef.current
    if (box) box.scrollTop = box.scrollHeight
  }, [messages])

  const updateLast = (fn) =>
    setMessages((list) => [...list.slice(0, -1), fn(list[list.length - 1])])

  const send = useCallback(
    async (text) => {
      const question = text.trim()
      if (!question || busy) return
      setInput('')
      setBusy(true)
      setMessages((list) => [
        ...list,
        { role: 'user', content: question },
        { role: 'assistant', content: '', activity: [], pending: true },
      ])
      const controller = new AbortController()
      abortRef.current = controller
      try {
        await streamChat(
          { message: question, conversationId: idRef.current, eventId },
          (event) => {
            if (event.type === 'conversation') {
              if (!idRef.current) {
                idRef.current = event.id
                shownRef.current = event.id
                onConversation?.(event.id)
              }
            } else if (event.type === 'text') {
              updateLast((m) => ({ ...m, content: m.content + event.text }))
            } else if (event.type === 'tool') {
              updateLast((m) => ({ ...m, activity: [...(m.activity ?? []), event.label] }))
            } else if (event.type === 'error') {
              updateLast((m) => ({ ...m, error: event.message }))
            }
          },
          controller.signal,
        )
      } catch (err) {
        updateLast((m) => ({ ...m, error: err.message }))
      } finally {
        updateLast((m) => ({ ...m, pending: false }))
        setBusy(false)
      }
    },
    [busy, eventId, onConversation],
  )

  const stop = () => abortRef.current?.abort()

  return (
    <div className={`flex min-h-0 flex-col ${className}`}>
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {messages.length === 0 && (
          <div className="space-y-2">
            <p className="text-[12px] text-ink-400">
              {eventId ? 'Ask anything about this alert.' : 'Try one of these:'}
            </p>
            <div className="flex flex-wrap gap-1.5">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => send(s)}
                  className="rounded-full border border-ink-700 bg-ink-800 px-2.5 py-1 text-left text-[12px] text-ink-200 transition-colors hover:border-brand-600 hover:text-ink-100"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="flex justify-end">
              <p className="max-w-[85%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-brand-900 px-3 py-2 text-[13px] text-brand-100">
                {m.content}
              </p>
            </div>
          ) : (
            <div key={i} className="max-w-[95%] text-[13px] leading-relaxed text-ink-200">
              {m.activity?.length > 0 && (
                <p className="mb-1 text-[11px] text-ink-400">
                  {m.activity.join(' · ')}
                  {m.pending && !m.content && '…'}
                </p>
              )}
              {m.content ? (
                <Markdown text={m.content} />
              ) : (
                m.pending &&
                !m.activity?.length && <p className="text-[12px] text-ink-400">Thinking…</p>
              )}
              {m.error && <p className="mt-1 text-[12px] text-alarm-400">{m.error}</p>}
            </div>
          ),
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
        className="flex items-end gap-2 border-t border-ink-800 p-2.5"
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send(input)
            }
          }}
          rows={1}
          placeholder={placeholder}
          aria-label="Message"
          className="max-h-32 min-h-9.5 flex-1 resize-none rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-[13px] text-ink-100 placeholder:text-ink-500 focus:border-brand-500 focus:outline-none"
        />
        {busy ? (
          <button
            type="button"
            onClick={stop}
            className="h-9.5 rounded-lg border border-ink-700 bg-ink-800 px-3 text-[12.5px] text-ink-200 hover:bg-ink-700"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="h-9.5 rounded-lg border border-brand-500 bg-brand-500 px-3 text-[12.5px] font-semibold text-on-brand transition-colors hover:bg-brand-600 disabled:opacity-40"
          >
            Send
          </button>
        )}
      </form>
    </div>
  )
}
