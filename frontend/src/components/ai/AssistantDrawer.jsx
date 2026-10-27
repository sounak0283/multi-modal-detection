import { useCallback, useEffect, useState } from 'react'
import { api } from '../../api'
import { useToast } from '../Toast'
import { Markdown } from '../../lib/markdown'
import ChatPanel from './ChatPanel'

/* The assistant, on every page: a button at the bottom right opens a side panel with the
 * chat, earlier conversations, what it remembers, and the daily digests. Hidden entirely
 * when the server has no AI configured. Sits below the fire/smoke overlay (z-100) so an
 * alarm always covers it. */

const TABS = [
  ['chat', 'Chat'],
  ['history', 'History'],
  ['memory', 'Memory'],
  ['digest', 'Digest'],
]

export default function AssistantDrawer({ view, user }) {
  const [available, setAvailable] = useState(false)
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState('chat')
  const [conversationId, setConversationId] = useState(null)

  useEffect(() => {
    api
      .aiStatus()
      .then((s) => setAvailable(Boolean(s.available && s.chat)))
      .catch(() => setAvailable(false))
  }, [])

  useEffect(() => {
    if (!open) return undefined
    const close = (e) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [open])

  if (!available) return null

  const startNew = () => {
    setConversationId(null)
    setTab('chat')
  }

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="fixed right-5 bottom-5 z-90 flex items-center gap-2 rounded-full border border-brand-600 bg-brand-500 px-4 py-2.5 text-[13px] font-semibold text-on-brand shadow-2xl transition-colors hover:bg-brand-600"
        >
          <SparkIcon />
          Assistant
        </button>
      )}
      {open && (
        <aside
          aria-label="Assistant"
          className="fixed top-0 right-0 bottom-0 z-90 flex w-full max-w-110 flex-col border-l border-ink-700 bg-ink-900 shadow-2xl"
        >
          <header className="flex items-center gap-2 border-b border-ink-800 px-3 py-2.5">
            <SparkIcon />
            <strong className="text-[14px] tracking-tight">Assistant</strong>
            <div className="ml-auto flex items-center gap-1">
              <button
                type="button"
                onClick={startNew}
                className="rounded-md px-2 py-1 text-[12px] text-ink-300 hover:bg-ink-800 hover:text-ink-100"
              >
                New chat
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close assistant"
                className="rounded-md px-2 py-1 text-[16px] leading-none text-ink-300 hover:bg-ink-800 hover:text-ink-100"
              >
                ×
              </button>
            </div>
          </header>
          <nav className="flex border-b border-ink-800 px-2">
            {TABS.map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
                className={`border-b-2 px-3 py-2 text-[12.5px] transition-colors ${
                  tab === id
                    ? 'border-brand-500 font-medium text-ink-100'
                    : 'border-transparent text-ink-400 hover:text-ink-200'
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="flex min-h-0 flex-1 flex-col">
            {/* Kept mounted while on other tabs, so a reply keeps streaming in. */}
            <div className={tab === 'chat' ? 'flex min-h-0 flex-1 flex-col' : 'hidden'}>
              <ChatPanel
                className="flex-1"
                view={view}
                conversationId={conversationId}
                onConversation={setConversationId}
              />
            </div>
            {tab === 'history' && (
              <History
                current={conversationId}
                onOpen={(id) => {
                  setConversationId(id)
                  setTab('chat')
                }}
                onDeleted={(id) => id === conversationId && setConversationId(null)}
              />
            )}
            {tab === 'memory' && <Memory user={user} />}
            {tab === 'digest' && <Digests isAdmin={user?.role === 'admin'} />}
          </div>
        </aside>
      )}
    </>
  )
}

function History({ current, onOpen, onDeleted }) {
  const [items, setItems] = useState(null)
  const toast = useToast()

  const load = useCallback(() => {
    api
      .conversations()
      .then((r) => setItems(r.conversations ?? []))
      .catch((err) => {
        setItems([])
        toast(err.message, 'error')
      })
  }, [toast])

  useEffect(load, [load])

  const remove = async (id) => {
    try {
      await api.deleteConversation(id)
      onDeleted(id)
      load()
    } catch (err) {
      toast(err.message, 'error')
    }
  }

  if (items === null) return <Note>Loading…</Note>
  if (items.length === 0) return <Note>No conversations yet.</Note>
  return (
    <ul className="min-h-0 flex-1 overflow-y-auto p-2">
      {items.map((c) => (
        <li key={c.id} className="group flex items-center gap-2 rounded-lg hover:bg-ink-800">
          <button
            type="button"
            onClick={() => onOpen(c.id)}
            className="min-w-0 flex-1 px-2.5 py-2 text-left"
          >
            <span
              className={`block truncate text-[13px] ${c.id === current ? 'text-brand-500' : 'text-ink-100'}`}
            >
              {c.title}
            </span>
            <span className="text-[11px] text-ink-400">{when(c.updated_at)}</span>
          </button>
          <button
            type="button"
            onClick={() => remove(c.id)}
            className="mr-1 rounded-md px-2 py-1 text-[11.5px] text-ink-400 opacity-0 group-hover:opacity-100 hover:text-alarm-400"
          >
            Delete
          </button>
        </li>
      ))}
    </ul>
  )
}

const SCOPE_LABEL = { system: 'Site', user: 'About you', alert: 'Alert notes' }

function Memory({ user }) {
  const [items, setItems] = useState(null)
  const [text, setText] = useState('')
  const [scope, setScope] = useState('user')
  const toast = useToast()

  const load = useCallback(() => {
    api
      .memories()
      .then((r) => setItems(r.memories ?? []))
      .catch((err) => {
        setItems([])
        toast(err.message, 'error')
      })
  }, [toast])

  useEffect(load, [load])

  const add = async (e) => {
    e.preventDefault()
    if (!text.trim()) return
    try {
      await api.addMemory({ text, scope })
      setText('')
      load()
    } catch (err) {
      toast(err.message, 'error')
    }
  }

  const remove = async (id) => {
    try {
      await api.deleteMemory(id)
      load()
    } catch (err) {
      toast(err.message, 'error')
    }
  }

  const canDelete = (m) =>
    m.scope === 'user' || user?.role === 'admin' || m.created_by === user?.id

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <p className="border-b border-ink-800 px-3 py-2 text-[11.5px] leading-snug text-ink-400">
        What the assistant has kept from conversations. Site notes are shared with everyone;
        "About you" only you see. Delete anything that is wrong.
      </p>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {items === null && <Note>Loading…</Note>}
        {items?.length === 0 && <Note>Nothing remembered yet.</Note>}
        {['system', 'user', 'alert'].map((s) => {
          const group = (items ?? []).filter((m) => m.scope === s)
          if (group.length === 0) return null
          return (
            <section key={s}>
              <h3 className="mb-1 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-400">
                {SCOPE_LABEL[s]}
              </h3>
              <ul className="space-y-1">
                {group.map((m) => (
                  <li
                    key={m.id}
                    className="group flex items-start gap-2 rounded-lg border border-ink-800 px-2.5 py-2 text-[12.5px] text-ink-200"
                  >
                    <span className="flex-1">
                      {m.text}
                      {m.event_id && (
                        <span className="block text-[10.5px] text-ink-500">alert {m.event_id}</span>
                      )}
                    </span>
                    {canDelete(m) && (
                      <button
                        type="button"
                        onClick={() => remove(m.id)}
                        className="text-[11.5px] text-ink-400 opacity-0 group-hover:opacity-100 hover:text-alarm-400"
                      >
                        Delete
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          )
        })}
      </div>
      <form onSubmit={add} className="flex gap-2 border-t border-ink-800 p-2.5">
        <select
          value={scope}
          onChange={(e) => setScope(e.target.value)}
          aria-label="Who this is about"
          className="rounded-lg border border-ink-700 bg-ink-800 px-2 text-[12px] text-ink-100"
        >
          <option value="user">About me</option>
          <option value="system">About the site</option>
        </select>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Teach it a fact…"
          className="min-w-0 flex-1 rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-[12.5px] text-ink-100 placeholder:text-ink-500 focus:border-brand-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={!text.trim()}
          className="rounded-lg border border-ink-700 bg-ink-800 px-3 text-[12.5px] text-ink-100 hover:bg-ink-700 disabled:opacity-40"
        >
          Save
        </button>
      </form>
    </div>
  )
}

function Digests({ isAdmin }) {
  const [items, setItems] = useState(null)
  const [making, setMaking] = useState(false)
  const toast = useToast()

  const load = useCallback(() => {
    api
      .digests()
      .then((r) => setItems(r.digests ?? []))
      .catch((err) => {
        setItems([])
        toast(err.message, 'error')
      })
  }, [toast])

  useEffect(load, [load])

  const makeNow = async () => {
    setMaking(true)
    try {
      await api.runDigest({})
      load()
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setMaking(false)
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2 border-b border-ink-800 px-3 py-2">
        <p className="flex-1 text-[11.5px] leading-snug text-ink-400">
          Yesterday's alerts, summarised and emailed to admins each morning.
        </p>
        {isAdmin && (
          <button
            type="button"
            onClick={makeNow}
            disabled={making}
            className="shrink-0 rounded-lg border border-ink-700 bg-ink-800 px-2.5 py-1 text-[12px] text-ink-100 hover:bg-ink-700 disabled:opacity-40"
          >
            {making ? 'Writing…' : "Today so far"}
          </button>
        )}
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {items === null && <Note>Loading…</Note>}
        {items?.length === 0 && <Note>No digests yet. The first arrives tomorrow morning.</Note>}
        {(items ?? []).map((d) => (
          <article key={d.day} className="rounded-lg border border-ink-800 p-3">
            <h3 className="mb-1.5 text-[12px] font-semibold text-ink-300">
              {new Date(`${d.day}T12:00:00`).toLocaleDateString(undefined, {
                weekday: 'long',
                day: 'numeric',
                month: 'long',
              })}
            </h3>
            <div className="text-[12.5px] leading-relaxed text-ink-200">
              <Markdown text={d.text} />
            </div>
            {d.emailed_to?.length > 0 && (
              <p className="mt-2 text-[10.5px] text-ink-500">Emailed to {d.emailed_to.join(', ')}</p>
            )}
          </article>
        ))}
      </div>
    </div>
  )
}

function Note({ children }) {
  return <p className="px-4 py-8 text-center text-[12.5px] text-ink-400">{children}</p>
}

function when(iso) {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString(undefined, {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function SparkIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 1.8 9.4 6.6 14.2 8 9.4 9.4 8 14.2 6.6 9.4 1.8 8 6.6 6.6z" />
    </svg>
  )
}
