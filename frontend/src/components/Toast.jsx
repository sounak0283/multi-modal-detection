import { createContext, useCallback, useContext, useMemo, useState } from 'react'

const ToastContext = createContext(() => {})

export const useToast = () => useContext(ToastContext)

export function ToastProvider({ children }) {
  const [toast, setToast] = useState(null)

  const show = useCallback((message, tone = 'info') => {
    setToast({ message, tone, key: Date.now() })
    // Errors stay longer: they usually name a zone and a reason the installer needs to
    // read and act on, not just acknowledge.
    setTimeout(() => setToast(null), tone === 'error' ? 7000 : 4000)
  }, [])

  const value = useMemo(() => show, [show])

  const accent = {
    error: 'border-l-alarm-400',
    ok: 'border-l-ok-400',
    info: 'border-l-brand-500',
  }[toast?.tone || 'info']

  return (
    <ToastContext.Provider value={value}>
      {children}
      {toast && (
        <div
          role="status"
          key={toast.key}
          className={`fixed bottom-6 left-1/2 z-50 max-w-[72vw] -translate-x-1/2 rounded-xl border border-ink-700 border-l-[3px] bg-ink-700 px-4 py-2.5 text-[13px] shadow-2xl ${accent}`}
        >
          {toast.message}
        </div>
      )}
    </ToastContext.Provider>
  )
}
