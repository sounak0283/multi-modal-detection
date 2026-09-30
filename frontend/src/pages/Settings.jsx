import { useState } from 'react'
import { Card, CardBody, CardHead, Toggle } from '../components/ui'
import { useToast } from '../components/Toast'
import { api } from '../api'
import { THEMES, getTheme, setTheme } from '../lib/theme'

const THEME_LABELS = { dark: 'Dark', light: 'Light' }

function Appearance() {
  const [theme, setThemeState] = useState(getTheme)
  return (
    <Card>
      <CardHead title="Appearance" />
      <CardBody className="space-y-3">
        <div
          role="radiogroup"
          aria-label="Theme"
          className="inline-flex overflow-hidden rounded-lg border border-ink-700"
        >
          {THEMES.map((value) => (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={theme === value}
              onClick={() => setThemeState(setTheme(value))}
              className={`px-4 py-1.5 text-[12.5px] transition-colors not-first:border-l not-first:border-ink-700 ${
                theme === value
                  ? 'bg-brand-500 font-semibold text-on-brand'
                  : 'bg-ink-800 text-ink-200 hover:bg-ink-700'
              }`}
            >
              {THEME_LABELS[value]}
            </button>
          ))}
        </div>
        <p className="text-[12px] text-ink-400">
          Saved in this browser only. Dark suits a control room next to a video wall; camera
          video is never recoloured.
        </p>
      </CardBody>
    </Card>
  )
}

export default function Settings({ features, onChanged }) {
  const toast = useToast()
  const [busy, setBusy] = useState(false)
  const enabled = features?.video_test !== false

  const setVideoTest = async (next) => {
    setBusy(true)
    try {
      await api.saveAppSettings({ video_test_enabled: next })
      toast(next ? 'Video test is on.' : 'Video test is off.', 'ok')
      await onChanged?.()
    } catch (err) {
      toast(err.message || 'Could not save the setting.', 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="max-w-2xl space-y-4">
      <Appearance />
      <Card>
        <CardHead title="Video test" />
        <CardBody className="space-y-3">
          <p className="text-[13px] text-ink-200">
            Show the Video test page, where an admin can upload a recording and run it through the
            detectors as if it were a live camera.
          </p>
          <Toggle
            checked={enabled}
            onChange={busy ? () => {} : setVideoTest}
            label={enabled ? 'Video test is on' : 'Video test is off'}
          />
          <p className="text-[12px] text-ink-400">
            Turning this off hides the page, stops any running test and rejects uploads. Uploaded
            files are kept until they expire.
          </p>
        </CardBody>
      </Card>
    </div>
  )
}
