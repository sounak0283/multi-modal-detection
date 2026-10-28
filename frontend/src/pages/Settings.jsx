import { useState } from 'react'
import { Card, CardHead, SegmentedControl, Toggle } from '../components/ui'
import { useToast } from '../components/Toast'
import { api } from '../api'
import { THEMES, getTheme, setTheme } from '../lib/theme'

const THEME_LABELS = { dark: 'Dark', light: 'Light' }

/* One setting per row: what it is and why on the left, the control on the right. */
function SettingRow({ title, description, children }) {
  return (
    <div className="flex flex-col gap-3 border-t border-ink-700 px-4 py-4 first:border-t-0 sm:flex-row sm:items-center sm:gap-8">
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium text-ink-100">{title}</p>
        <p className="mt-1 text-[12px] leading-snug text-ink-400">{description}</p>
      </div>
      <div className="shrink-0 sm:w-56">{children}</div>
    </div>
  )
}

function Appearance() {
  const [theme, setThemeState] = useState(getTheme)
  return (
    <Card>
      <CardHead title="Appearance" />
      <SettingRow
        title="Theme"
        description="Saved in this browser only. Dark suits a control room next to a video wall; camera video is never recoloured."
      >
        <SegmentedControl
          className="w-full [&>button]:flex-1"
          value={theme}
          onChange={(value) => setThemeState(setTheme(value))}
          options={THEMES.map((value) => ({ value, label: THEME_LABELS[value] }))}
        />
      </SettingRow>
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
    <div className="max-w-3xl space-y-4">
      <Appearance />
      <Card>
        <CardHead title="Video test" />
        <SettingRow
          title="Show the Video test page"
          description="An admin can upload a recording and run it through the detectors as if it were a live camera. Turning this off hides the page, stops any running test and rejects uploads. Uploaded files are kept until they expire."
        >
          <Toggle
            checked={enabled}
            onChange={busy ? () => {} : setVideoTest}
            label={enabled ? 'On' : 'Off'}
          />
        </SettingRow>
      </Card>
    </div>
  )
}
