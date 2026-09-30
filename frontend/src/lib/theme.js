/* Dark / light theme. A per-browser preference: kept in localStorage and applied as
 * <html data-theme="...">, which index.css keys the light palette off. index.html applies
 * the stored value before first paint (no dark flash on reload); this module owns reads
 * and writes after that. */

export const THEMES = ['dark', 'light']
const KEY = 'perimeter.theme'

export function getTheme() {
  try {
    return localStorage.getItem(KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

export function setTheme(theme) {
  const next = theme === 'light' ? 'light' : 'dark'
  document.documentElement.dataset.theme = next
  try {
    localStorage.setItem(KEY, next)
  } catch {
    /* private window / storage blocked - applies for this page view only */
  }
  return next
}
