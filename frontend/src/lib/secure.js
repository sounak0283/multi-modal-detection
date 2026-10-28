/* Storage addresses are infrastructure detail (bucket names, key layout), not something
 * to put on screen or in a screenshot. The UI shows a masked form that still lets an
 * operator tell clips apart, and offers "Copy secure link" for the real thing. */

/** s3://my-bucket/events/2026/10/cam_01/abc.mp4 -> s3://••••/…/abc.mp4 */
export function maskStoragePath(uri) {
  if (!uri) return ''
  const name = String(uri).split('?')[0].split('/').filter(Boolean).pop() ?? ''
  return `s3://••••••/…/${name}`
}

/** Copy text to the clipboard; resolves true on success. Never throws. */
export async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
