#!/usr/bin/env node
/* npm licence gate.
 *
 * The Python gate (pip-licenses) cannot see npm packages at all, so without this the
 * frontend is a hole straight through the licence policy - and it is the half that
 * actually ships compiled into the bundle a customer receives.
 *
 * Written by hand rather than pulling in license-checker: reading a `license` field out
 * of package.json is thirty lines, and adding a dependency to audit dependencies has an
 * obvious problem.
 */

import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')
const MODULES = join(ROOT, 'node_modules')

// Mirrors NOTICE.md. Anything not on this list fails and must be reviewed, rather than
// being waved through because it looked familiar.
const ALLOWED = new Set([
  'MIT', 'ISC', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', '0BSD',
  'BlueOak-1.0.0', 'CC0-1.0', 'Unlicense', 'Python-2.0',
  '(MIT OR CC0-1.0)', '(MIT OR Apache-2.0)', 'MIT-0',

  // MPL-2.0: file-level copyleft, accepted UNMODIFIED ONLY - the same position already
  // recorded for certifi and tqdm on the Python side. Reaches us via lightningcss,
  // which Tailwind uses to transform CSS at BUILD time; no MPL code is in the shipped
  // bundle. Patching it would trigger a publication obligation on the patched file.
  'MPL-2.0',

  // CC-BY-4.0: caniuse-lite's browser-support data, consumed by the build to decide
  // which CSS prefixes to emit. Attribution is required and is recorded in NOTICE.md.
  // Data, not code, and not shipped.
  'CC-BY-4.0',
])

const FORBIDDEN = /GPL|AGPL|LGPL|CC-BY-NC|CC-BY-SA|SSPL|BUSL|Commons Clause|Proprietary/i

function* packages(dir) {
  let entries
  try {
    entries = readdirSync(dir)
  } catch {
    return
  }
  for (const entry of entries) {
    const path = join(dir, entry)
    if (entry.startsWith('@')) {
      yield* packages(path)
      continue
    }
    // Skip tooling scratch directories (.bin, .vite-temp, .cache): they are not
    // packages and have no licence to check.
    if (entry.startsWith('.') || !statSync(path).isDirectory()) continue
    try {
      const manifest = JSON.parse(readFileSync(join(path, 'package.json'), 'utf8'))
      yield { name: manifest.name ?? entry, license: normalise(manifest) }
    } catch {
      yield { name: entry, license: null }
    }
    yield* packages(join(path, 'node_modules'))
  }
}

function normalise(manifest) {
  if (typeof manifest.license === 'string') return manifest.license
  if (manifest.license?.type) return manifest.license.type
  if (Array.isArray(manifest.licenses)) return manifest.licenses.map((l) => l.type ?? l).join(' OR ')
  return null
}

const found = [...packages(MODULES)]
const problems = found.filter(
  ({ license }) => license === null || FORBIDDEN.test(license) || !ALLOWED.has(license),
)

const counts = new Map()
for (const { license } of found) counts.set(license, (counts.get(license) ?? 0) + 1)

console.log(`Checked ${found.length} npm package(s).`)
for (const [license, n] of [...counts].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${String(n).padStart(4)}  ${license ?? '(none declared)'}`)
}

if (problems.length) {
  console.error(`\nFAILED - ${problems.length} package(s) need review:`)
  for (const { name, license } of problems) {
    console.error(`  ${name}: ${license ?? '(no licence declared)'}`)
  }
  console.error('\nAdd to the allow-list in this script only after recording it in NOTICE.md.')
  process.exit(1)
}

console.log('\nAll npm licences are permissive and on the allow-list.')
