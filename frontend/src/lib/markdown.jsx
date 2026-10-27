/* The little markdown the assistant writes: paragraphs, bullet and numbered lists,
 * ### headings, **bold** and `code`. Built as React elements, never as HTML, so nothing in
 * a reply can inject markup. Anything else shows as plain text. */

function inline(text, keyBase) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    const key = `${keyBase}-${i}`
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return (
        <strong key={key} className="font-semibold text-ink-100">
          {part.slice(2, -2)}
        </strong>
      )
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code key={key} className="rounded bg-ink-800 px-1 font-mono text-[0.92em]">
          {part.slice(1, -1)}
        </code>
      )
    }
    return part
  })
}

export function Markdown({ text }) {
  const blocks = []
  let list = null
  const flush = () => {
    if (list) blocks.push(list)
    list = null
  }

  text.split('\n').forEach((raw, index) => {
    const line = raw.trimEnd()
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    const numbered = line.match(/^\s*(\d+)[.)]\s+(.*)$/)
    if (bullet || numbered) {
      const ordered = Boolean(numbered)
      if (!list || list.ordered !== ordered) {
        flush()
        list = { type: 'list', ordered, items: [] }
      }
      list.items.push(bullet ? bullet[1] : numbered[2])
      return
    }
    flush()
    if (!line.trim()) {
      blocks.push({ type: 'gap', index })
      return
    }
    const heading = line.match(/^#{1,4}\s+(.*)$/)
    blocks.push(heading ? { type: 'heading', text: heading[1] } : { type: 'p', text: line })
  })
  flush()

  return (
    <div className="space-y-1.5 break-words">
      {blocks.map((block, i) => {
        if (block.type === 'gap') return null
        if (block.type === 'heading') {
          return (
            <p key={i} className="pt-1 font-semibold text-ink-100">
              {inline(block.text, i)}
            </p>
          )
        }
        if (block.type === 'list') {
          const Tag = block.ordered ? 'ol' : 'ul'
          return (
            <Tag key={i} className={`space-y-1 pl-5 ${block.ordered ? 'list-decimal' : 'list-disc'}`}>
              {block.items.map((item, j) => (
                <li key={j}>{inline(item, `${i}-${j}`)}</li>
              ))}
            </Tag>
          )
        }
        return <p key={i}>{inline(block.text, i)}</p>
      })}
    </div>
  )
}
