// 更新已存在的 GitHub 仓库：把所有文件做成一个提交推上去。
// 用途：本机 github.com:443 直连不通（git push 用不了），只走 api.github.com。
// 令牌从 `gh auth token` 拿，**不打印、不落盘**。
import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const REPO = process.argv[2]
const ROOT = process.argv[3]
const MSG = process.argv[4] || 'chore: update'
const token = execFileSync('gh', ['auth', 'token'], { encoding: 'utf8' }).trim()
const H = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', 'User-Agent': 'node-fetch' }

async function api(path, body, method = 'POST') {
  const r = await fetch(`https://api.github.com${path}`, { method, headers: H, body: body ? JSON.stringify(body) : undefined })
  const j = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(`${method} ${path} → ${r.status} ${JSON.stringify(j).slice(0, 400)}`)
  return j
}
const b64 = (p) => readFileSync(p).toString('base64').replace(/\n/g, '')
function walk(dir, out = []) {
  for (const e of readdirSync(dir)) {
    if (['.git', 'node_modules', '__pycache__'].includes(e)) continue
    const p = join(dir, e)
    statSync(p).isDirectory() ? walk(p, out) : out.push(p)
  }
  return out
}

const files = walk(ROOT).map((f) => ({ abs: f, rel: relative(ROOT, f).split('\\').join('/') }))
const ref = await api(`/repos/${REPO}/git/ref/heads/main`, null, 'GET')
const parent = await api(`/repos/${REPO}/git/commits/${ref.object.sha}`, null, 'GET')

const tree = []
for (const f of files) {
  const b = await api(`/repos/${REPO}/git/blobs`, { content: b64(f.abs), encoding: 'base64' })
  tree.push({ path: f.rel, mode: '100644', type: 'blob', sha: b.sha })
  console.log(`  blob ${f.rel}  ${b.sha.slice(0, 8)}`)
}
const t = await api(`/repos/${REPO}/git/trees`, { base_tree: parent.tree.sha, tree })
const c = await api(`/repos/${REPO}/git/commits`, { message: MSG, tree: t.sha, parents: [ref.object.sha] })
await api(`/repos/${REPO}/git/refs/heads/main`, { sha: c.sha, force: true }, 'PATCH')
console.log(`\n✓ 已更新 → https://github.com/${REPO}   commit ${c.sha.slice(0, 8)}  共 ${files.length} 个文件`)
