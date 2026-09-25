// tools/push_via_api.mjs
//
// 用途：**当 `git push` 用不了的时候**（典型症状：`github.com:443` 直连超时 /
// Connection reset，但 `gh auth status` 和 `api.github.com` 都正常），
// 改走 GitHub 的 Git Data API 把工作区推上去。
//
// 用法：
//   node tools/push_via_api.mjs <owner>/<repo> [目录，默认当前] [提交信息]
//
// 令牌从 `gh auth token` 取（需要先 `gh auth login`），**不打印、不落盘**。
// 特点：把目录里所有文件做成**一个提交**；磁盘上没有、但仓库里有的文件会被**删除**。

import { execFileSync } from 'node:child_process'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'

const REPO = process.argv[2]
const ROOT = process.argv[3] || '.'
const MSG = process.argv[4] || 'chore: update'
const BRANCH = process.env.BRANCH || 'main'
if (!REPO) throw new Error('用法：node tools/push_via_api.mjs <owner>/<repo> [目录] [提交信息]')

const SKIP = new Set(['.git', 'node_modules', '__pycache__', '.venv', 'out'])
const token = execFileSync('gh', ['auth', 'token'], { encoding: 'utf8' }).trim()
const H = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', 'User-Agent': 'push-via-api' }

async function api(path, body, method = 'POST') {
  const r = await fetch(`https://api.github.com${path}`, {
    method, headers: H, body: body ? JSON.stringify(body) : undefined,
  })
  const j = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(`${method} ${path} → ${r.status} ${JSON.stringify(j).slice(0, 400)}`)
  return j
}

function walk(dir, out = []) {
  for (const e of readdirSync(dir)) {
    if (SKIP.has(e)) continue
    const p = join(dir, e)
    statSync(p).isDirectory() ? walk(p, out) : out.push(p)
  }
  return out
}

const files = walk(ROOT).map((f) => ({ abs: f, rel: relative(ROOT, f).split('\\').join('/') }))

let ref, parent = null
try {
  ref = await api(`/repos/${REPO}/git/ref/heads/${BRANCH}`, null, 'GET')
  parent = await api(`/repos/${REPO}/git/commits/${ref.object.sha}`, null, 'GET')
} catch {
  // 空仓库：Git Data API 会 409，必须先建一个初始提交
  const seed = files[0]
  await api(`/repos/${REPO}/contents/${encodeURIComponent(seed.rel)}`, {
    message: 'chore: init repository',
    content: readFileSync(seed.abs).toString('base64').replace(/\n/g, ''),
    branch: BRANCH,
  }, 'PUT')
  ref = await api(`/repos/${REPO}/git/ref/heads/${BRANCH}`, null, 'GET')
  parent = await api(`/repos/${REPO}/git/commits/${ref.object.sha}`, null, 'GET')
  console.log('  空仓库 → 已建初始提交')
}

const tree = []
for (const f of files) {
  const b = await api(`/repos/${REPO}/git/blobs`, {
    content: readFileSync(f.abs).toString('base64').replace(/\n/g, ''), encoding: 'base64',
  })
  tree.push({ path: f.rel, mode: '100644', type: 'blob', sha: b.sha })
  console.log(`  blob ${f.rel}  ${b.sha.slice(0, 8)}`)
}

// 磁盘上没有、仓库里有的文件 → 删除（sha: null）
const existing = await api(`/repos/${REPO}/git/trees/${parent.tree.sha}?recursive=1`, null, 'GET')
const onDisk = new Set(files.map((f) => f.rel))
for (const e of existing.tree || []) {
  if (e.type === 'blob' && !onDisk.has(e.path) && !e.path.startsWith('tools/push_via_api')) {
    tree.push({ path: e.path, mode: '100644', type: 'blob', sha: null })
    console.log(`  del  ${e.path}`)
  }
}

const t = await api(`/repos/${REPO}/git/trees`, { base_tree: parent.tree.sha, tree })
const c = await api(`/repos/${REPO}/git/commits`, { message: MSG, tree: t.sha, parents: [ref.object.sha] })
await api(`/repos/${REPO}/git/refs/heads/${BRANCH}`, { sha: c.sha, force: true }, 'PATCH')
console.log(`\n✓ 已推送 → https://github.com/${REPO}   commit ${c.sha.slice(0, 8)}  共 ${files.length} 个文件`)
