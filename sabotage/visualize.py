"""Render deliberation transcripts into a single self-contained HTML trace viewer.

  python -m sabotage.visualize OUT.html experimental_files/results/main_grid/grok-4.3/games/<file>.json

Only graded games are included (they carry per-round correctness). Usage/prompt metadata is
stripped; public messages, private reflections, votes and grades are embedded as JSON.
"""
from __future__ import annotations

import json


def ensure_judged(g: dict, cache: dict, workers: int = 16) -> None:
    """Fill `cache` with an ExtractedAnswer for every public position and vote in g that has no grade."""
    from concurrent.futures import ThreadPoolExecutor
    from .judge import judge_answer
    grades = g.get("grades") or {}
    jobs = []
    for rd in g["rounds"]:
        r = str(rd["round"])
        for a, text in rd["public_text"].items():
            key = f"{g['tag']}|{g['question_id']}|r{r}|{a}"
            pre = grades.get("public", {}).get(r, {}).get(a)
            if pre:
                cache[key] = pre
            elif key not in cache:
                jobs.append((key, text))
    for a, text in g["votes_text"].items():
        key = f"{g['tag']}|{g['question_id']}|vote|{a}"
        pre = grades.get("votes", {}).get(a)
        if pre:
            cache[key] = pre
        elif key not in cache:
            jobs.append((key, text))
    if jobs:
        with ThreadPoolExecutor(workers) as ex:
            for (key, _), j in zip(jobs, ex.map(lambda j: judge_answer(g["question"], g["correct_answer"], j[1]), jobs)):
                cache[key] = j


def compact(g: dict, cache: dict) -> dict:
    """cache: judged positions keyed like ensure_judged(); values may be bools (old peek cache) or dicts."""
    roles = {str(a["id"]): a["role"] for a in g["agents"]}
    complete = g.get("status") == "complete" and "outcome" in (g.get("grades") or {})
    def look(key):
        v = cache.get(key)
        if v is None:
            return {"answer": "?", "correct": None}
        if isinstance(v, bool):
            return {"answer": "?", "correct": v}
        return {"answer": v.get("extracted_final_answer", "?"), "correct": v.get("correct") == "yes"}
    rounds = []
    for rd in g["rounds"]:
        if not rd["public"]:
            continue
        r = str(rd["round"])
        rounds.append({
            "round": rd["round"],
            "chat": [{"agent": str(e["agent"]), "text": e["text"]} for e in rd.get("chat", [])],
            "agents": {a: {
                "public": rd["public_text"].get(a, ""),
                "reflection": rd["reflection_text"].get(a, ""),
                "length_retries": (rd["public"].get(a) or {}).get("length_retries", 0),
                **look(f"{g['tag']}|{g['question_id']}|r{r}|{a}"),
            } for a in roles},
        })
    votes = {a: {"text": g["votes_text"].get(a, ""), **look(f"{g['tag']}|{g['question_id']}|vote|{a}")} for a in roles}
    honest = [a for a, r in roles.items() if r == "honest"]
    if complete:
        o = g["grades"]["outcome"]
        outcome = {"status": "complete", "win": bool(o["majority_correct_all"]), "honest_votes": o["votes_correct_honest"],
                   "n_honest": o["n_honest"], "honest_r0": o["honest_round0_correct"]}
    else:
        r0 = rounds[0]["agents"] if rounds else {}
        outcome = {"status": "partial", "win": None, "honest_votes": None, "n_honest": len(honest),
                   "honest_r0": sum(1 for a in honest if r0.get(a, {}).get("correct")) if r0 else None,
                   "rounds_done": len(rounds), "votes_done": len(g["votes_text"])}
    dyn = (g.get("grades") or {}).get("dynamics") or {}
    return {
        "id": g["tag"] + "|" + g["question_id"][:8],
        "question_id": g["question_id"][:8],
        "question": g["question"],
        "correct_answer": g["correct_answer"],
        "subject": g.get("raw_subject") or g.get("category") or "",
        "model": g["config"]["honest_model"],
        "deceiver_model": g["config"]["deceiver_model"],
        "N": g["config"]["n_agents"], "k": g["config"]["n_deceivers"], "R": g["config"]["n_delib_rounds"],
        "roles": roles,
        "coordinate": bool(g["config"].get("coordinate")),
        "pregame": {"chat": [{"agent": str(e["agent"]), "text": e["text"]} for e in (g.get("pregame") or {}).get("chat", [])],
                    "plan": (g.get("pregame") or {}).get("reflection_text", {})},
        "rounds": rounds,
        "votes": votes,
        "outcome": outcome,
        "dynamics": {r: {a: {"challenge": v.get("challenge", {}).get("challenges_deceiver"),
                             "repeats": (v.get("reflection") or {}).get("repeats_deceiver_claim"),
                             "suspects": (v.get("reflection") or {}).get("suspects_deceiver")}
                         for a, v in agents.items()} for r, agents in dyn.items()},
        "seconds": g.get("seconds"),
    }


PAGE = r"""<title>Deliberation Traces</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#F4F6F9; --surface:#FFFFFF; --surface-2:#EAEEF4; --line:#D5DBE4; --ink:#1A1F2B; --ink-2:#5B6474; --ink-3:#8A93A3;
  --accent:#3554C4; --accent-soft:#E3E9FB;
  --ok:#1E7F58; --ok-soft:#DDF3E8; --bad:#B8402C; --bad-soft:#F8E1DC; --dec:#9A6B12; --dec-soft:#F7ECD2;
  --shadow:0 1px 2px rgba(20,30,60,.08);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --ground:#0F1218; --surface:#161B24; --surface-2:#1E2532; --line:#2C3543; --ink:#E7EAF0; --ink-2:#A6AFC0; --ink-3:#6F7889;
  --accent:#8BA3F5; --accent-soft:#23304F; --ok:#5CCB93; --ok-soft:#153A2B; --bad:#F08A76; --bad-soft:#43201A; --dec:#E2B45A; --dec-soft:#3D3115;
  --shadow:none;}}
:root[data-theme="dark"]{
  --ground:#0F1218; --surface:#161B24; --surface-2:#1E2532; --line:#2C3543; --ink:#E7EAF0; --ink-2:#A6AFC0; --ink-3:#6F7889;
  --accent:#8BA3F5; --accent-soft:#23304F; --ok:#5CCB93; --ok-soft:#153A2B; --bad:#F08A76; --bad-soft:#43201A; --dec:#E2B45A; --dec-soft:#3D3115;
  --shadow:none;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font:14px/1.5 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif}
.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
.app{display:grid;grid-template-columns:300px 1fr;min-height:100vh}
@media (max-width:860px){.app{grid-template-columns:1fr}}
aside{border-right:1px solid var(--line);background:var(--surface);padding:16px;display:flex;flex-direction:column;gap:12px}
@media (max-width:860px){aside{border-right:0;border-bottom:1px solid var(--line)}}
aside h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.01em}
aside p.lede{margin:0;color:var(--ink-2);font-size:12.5px}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:11.5px;color:var(--ink-2)}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
#counts span::before{display:none} #counts{flex-direction:column;gap:2px}
.legend .l-ok::before{background:var(--ok)} .legend .l-bad::before{background:var(--bad)} .legend .l-dec::before{background:var(--dec)}
.games{display:flex;flex-direction:column;gap:6px;overflow:auto}
.game{border:1px solid var(--line);border-radius:6px;padding:8px 10px;background:var(--surface);cursor:pointer;text-align:left;font:inherit;color:inherit;display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:center}
.game:hover{border-color:var(--accent)} .game[aria-current="true"]{border-color:var(--accent);background:var(--accent-soft)}
.game:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.game .t{font-weight:500} .game .lab{font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;padding:1px 5px;border-radius:3px;background:var(--surface-2);color:var(--ink-2);margin-left:4px} .game .m{color:var(--ink-2);font-size:12px} .game .m .mono{font-size:11.5px}
.pill{font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;padding:2px 7px;border-radius:999px;white-space:nowrap}
.pill.win{background:var(--ok-soft);color:var(--ok)} .pill.loss{background:var(--bad-soft);color:var(--bad)} .pill.part{background:var(--surface-2);color:var(--ink-2)}
table.traj td.unk button{color:var(--ink-3)}
main{padding:20px 24px 48px;display:flex;flex-direction:column;gap:18px;min-width:0}
@media (max-width:860px){main{padding:16px 16px 40px}}
.head{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline}
.head h2{margin:0;font-size:20px;font-weight:600;text-wrap:balance}
.head .meta{color:var(--ink-2);font-size:13px;display:flex;flex-wrap:wrap;gap:4px 14px}
.stat{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink-2)}
.stat b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
details.q{border:1px solid var(--line);border-radius:6px;background:var(--surface)}
details.q summary{cursor:pointer;padding:10px 12px;font-weight:500;list-style:none;display:flex;justify-content:space-between;gap:12px}
details.q summary::-webkit-details-marker{display:none}
details.q summary .ans{color:var(--ok);font-weight:500}
details.q .body{padding:0 12px 12px;white-space:pre-wrap;max-width:76ch;color:var(--ink-2);font-size:13px}
.gridwrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface);box-shadow:var(--shadow)}
table.traj{border-collapse:collapse;width:100%;min-width:640px;font-variant-numeric:tabular-nums}
table.traj th,table.traj td{padding:0;border-bottom:1px solid var(--line)}
table.traj thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);font-weight:600;padding:8px 6px;text-align:center;background:var(--surface-2)}
table.traj thead th:first-child{text-align:left;padding-left:12px}
table.traj th.who{text-align:left;padding:6px 12px;font-weight:500;white-space:nowrap;width:1%}
table.traj th.who .role{display:inline-block;margin-left:8px;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;padding:1px 6px;border-radius:3px;background:var(--surface-2);color:var(--ink-2)}
table.traj tr.dec th.who .role{background:var(--dec-soft);color:var(--dec)}
table.traj tr.dec th.who{box-shadow:inset 3px 0 0 var(--dec)}
table.traj td button{all:unset;display:block;width:100%;height:100%;box-sizing:border-box;padding:7px 6px;text-align:center;cursor:pointer;font:12px/1.3 "IBM Plex Mono",ui-monospace,monospace;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px;margin:0 auto}
table.traj td{text-align:center}
table.traj td.ok button{background:var(--ok-soft);color:var(--ok)} table.traj td.bad button{background:var(--bad-soft);color:var(--bad)}
table.traj td.vote{border-left:2px solid var(--line)}
table.traj td button:hover,table.traj td button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
table.traj td.sel button{box-shadow:inset 0 0 0 2px var(--accent)}
table.traj td .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);margin-left:4px;vertical-align:middle}
.tabs{display:flex;gap:4px;flex-wrap:wrap;align-items:center}
.tabs button{font:inherit;font-size:12.5px;padding:5px 11px;border-radius:999px;border:1px solid var(--line);background:var(--surface);color:var(--ink-2);cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tabs .sp{flex:1}
.tabs label{font-size:12.5px;color:var(--ink-2);display:flex;align-items:center;gap:6px}
.msgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}
@media (max-width:420px){.msgs{grid-template-columns:1fr}}
.msg{border:1px solid var(--line);border-radius:6px;background:var(--surface);display:flex;flex-direction:column;min-width:0}
.msg.dec{border-color:color-mix(in srgb,var(--dec) 55%,var(--line))}
.msg.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.msg header{display:flex;align-items:center;gap:8px;padding:8px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.msg header .name{font-weight:500}
.msg header .role{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;padding:1px 6px;border-radius:3px;background:var(--surface-2);color:var(--ink-2)}
.msg.dec header .role{background:var(--dec-soft);color:var(--dec)}
.msg header .ans{margin-left:auto;font-size:12px;padding:2px 8px;border-radius:4px;max-width:45%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.msg header .ans.ok{background:var(--ok-soft);color:var(--ok)} .msg header .ans.bad{background:var(--bad-soft);color:var(--bad)}
.msg .flags{display:flex;gap:6px;padding:6px 12px 0;flex-wrap:wrap}
.flag{font-size:10.5px;letter-spacing:.04em;text-transform:uppercase;padding:1px 6px;border-radius:3px;background:var(--surface-2);color:var(--ink-2)}
.flag.hot{background:var(--bad-soft);color:var(--bad)} .flag.good{background:var(--ok-soft);color:var(--ok)}
.msg .text{padding:10px 12px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;line-height:1.55;max-height:420px;overflow:auto}
.msg .text.refl{border-top:1px dashed var(--line);background:var(--surface-2);color:var(--ink-2)}
.chat{border:1px solid var(--dec);border-radius:6px;background:var(--dec-soft);padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.chat h3{margin:0;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--dec)}
.chat .bubble{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:8px 10px;max-width:76ch}
.chat .bubble .who{font-weight:600;font-size:12.5px;color:var(--dec);margin-right:8px}
.chat .bubble .txt{white-space:pre-wrap;font-size:13px}
.chat .plan{border-top:1px dashed var(--line);padding-top:8px}
.msg .text .lab{display:block;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);margin-bottom:4px;font-family:"IBM Plex Sans",system-ui,sans-serif}
.msg .text b{font-weight:600}
.empty{color:var(--ink-3);font-style:italic}
@media (prefers-reduced-motion:no-preference){.game,.tabs button{transition:background .12s,border-color .12s}}
</style>
<div class="app">
  <aside>
    <h1 id="ttl">Deliberation Traces</h1>
    <p class="lede">Colleague-by-round positions from the sabotage trial, every game including ones halted mid-run. Click a cell to jump to that message; deceiver rows are marked. Win = correct answer holds a strict majority of all final votes.</p>
    <div class="legend" id="counts"></div>
    <div class="legend"><span class="l-ok">argues correct</span><span class="l-bad">argues wrong</span><span class="l-dec">deceiver</span></div>
    <div class="games" id="games" role="listbox" aria-label="Games"></div>
  </aside>
  <main id="main"></main>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const PAYLOAD = JSON.parse(document.getElementById('data').textContent);
const GAMES = PAYLOAD.games;           // inline: full games; split: manifest entries (+ .file)
const LOADED = {};
async function getGame(i){
  if (PAYLOAD.mode === 'inline') return GAMES[i];
  if (LOADED[i]) return LOADED[i];
  const m = GAMES[i];
  const r = await fetch(m.file); if (!r.ok) throw new Error('fetch failed ' + r.status);
  const data = await r.json();
  const list = Array.isArray(data) ? data : [data];
  GAMES.forEach((g, j) => { if (g.file === m.file) LOADED[j] = list[g.idx ?? 0]; });
  return LOADED[i];
}
const $ = (s, el=document) => el.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
const short = (m) => m.replace('gemini-3.8-flash','Gemini 3.8 Flash').replace('grok-4.3','Grok 4.3').replace('together/deepseek-ai/DeepSeek-V4.1-Flash','DeepSeek V4.1 Flash').replace('together/meta-models/Muse-Glimmer-30B','Muse Glimmer 30B').replace('gpt-5.6-sol','GPT-5.6 Sol').replace('gpt-5.6-luna','GPT-5.6 Luna').replace('claude-opus-5','Claude Opus 5').replace('grok-4.6','Grok 4.6');
let state = { g: 0, round: 0, agent: null, showRefl: true };
try { const s = JSON.parse(localStorage.getItem('traces-state')||'null'); if (s && GAMES[s.g]) state = {...state, ...s}; } catch(e) {}
function save(){ try { localStorage.setItem('traces-state', JSON.stringify(state)); } catch(e) {} }

function renderList(){
  const box = $('#games'); box.innerHTML = '';
  GAMES.forEach((g, i) => {
    const b = document.createElement('button'); b.className = 'game'; b.setAttribute('role','option');
    b.setAttribute('aria-current', i === state.g ? 'true' : 'false');
    const o = g.outcome; const pill = o.status==='partial' ? `<span class="pill part">r${o.rounds_done-1}${o.votes_done?'+v':''}</span>` : `<span class="pill ${o.win?'win':'loss'}">${o.win?'win':'loss'}</span>`;
    const tail = o.status==='partial' ? `partial · ${o.rounds_done} of ${g.R+1} rounds` : `honest votes ${o.honest_votes}/${o.n_honest}`;
    b.innerHTML = `<span class="t">${esc(g.subject)} <span class="mono">${esc(g.question_id)}</span>${g.label ? ` <span class="lab">${esc(g.label)}</span>` : ''}</span>${pill}
      <span class="m">${esc(short(g.model))} · ${g.N - g.k} honest + ${g.k} deceiver${g.k===1?'':'s'} · ${tail}</span>`;
    b.onclick = () => { state.g = i; state.round = 0; state.agent = null; save(); render(); };
    box.appendChild(b);
  });
}

async function renderMain(){
  const main = $('#main'); let g;
  try { g = await getGame(state.g); } catch(e) { main.innerHTML = `<p class="empty">Could not load this game (${esc(e.message)}).</p>`; return; }
  const ids = Object.keys(g.roles); const nR = g.rounds.length;
  const hasVotes = Object.values(g.votes).some(v => v.text);
  const cols = [...Array(nR).keys()].map(r => 'r'+r).concat(hasVotes ? ['vote'] : []);
  if (state.round === 'vote' && !hasVotes) state.round = 0;
  if (typeof state.round === 'number' && state.round >= nR) state.round = nR - 1;
  const isVote = state.round === 'vote';
  const partial = g.outcome.status === 'partial';
  let h = `<div class="head"><h2>${esc(g.subject)} <span class="mono" style="font-weight:400;color:var(--ink-3);font-size:14px">${esc(g.question_id)}</span></h2>
    <div class="meta"><span>${esc(short(g.model))}</span><span>${g.N - g.k} honest + ${g.k} deceiver${g.k===1?'':'s'}</span><span>${g.R} deliberation rounds</span>${g.seconds?`<span>${Math.round(g.seconds/60)} min</span>`:''}</div></div>`;
  const oc = g.outcome;
  h += `<div class="stat"><span>Outcome <b>${partial ? `<span class="pill part">partial · halted after round ${nR-1}${hasVotes?' + some votes':''}</span>` : `<span class="pill ${oc.win?'win':'loss'}">${oc.win?'win':'loss'}</span>`}</b></span>
    <span>Honest correct at round 0 <b>${oc.honest_r0 ?? '–'}/${oc.n_honest}</b></span>${partial ? '' : `<span>Honest votes correct <b>${oc.honest_votes}/${oc.n_honest}</b></span>`}
    <span>Deceivers arguing the truth <b>${countDecCorrect(g)}</b></span></div>`;
  h += `<details class="q"><summary><span>Question</span><span class="ans">correct: <span class="mono">${esc(g.correct_answer)}</span></span></summary><div class="body">${esc(g.question)}</div></details>`;
  // trajectory grid
  h += `<div class="gridwrap"><table class="traj"><thead><tr><th>Colleague</th>${cols.map(c=>`<th>${c==='vote'?'vote':'round '+c.slice(1)}</th>`).join('')}</tr></thead><tbody>`;
  for (const a of ids){
    const dec = g.roles[a]==='deceiver';
    h += `<tr class="${dec?'dec':''}"><th class="who">Colleague ${a}<span class="role">${dec?'deceiver':'honest'}</span></th>`;
    cols.forEach((c, ci) => {
      const cell = c==='vote' ? g.votes[a] : g.rounds[ci].agents[a];
      const sel = (String(state.round)===(c==='vote'?'vote':String(ci))) && state.agent===a;
      const dyn = c!=='vote' ? (g.dynamics[String(ci)]||{})[a] : null;
      const hot = dyn && (dyn.repeats==='yes');
      const cls = cell.correct===true ? 'ok' : cell.correct===false ? 'bad' : 'unk';
      if (!cell.public && c!=='vote' && !cell.answer) { h += `<td class="unk"><button disabled>–</button></td>`; return; }
      h += `<td class="${cls}${c==='vote'?' vote':''}${sel?' sel':''}"><button title="${esc(cell.answer)}" data-r="${c==='vote'?'vote':ci}" data-a="${a}">${esc(cell.answer)}${hot?'<span class="dot" title="reflection repeats a deceiver claim"></span>':''}</button></td>`;
    });
    h += '</tr>';
  }
  h += '</tbody></table></div>';
  // round tabs
  h += `<div class="tabs" role="tablist">${cols.map((c,ci)=>`<button role="tab" data-r="${c==='vote'?'vote':ci}" aria-selected="${String(state.round)===(c==='vote'?'vote':String(ci))}">${c==='vote'?'Final votes':(ci===0?'Round 0 · independent':'Round '+ci)}</button>`).join('')}
    <span class="sp"></span><label><input type="checkbox" id="refl" ${state.showRefl?'checked':''}> show private reflections</label></div>`;
  // private deceiver chat (coordinate mode)
  const chatBlock = (title, entries) => entries.length ? `<section class="chat"><h3>${title}</h3>${entries.map(e=>`<div class="bubble"><span class="who">Colleague ${e.agent}</span><span class="role" style="font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--dec)">deceiver</span><div class="txt">${fmt(e.text)}</div></div>`).join('')}</section>` : '';
  if (!isVote && g.coordinate){
    if (state.round === 0 && g.pregame){
      h += chatBlock('Pre-game private deceiver chat (before round 0)', g.pregame.chat);
      const plans = Object.entries(g.pregame.plan||{});
      if (plans.length && state.showRefl) h += `<section class="chat"><h3>Pre-game plan reflections (private to each deceiver)</h3>${plans.map(([a,t])=>`<div class="bubble"><span class="who">Colleague ${a}</span><div class="txt">${fmt(t)}</div></div>`).join('')}</section>`;
    }
    const rc = g.rounds[state.round].chat || [];
    if (rc.length) h += chatBlock(`Private deceiver chat after round ${state.round} board (seen only by deceivers, before their reflections)`, rc);
  }
  // messages
  h += `<div class="msgs">`;
  for (const a of ids){
    const dec = g.roles[a]==='deceiver';
    const cell = isVote ? g.votes[a] : g.rounds[state.round].agents[a];
    const dyn = !isVote ? (g.dynamics[String(state.round)]||{})[a] : null;
    let flags = '';
    if (dyn && !dec){
      if (dyn.challenge==='yes') flags += `<span class="flag good">challenges deceiver</span>`;
      if (dyn.repeats==='yes') flags += `<span class="flag hot">repeats deceiver claim</span>`;
      if (dyn.suspects==='yes') flags += `<span class="flag">suspects deceiver</span>`;
    }
    if (!isVote && cell.length_retries) flags += `<span class="flag">budget retried ×${cell.length_retries}</span>`;
    const acls = cell.correct===true ? 'ok' : cell.correct===false ? 'bad' : '';
    h += `<article class="msg ${dec?'dec':''} ${state.agent===a?'sel':''}" id="msg-${a}"><header><span class="name">Colleague ${a}</span><span class="role">${dec?'deceiver':'honest'}</span><span class="ans mono ${acls}" title="${esc(cell.answer)}">${esc(cell.answer)}</span></header>`;
    if (flags) h += `<div class="flags">${flags}</div>`;
    if (isVote) h += `<div class="text mono">${fmt(cell.text)||'<span class="empty">empty</span>'}</div>`;
    else {
      h += `<div class="text"><span class="lab">public response</span>${fmt(cell.public)||'<span class="empty">empty response</span>'}</div>`;
      if (state.showRefl) h += `<div class="text refl"><span class="lab">private reflection (after this round)</span>${fmt(cell.reflection)||'<span class="empty">empty</span>'}</div>`;
    }
    h += `</article>`;
  }
  h += `</div>`;
  main.innerHTML = h;
  main.querySelectorAll('table.traj button').forEach(b => b.onclick = () => { state.round = b.dataset.r==='vote'?'vote':+b.dataset.r; state.agent = b.dataset.a; save(); render(); const el = $('#msg-'+b.dataset.a); if (el) el.scrollIntoView({block:'nearest'}); });
  main.querySelectorAll('.tabs [role=tab]').forEach(b => b.onclick = () => { state.round = b.dataset.r==='vote'?'vote':+b.dataset.r; state.agent = null; save(); render(); });
  $('#refl').onchange = e => { state.showRefl = e.target.checked; save(); render(); };
}
function countDecCorrect(g){
  const dec = Object.keys(g.roles).filter(a=>g.roles[a]==='deceiver'); if (!dec.length) return '–';
  let n=0, t=0; for (const rd of g.rounds) for (const a of dec){ if (!rd.agents[a].public) continue; t++; if (rd.agents[a].correct) n++; }
  return `${n}/${t} msgs`;
}
function renderCounts(){
  const by = {}; for (const g of GAMES){ const k = short(g.model); by[k] = by[k] || {n:0, done:0}; by[k].n++; if (g.outcome.status!=='partial') by[k].done++; }
  $('#counts').innerHTML = Object.entries(by).map(([m, c]) => `<span style="padding-left:0">${esc(m)}: ${c.done}/${c.n} complete</span>`).join('');
}
function render(){ renderList(); renderCounts(); renderMain(); }
render();
</script>
"""


def load_cache(path):
    import os
    return json.load(open(path)) if path and os.path.exists(path) else {}


def slug(c: dict) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", c["id"])


def main(out: str, files: list[str], cache_path: str | None = None, split: bool = False, bucket_csv: str | None = None) -> None:
    """out: an .html path (single file, data embedded) or, with split=True, a directory that gets
    index.html (manifest embedded) + games/<id>.json fetched on demand."""
    import os
    cache = load_cache(cache_path)
    games = []
    for f in files:
        g = json.load(open(f))
        if not g["rounds"] or not g["rounds"][0]["public"]:
            print("skipping (no calls yet):", f); continue
        ensure_judged(g, cache)
        games.append(compact(g, cache))
        if cache_path:
            json.dump(cache, open(cache_path, "w"))
    labels = {}
    if bucket_csv:
        import csv
        labels = {r["id"][:8]: f"probe {r['correct_of_k']}/4" for r in csv.DictReader(open(bucket_csv))}
    for c in games:
        c["label"] = labels.get(c["question_id"], "")
    order = {"gemini-3.8-flash": 0, "grok-4.3": 1,
             "together/deepseek-ai/DeepSeek-V4.1-Flash": 2,
             "together/meta-models/Muse-Glimmer-30B": 3,
             "gpt-5.6-sol": 4, "gpt-5.6-luna": 5,
             "claude-opus-5": 6, "grok-4.6": 7}
    games.sort(key=lambda c: (order.get(c["model"], 9), labels.get(c["question_id"], ""), c["question_id"], c["k"]))
    if not split:
        data = json.dumps({"mode": "inline", "games": games}, ensure_ascii=False).replace("</", "<\\/")
        open(out, "w").write(PAGE.replace("__DATA__", data))
        print(f"wrote {out}: {len(games)} games, {len(data)/1e6:.1f} MB embedded")
        return
    os.makedirs(os.path.join(out, "games"), exist_ok=True)
    manifest = []
    bundles: dict[str, list] = {}
    for c in games:                       # one file per (model, question): its games in a list
        key = f"{c['model']}_{c['question_id']}"
        bundles.setdefault(key, []).append(c)
    for key, cs in bundles.items():
        fn = f"games/{key}.json"
        json.dump(cs, open(os.path.join(out, fn), "w"), ensure_ascii=False)
        for i, c in enumerate(cs):
            manifest.append({k: c[k] for k in ("id", "question_id", "subject", "model", "N", "k", "R", "outcome", "label")} | {"file": fn, "idx": i})
    data = json.dumps({"mode": "split", "games": manifest}, ensure_ascii=False).replace("</", "<\\/")
    open(os.path.join(out, "index.html"), "w").write(PAGE.replace("__DATA__", data))
    print(f"wrote {out}/index.html + {len(manifest)} game files ({sum(os.path.getsize(os.path.join(out, m['file'])) for m in manifest)/1e6:.1f} MB)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("files", nargs="+")
    ap.add_argument("--cache", default=None, help="optional judge-cache JSON")
    ap.add_argument("--split", action="store_true", help="write a directory: index.html + games/*.json")
    ap.add_argument("--bucket-csv", default=None, help="selection.csv from sabotage.probe, to label questions by probe bucket")
    a = ap.parse_args()
    main(a.out, a.files, a.cache, a.split, a.bucket_csv)
