"""Giao diện duyệt dùng chung: clip thật (data_pipeline 04) và mẫu sinh (generation 04).

Một trang HTML tĩnh nhận cấu hình JSON: nguồn dữ liệu, khoá, trường lọc/hiển thị, cặp đối chiếu
(real gốc cạnh fake) và các khoảng cần vạch trên timeline. Không phụ thuộc thư viện ngoài.
"""

import json

PAGE = """<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<style>
:root{--bg:#0f1720;--panel:#16212d;--line:#2a3a4b;--text:#e8eef5;--muted:#93a4b8;--keep:#2e9e5b;
--reject:#c8453b;--uncertain:#c9932b;--fake:#e0584c;--mismatch:#e5a23a;--accent:#4c8dd6}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
#app{display:grid;grid-template-columns:330px 1fr;height:100vh}
aside{border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0;background:var(--panel)}
aside header{padding:12px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 0 6px}#progress{color:var(--muted);font-size:12px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:6px;display:flex}
.bar span{display:block;height:100%}
#filters{padding:8px 12px;display:grid;grid-template-columns:1fr 1fr;gap:6px;border-bottom:1px solid var(--line)}
#filters input,#filters select{width:100%;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px}
#filters input{grid-column:1/-1}
#list{overflow:auto;flex:1;margin:0;padding:0;list-style:none}
#list li{padding:6px 12px;border-bottom:1px solid var(--line);cursor:pointer;display:flex;gap:8px;align-items:center}
#list li.active{background:#22344a}#list li:hover{background:#1c2b3b}
.dot{width:9px;height:9px;border-radius:50%;flex:none;background:var(--muted)}
.keep{background:var(--keep)}.reject{background:var(--reject)}.uncertain{background:var(--uncertain)}
.name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}.tag{color:var(--muted);font-size:12px}
main{display:flex;flex-direction:column;min-width:0;min-height:0;padding:12px 16px;gap:10px;overflow:auto}
#rules{color:var(--muted);margin:0;font-size:13px}
#videos{display:grid;gap:10px;grid-template-columns:1fr}#videos.pair{grid-template-columns:1fr 1fr}
figure{margin:0}figcaption{color:var(--muted);font-size:12px;margin-bottom:3px}
video{width:100%;max-height:52vh;background:#000;border-radius:6px}
#timeline{position:relative;height:22px;background:var(--line);border-radius:4px;cursor:pointer}
#timeline .span{position:absolute;top:0;bottom:0;opacity:.85}#timeline .head{position:absolute;top:-3px;bottom:-3px;width:2px;background:#fff}
#legend{color:var(--muted);font-size:12px;display:flex;gap:14px}#legend i{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}
#controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{border:0;border-radius:7px;padding:9px 14px;color:#fff;background:#34495e;cursor:pointer;font-size:14px}
button.keep{background:var(--keep)}button.reject{background:var(--reject)}button.uncertain{background:var(--uncertain)}
button:focus-visible{outline:2px solid var(--accent)}kbd{background:#0006;border-radius:3px;padding:0 4px;font-size:12px}
label{color:var(--muted);display:flex;gap:4px;align-items:center}
select.speed{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px}
#info{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:13px;margin:0}
#info dt{color:var(--muted)}#info dd{margin:0;word-break:break-all}#status{min-height:18px;color:var(--muted)}
@media (max-width:900px){#app{grid-template-columns:1fr;height:auto}aside{max-height:45vh}#videos.pair{grid-template-columns:1fr}}
</style>
<div id="app"><aside><header><h1>__TITLE__</h1><div id="progress"></div><div class="bar" id="bar"></div></header>
<div id="filters"><input id="search" placeholder="Tìm theo ID/nguồn/speaker…"></div><ul id="list"></ul></aside>
<main><p id="rules"></p><div id="videos"><figure><figcaption id="cap">Mẫu đang duyệt</figcaption><video id="video" controls></video></figure>
<figure id="refbox" hidden><figcaption id="refcap">Real gốc (đối chiếu)</figcaption><video id="ref" controls muted></video></figure></div>
<div id="timeline" title="Bấm để tua"></div><div id="legend"></div>
<div id="controls"><button class="keep" data-d="keep">Giữ <kbd>K</kbd></button><button class="reject" data-d="reject">Loại <kbd>R</kbd></button>
<button class="uncertain" data-d="uncertain">Chưa rõ <kbd>U</kbd></button><button id="prev">← Trước</button><button id="next">Sau →</button>
<button id="todo">Tới mẫu chưa chắc <kbd>N</kbd></button><label><input type="checkbox" id="loop" checked> Lặp <kbd>L</kbd></label>
<label><input type="checkbox" id="auto" checked> Tự phát</label><label>Tốc độ <select class="speed" id="speed"><option>0.5</option><option>0.75</option><option selected>1</option><option>1.25</option></select></label></div>
<div id="status"></div><dl id="info"></dl></main></div>
<script>
const C=__CONFIG__,$=id=>document.getElementById(id);let rows=[],view=[],pos=0;
const key=(r,i)=>C.key==="index"?r.__i:r[C.key];
const pending=r=>C.pending.includes(r.decision);
$('rules').textContent=C.rules;
const COLORS={fake_intervals:'var(--fake)',av_mismatch_intervals:'var(--mismatch)'};
$('legend').innerHTML=Object.entries(C.timeline).map(([k,t])=>`<span><i style="background:${COLORS[k]||'var(--accent)'}"></i>${t}</span>`).join('');
for(const f of C.filters){const s=document.createElement('select');s.id='f_'+f;s.title=f;s.onchange=filter;$('filters').append(s)}
$('search').oninput=filter;
async function load(){const r=await fetch(C.items);rows=(await r.json()).map((x,i)=>({...x,__i:i}));
 for(const f of C.filters){const s=$('f_'+f),vals=[...new Set(rows.map(r=>String(r[f]??'-')))].sort();
  s.innerHTML=`<option value="">${f}: tất cả</option>`+vals.map(v=>`<option>${v}</option>`).join('')}
 filter();const first=view.findIndex(pending);if(first>=0)go(first)}
function filter(){const q=$('search').value.toLowerCase(),cur=view[pos];
 view=rows.filter(r=>C.filters.every(f=>!$('f_'+f).value||String(r[f]??'-')===$('f_'+f).value)&&
  (!q||C.search.some(f=>String(r[f]??'').toLowerCase().includes(q))));
 pos=Math.max(0,view.indexOf(cur));renderList();show()}
function renderList(){const list=$('list');list.innerHTML='';view.forEach((r,i)=>{const li=document.createElement('li');
 li.innerHTML=`<span class="dot ${r.decision}"></span><span class="name">${key(r)}</span><span class="tag">${C.tags.map(t=>r[t]??'').filter(Boolean).join(' · ')}</span>`;
 li.onclick=()=>go(i);if(i===pos)li.className='active';list.append(li)});progress()}
function progress(){const n=rows.length,c={};rows.forEach(r=>c[r.decision]=(c[r.decision]||0)+1);
 $('progress').textContent=`${n} mẫu · giữ ${c.keep||0} · loại ${c.reject||0} · chưa rõ ${c.uncertain||0}`+(c.pending?` · chưa duyệt ${c.pending}`:'')+` · đang lọc ${view.length}`;
 $('bar').innerHTML=['keep','reject','uncertain'].map(d=>`<span class="${d}" style="width:${100*(c[d]||0)/Math.max(n,1)}%"></span>`).join('')}
function go(i){if(!view.length)return;pos=Math.max(0,Math.min(view.length-1,i));
 [...$('list').children].forEach((li,j)=>li.className=j===pos?'active':'');$('list').children[pos]?.scrollIntoView({block:'nearest'});show()}
function show(){const r=view[pos],v=$('video');if(!r){$('status').textContent='Không có mẫu khớp bộ lọc';return}
 v.src=C.media.replace('{key}',encodeURIComponent(key(r)));v.loop=$('loop').checked;v.playbackRate=+$('speed').value;
 if($('auto').checked)v.play().catch(()=>{});
 $('cap').textContent=`${pos+1}/${view.length} · ${key(r)} · ${r.decision}`;
 const ref=C.pair&&r[C.pair.match]!=null?rows.find(x=>x!==r&&x[C.pair.match]===r[C.pair.match]&&Object.entries(C.pair.where).every(([k,w])=>x[k]===w)):null;
 $('refbox').hidden=!ref;$('videos').className=ref?'pair':'';
 if(ref){$('ref').src=C.media.replace('{key}',encodeURIComponent(key(ref)));$('ref').loop=v.loop;$('refcap').textContent='Real gốc: '+key(ref)}
 else $('ref').removeAttribute('src');
 $('info').innerHTML=C.info.filter(f=>r[f]!==undefined&&r[f]!=='').map(f=>`<dt>${f}</dt><dd>${typeof r[f]==='object'?JSON.stringify(r[f]):r[f]}</dd>`).join('');
 drawTimeline()}
function drawTimeline(){const r=view[pos],t=$('timeline'),d=+(r?.duration_s||$('video').duration||0);t.innerHTML='';if(!r||!d)return;
 for(const [k] of Object.entries(C.timeline))for(const [a,b] of (r[k]||[])){const s=document.createElement('div');s.className='span';
  s.style.cssText=`left:${100*a/d}%;width:${100*(b-a)/d}%;background:${COLORS[k]||'var(--accent)'}`;s.title=`${k}: ${a.toFixed(2)}-${b.toFixed(2)} s`;t.append(s)}
 const h=document.createElement('div');h.className='head';h.id='head';t.append(h)}
$('video').ontimeupdate=()=>{const r=view[pos],d=+(r?.duration_s||$('video').duration||0),h=$('head');if(h&&d)h.style.left=(100*$('video').currentTime/d)+'%';
 const ref=$('ref');if(!$('refbox').hidden&&Math.abs(ref.currentTime-$('video').currentTime)>0.15)ref.currentTime=$('video').currentTime};
$('video').onplay=()=>{if(!$('refbox').hidden)$('ref').play().catch(()=>{})};$('video').onpause=()=>$('ref').pause();
$('video').onloadedmetadata=drawTimeline;
$('timeline').onclick=e=>{const b=e.currentTarget.getBoundingClientRect(),v=$('video'),d=v.duration||+(view[pos]?.duration_s||0);v.currentTime=d*(e.clientX-b.left)/b.width};
async function save(decision){const r=view[pos];if(!r)return;
 const res=await fetch(C.save.replace('{key}',encodeURIComponent(key(r))),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision})});
 if(!res.ok){$('status').textContent='Lỗi lưu: '+await res.text();return}
 r.decision=decision;$('status').textContent=`Đã lưu ${key(r)} = ${decision}`;renderList();go(pos+1)}
document.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>save(b.dataset.d));
$('prev').onclick=()=>go(pos-1);$('next').onclick=()=>go(pos+1);
$('todo').onclick=()=>{const i=view.findIndex((r,j)=>j>pos&&pending(r));const k=i>=0?i:view.findIndex(pending);if(k>=0)go(k);else $('status').textContent='Không còn mẫu chưa chắc trong bộ lọc'};
$('loop').onchange=()=>{$('video').loop=$('ref').loop=$('loop').checked};
$('speed').onchange=()=>{$('video').playbackRate=$('ref').playbackRate=+$('speed').value};
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;const k=e.key.toLowerCase();
 if(k==='k')save('keep');else if(k==='r')save('reject');else if(k==='u')save('uncertain');else if(k==='n')$('todo').click();
 else if(k==='l'){$('loop').checked=!$('loop').checked;$('loop').onchange()}
 else if(e.key==='ArrowRight')go(pos+1);else if(e.key==='ArrowLeft')go(pos-1);
 else if(e.key===' '){e.preventDefault();const v=$('video');v.paused?v.play():v.pause()}else return;e.preventDefault()});
load().catch(e=>$('status').textContent=e.message);
</script></html>"""


def review_page(
    title,
    rules,
    *,
    items,
    media,
    save,
    key,
    filters,
    tags,
    info,
    search,
    pending,
    timeline=None,
    pair=None,
):
    """HTML trang duyệt. `media`/`save` chứa `{key}`; `key="index"` dùng vị trí dòng.

    pending: quyết định coi là chưa chắc (phím N nhảy tới). pair: {"match": trường, "where": {...}}
    để chiếu real gốc cạnh mẫu. timeline: {trường khoảng [[a,b],...]: chú thích}.
    """
    config = dict(
        rules=rules,
        items=items,
        media=media,
        save=save,
        key=key,
        filters=filters,
        tags=tags,
        info=info,
        search=search,
        pending=pending,
        timeline=timeline or {},
        pair=pair,
    )
    data = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__TITLE__", title).replace("__CONFIG__", data)
