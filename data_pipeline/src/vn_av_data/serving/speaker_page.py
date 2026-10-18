"""Trang duyệt theo speaker: một màn hình, không cuộn dọc, thao tác chính có phím tắt."""

PAGE = r"""<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Duyệt speaker</title>
<style>
:root{--bg:#0f1720;--panel:#16212d;--line:#2a3a4b;--text:#e8eef5;--muted:#93a4b8;--keep:#2e9e5b;
--reject:#c8453b;--warn:#d6a03a;--accent:#4c8dd6}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--text);font:14px/1.4 system-ui,sans-serif;overflow:hidden;
 display:grid;grid-template-rows:46px minmax(0,1fr) 124px}
header{display:flex;align-items:center;gap:14px;padding:0 14px;border-bottom:1px solid var(--line);background:var(--panel)}
header b{font-size:15px}.muted{color:var(--muted)}.sp{flex:1}
button{border:0;border-radius:6px;padding:6px 11px;color:#fff;background:#34495e;cursor:pointer;font-size:13px}
button:hover{filter:brightness(1.15)}button.on{background:var(--accent)}button.keep{background:var(--keep)}
button.reject{background:var(--reject)}button.warn{background:var(--warn);color:#111}
kbd{background:#0006;border-radius:3px;padding:0 4px;font-size:11px}
select,input{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
main{display:grid;grid-template-columns:250px minmax(0,1.35fr) minmax(0,1fr);gap:12px;padding:10px 12px;min-height:0}
section{min-height:0;display:flex;flex-direction:column;gap:8px}
#left{background:var(--panel);border-radius:8px;padding:10px;overflow:hidden}
#photos{display:flex;gap:6px}#photos img{width:72px;height:72px;border-radius:6px;object-fit:cover;background:#000}
#sid{font-size:18px;font-weight:700}.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
#sugg{border:1px solid var(--warn);border-radius:8px;padding:8px;display:none}
#sugg img{width:56px;height:56px;border-radius:5px;object-fit:cover}
#center video{width:100%;flex:1;min-height:0;background:#000;border-radius:8px}
#strip{height:64px;align-self:flex-start;border-radius:4px;background:#000}
#cinfo{color:var(--muted);font-size:13px;min-height:18px}
#tabs{display:flex;gap:4px;flex-wrap:nowrap;overflow:hidden}
#tabs button{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:190px;background:#22313f}
#tabs button.on{background:var(--accent)}
#grid{flex:1;min-height:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));grid-auto-rows:104px;gap:6px;overflow:hidden;align-content:start}
.tile{position:relative;border-radius:6px;background-color:#000;background-size:500% 100%;background-position:50% 0;cursor:pointer;border:2px solid transparent}
.tile.cur{border-color:#fff}.tile.keep{border-color:var(--keep)}.tile.reject{opacity:.35}
.tile .n{position:absolute;left:3px;top:2px;font-size:11px;background:#0009;border-radius:3px;padding:0 3px}
.tile .f{position:absolute;right:3px;top:2px;font-size:12px}
.tile .k{position:absolute;right:3px;bottom:2px;background:var(--keep);border-radius:50%;width:18px;height:18px;text-align:center;font-size:12px}
#pager{display:flex;align-items:center;gap:8px;color:var(--muted)}
footer{border-top:1px solid var(--line);background:var(--panel);display:flex;align-items:center;gap:12px;padding:8px 12px}
#slots{display:flex;gap:6px}
.slot{width:96px;height:96px;border-radius:6px;border:1px dashed var(--line);background-size:500% 100%;background-position:50% 0;cursor:pointer;position:relative}
.slot.full{border:2px solid var(--keep)}.slot .x{position:absolute;right:2px;top:0;font-size:12px;opacity:.8}
#banner{font-size:15px}#banner.ok{color:var(--keep);font-weight:700}
#toast{position:fixed;left:50%;top:54px;transform:translateX(-50%);background:#000d;padding:8px 14px;border-radius:6px;display:none;z-index:9}
#modal{position:fixed;inset:0;background:#0009;display:none;align-items:center;justify-content:center;z-index:8}
#modal>div{background:var(--panel);border-radius:10px;padding:14px;min-width:420px}
.cand{display:flex;gap:8px;align-items:center;padding:6px;border-radius:6px;cursor:pointer}
.cand:hover{background:#22344a}.cand img{width:48px;height:48px;border-radius:5px;object-fit:cover}
</style>
<header><b>Duyệt speaker</b><span id="prog" class="muted"></span><span class="sp"></span>
<label class="muted">Lọc <select id="filter"><option value="pending">chưa duyệt</option><option value="all">tất cả</option>
<option value="suggest">có gợi ý gộp</option></select></label>
<select id="jump" title="Nhảy tới speaker"></select>
<button id="prevS">← speaker <kbd>B</kbd></button><button id="nextS" class="on">Speaker tiếp <kbd>N</kbd></button></header>
<main>
<section id="left"><div id="photos"></div><div id="sid"></div><div id="sstate" class="muted"></div>
<div class="row"><button id="gM">Nam <kbd>M</kbd></button><button id="gF">Nữ <kbd>F</kbd></button></div>
<div id="sugg"><div class="muted">Có thể là cùng một người với:</div><div class="row" id="suggP"></div>
<div class="row"><button class="warn" id="mergeBtn">Gộp vào <kbd>G</kbd></button><button id="dismissBtn">Khác người</button></div></div>
<div class="row"><button class="reject" id="exclude">Loại speaker <kbd>E</kbd></button></div>
<div id="sinfo" class="muted"></div>
<div class="muted" style="margin-top:auto;font-size:12px">←/→ clip · ↑/↓ video · S chọn · X loại · R đổi speaker · Space phát/dừng · PgUp/PgDn trang</div></section>
<section id="center"><video id="video" controls loop></video><img id="strip" alt="">
<div id="cinfo"></div>
<div class="row"><button class="keep" id="bKeep">Chọn <kbd>S</kbd></button><button class="reject" id="bRej">Loại clip <kbd>X</kbd></button>
<button id="bMove">Sai speaker <kbd>R</kbd></button><label class="muted"><input type="checkbox" id="auto" checked> tự phát</label></div></section>
<section id="right"><div id="tabs"></div><div id="grid"></div>
<div id="pager"><button id="pPrev">◀</button><span id="pText"></span><button id="pNext">▶</button></div></section>
</main>
<footer><div><div class="muted">Đã chọn</div><div id="count" style="font-size:22px;font-weight:700"></div></div>
<div id="slots"></div><div id="banner"></div></footer>
<div id="toast"></div><div id="modal"><div><b>Clip này của ai?</b><div id="cands"></div>
<div class="row" style="margin-top:8px"><button id="newSpk">Speaker mới</button><button id="closeM">Huỷ</button></div></div></div>
<script>
const PER=__PER__,TARGET=__TARGET__,$=id=>document.getElementById(id);
let S=null,D=null,cur=-1,tab=0,page=0,clips=[];
async function api(url,body){const r=await fetch(url,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok){const t=await r.text();let m=t;try{m=JSON.parse(t).detail}catch(e){}toast(m);throw new Error(m)}return r.json()}
function toast(m){const t=$('toast');t.textContent=m;t.style.display='block';clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',2600)}
function visible(){const f=$('filter').value;return S.speakers.filter(s=>f==='all'||(f==='pending'?s.review==='pending'&&s.part===null:s.suggestion&&s.part===null))}
async function loadState(){S=await api('/api/state');const c=S.complete;
 $('prog').textContent=`${S.speakers.length} speaker · sẵn cho part kế tiếp (đủ ${PER} clip, duyệt xong): Nam ${c.M}/${TARGET} · Nữ ${c.F}/${TARGET} · còn chờ ${S.speakers.filter(s=>s.review==='pending'&&s.part===null).length}`;
 const j=$('jump');j.innerHTML=S.speakers.map(s=>`<option value="${s.id}">${s.id} · ${s.keep}/${PER} · ${s.review}${s.gender?' · '+s.gender:''}${s.part?' · part '+s.part:''}</option>`).join('');
 if(D)j.value=D.id}
async function openSpeaker(id){D=await api('/api/speaker/'+id);clips=[];D.videos.forEach((v,vi)=>v.clips.forEach(c=>clips.push({...c,vi})));
 tab=0;page=0;cur=clips.length?0:-1;const firstPending=clips.findIndex(c=>c.decision==='pending');if(firstPending>=0){cur=firstPending;tab=clips[cur].vi}
 renderLeft();renderTabs();renderGrid();renderSlots();play();$('jump').value=id}
function renderLeft(){$('photos').innerHTML=D.photos.map(p=>`<img src="${p}">`).join('');$('sid').textContent=D.id;
 $('sstate').textContent=(D.part?`Đã vào part ${D.part} (khoá)`:`Trạng thái: ${D.review}`)+` · ${D.videos.length} video · ${clips.length} clip`;
 $('gM').className=D.gender==='M'?'on':'';$('gF').className=D.gender==='F'?'on':'';
 $('sinfo').textContent='Kênh: '+(D.channels||[]).join(', ');
 const g=D.suggestion;$('sugg').style.display=g?'block':'none';
 if(g)$('suggP').innerHTML=g.photos.map(p=>`<img src="${p}">`).join('')+`<span>${g.speaker_id}<br><span class="muted">mặt ${g.face_sim}${g.gender?' · '+g.gender:''}</span></span>`}
function syncText(v){const s=v.sync||{};return ['ok','few'].includes(s.status)?(s.lag?` · lệch ${s.lag>0?'+':''}${s.lag}f`:''):' · ⚠ không đo được lệch tiếng (lời dẫn đè hình?)'}
function renderTabs(){$('tabs').innerHTML=D.videos.map((v,i)=>{const k=v.clips.filter(c=>c.decision==='keep').length;
 return `<button class="${i===tab?'on':''}" data-i="${i}" title="${(v.title||'').replace(/"/g,'')}${syncText(v)}">${i+1}. ${(v.title||v.video_id).slice(0,22)} (${k}/${v.clips.length})${['ok','few'].includes((v.sync||{}).status)?'':' ⚠'}</button>`}).join('');
 [...$('tabs').children].forEach(b=>b.onclick=()=>{tab=+b.dataset.i;page=0;const i=clips.findIndex(c=>c.vi===tab);if(i>=0){cur=i;play()}renderTabs();renderGrid()})}
function pageSize(){const g=$('grid'),cols=Math.max(1,Math.floor((g.clientWidth+6)/110)),rows=Math.max(1,Math.floor((g.clientHeight+6)/110));return cols*rows}
function tabClips(){return clips.map((c,i)=>({c,i})).filter(x=>x.c.vi===tab)}
function renderGrid(){const list=tabClips(),n=pageSize(),pages=Math.max(1,Math.ceil(list.length/n));page=Math.min(page,pages-1);
 const pos=list.findIndex(x=>x.i===cur);if(pos>=0&&Math.floor(pos/n)!==page&&renderGrid.follow){page=Math.floor(pos/n)}renderGrid.follow=false;
 $('grid').innerHTML=list.slice(page*n,page*n+n).map(({c,i})=>`<div class="tile ${c.decision} ${i===cur?'cur':''}" data-i="${i}" style="background-image:url(/media/strip/${c.clip_id})" title="${c.start_s.toFixed(1)}-${c.end_s.toFixed(1)} s ${c.flags||''}">
 <span class="n">${i+1}</span>${c.flags?'<span class="f">⚠</span>':''}${c.decision==='keep'?'<span class="k">✔</span>':''}</div>`).join('');
 $('pText').textContent=`trang ${page+1}/${pages} · ${list.length} clip`;
 [...$('grid').children].forEach(t=>{t.onclick=()=>{cur=+t.dataset.i;play();renderGrid()};t.ondblclick=()=>{cur=+t.dataset.i;toggle('keep')}})}
function renderSlots(){const keep=clips.filter(c=>c.decision==='keep');$('count').textContent=`${keep.length}/${PER}`;
 $('slots').innerHTML=Array.from({length:PER},(_,k)=>{const c=keep[k];return c?`<div class="slot full" data-id="${c.clip_id}" style="background-image:url(/media/strip/${c.clip_id})" title="Bấm để bỏ chọn"><span class="x">✕</span></div>`:'<div class="slot"></div>'}).join('');
 [...$('slots').children].forEach(s=>{if(s.dataset.id)s.onclick=()=>{cur=clips.findIndex(c=>c.clip_id===s.dataset.id);setDecision('pending')}});
 const left=clips.filter(c=>c.decision==='pending').length;
 $('banner').className=keep.length>=PER?'ok':'';
 $('banner').textContent=keep.length>=PER?`ĐỦ ${PER} clip - bấm N để sang speaker tiếp (bỏ qua ${left} clip còn lại)${D.gender?'':' · nhớ gán giới (M/F)'}`:(clips.length<PER?`Speaker chỉ có ${clips.length} clip (< ${PER})`:'')}
function play(){const c=clips[cur],v=$('video');if(!c){v.removeAttribute('src');$('strip').removeAttribute('src');$('cinfo').textContent='Không có clip';return}
 if(c.vi!==tab){tab=c.vi;renderTabs()}renderGrid.follow=true;
 v.src='/media/clip/'+c.clip_id;if($('auto').checked)v.play().catch(()=>{});$('strip').src='/media/strip/'+c.clip_id;
 const flags=(c.flags||'').split(',').filter(Boolean).map(f=>({jump_suspect:'nghi jump cut tại '+(c.jump||'?')+' s',face_outlier:'mặt lệch so với speaker',voice_outlier:'giọng lệch so với speaker',face_gap:'hụt mặt vài frame',sync_few:'độ lệch tiếng đo trên ít clip, nên xem kỹ',sync_unreliable:'video không đo được độ lệch tiếng (tự loại; chọn lại nếu mặt đúng là người nói)'}[f]||f));
 $('cinfo').textContent=`#${cur+1}/${clips.length} · video ${c.vi+1} · ${c.start_s.toFixed(1)}-${c.end_s.toFixed(1)} s · mặt ổn định ${c.face_consistency?.toFixed(2)} · ${c.decision}`+(flags.length?' · ⚠ '+flags.join('; '):'')+(c.location!=='local'?' · không còn ở máy':'')}
async function setDecision(d){const c=clips[cur];if(!c)return;try{await api('/api/clip/'+c.clip_id,{decision:d})}catch(e){return}
 c.decision=d;renderTabs();renderGrid();renderSlots();loadState()}
function toggle(d){const c=clips[cur];if(!c)return;const next=c.decision===d?'pending':d;setDecision(next).then(()=>{if(next!=='pending'&&cur<clips.length-1){cur++;play();renderGrid()}})}
async function gender(g){if(D.part)return toast('Speaker đã khoá');await api('/api/speaker/'+D.id,{gender:D.gender===g?'':g});D.gender=D.gender===g?'':g;renderLeft();renderSlots();loadState()}
async function nextSpeaker(step=1,finish=true){
 if(finish&&D&&!D.part&&D.review==='pending'){const k=clips.filter(c=>c.decision==='keep').length;
  if(k<PER&&!confirm(`Speaker này mới chọn ${k}/${PER} clip. Vẫn đánh dấu duyệt xong và sang speaker tiếp?`))return;
  if(k>=PER&&!D.gender&&!confirm('Chưa gán giới (M/F). Vẫn sang speaker tiếp?'))return;
  await api('/api/speaker/'+D.id,{review:'done'})}
 await loadState();const list=visible(),ids=S.speakers.map(s=>s.id),here=D?ids.indexOf(D.id):-1;
 const order=step>0?S.speakers.slice(here+1):S.speakers.slice(0,Math.max(0,here)).reverse();
 const target=order.find(s=>list.includes(s));if(!target)return toast(step>0?'Hết speaker theo bộ lọc':'Không còn speaker trước');openSpeaker(target.id)}
async function nearest(){const c=clips[cur];if(!c||D.part)return;const list=await api(`/api/clip/${c.clip_id}/nearest`);
 $('cands').innerHTML=list.map(s=>`<div class="cand" data-id="${s.id}">${s.photos.map(p=>`<img src="${p}">`).join('')}<span>${s.id}<br><span class="muted">mặt ${s.sim}</span></span></div>`).join('')||'<div class="muted">Không có speaker khác</div>';
 [...$('cands').children].forEach(d=>{if(d.dataset.id)d.onclick=()=>moveTo(d.dataset.id)});$('modal').style.display='flex'}
async function moveTo(id){const c=clips[cur];$('modal').style.display='none';const r=await api(`/api/clip/${c.clip_id}/move`,{speaker_id:id});toast(`Đã chuyển clip sang ${r.speaker_id}`);
 const keepCur=Math.min(cur,clips.length-2);await openSpeaker(D.id);cur=Math.max(0,keepCur);play();renderGrid();loadState()}
$('newSpk').onclick=()=>moveTo('new');$('closeM').onclick=()=>$('modal').style.display='none';
$('gM').onclick=()=>gender('M');$('gF').onclick=()=>gender('F');
$('bKeep').onclick=()=>toggle('keep');$('bRej').onclick=()=>toggle('reject');$('bMove').onclick=nearest;
$('nextS').onclick=()=>nextSpeaker(1);$('prevS').onclick=()=>nextSpeaker(-1,false);
$('exclude').onclick=async()=>{if(D.part||!confirm(`Loại hẳn ${D.id} (không vào part nào)?`))return;await api('/api/speaker/'+D.id,{review:'excluded'});nextSpeaker(1,false)};
$('mergeBtn').onclick=async()=>{const g=D.suggestion;if(!g)return;if(!confirm(`Gộp ${D.id} vào ${g.speaker_id}?`))return;const r=await api(`/api/speaker/${D.id}/merge`,{speaker_id:g.speaker_id});toast('Đã gộp vào '+r.into);await loadState();openSpeaker(r.into)};
$('dismissBtn').onclick=async()=>{await api(`/api/speaker/${D.id}/dismiss`,{});D.suggestion=null;renderLeft();loadState()};
$('pPrev').onclick=()=>{page=Math.max(0,page-1);renderGrid()};$('pNext').onclick=()=>{page++;renderGrid()};
$('filter').onchange=()=>nextSpeaker(1,false);$('jump').onchange=e=>openSpeaker(e.target.value);
window.onresize=()=>renderGrid();
document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(e.target.tagName)||$('modal').style.display==='flex'&&e.key!=='Escape')return;const k=e.key;
 if(k==='ArrowRight'){if(cur<clips.length-1){cur++;play();renderGrid()}}else if(k==='ArrowLeft'){if(cur>0){cur--;play();renderGrid()}}
 else if(k==='ArrowDown'||k==='ArrowUp'){const t=Math.max(0,Math.min(D.videos.length-1,tab+(k==='ArrowDown'?1:-1)));const i=clips.findIndex(c=>c.vi===t);if(i>=0){cur=i;play();renderTabs();renderGrid()}}
 else if(k==='s'||k==='S')toggle('keep');else if(k==='x'||k==='X')toggle('reject');else if(k==='r'||k==='R')nearest();
 else if(k==='m'||k==='M')gender('M');else if(k==='f'||k==='F')gender('F');else if(k==='n'||k==='N')nextSpeaker(1);
 else if(k==='b'||k==='B')nextSpeaker(-1,false);else if(k==='e'||k==='E')$('exclude').click();else if(k==='g'||k==='G')$('mergeBtn').click();
 else if(k==='PageDown'){page++;renderGrid()}else if(k==='PageUp'){page=Math.max(0,page-1);renderGrid()}
 else if(k===' '){const v=$('video');v.paused?v.play():v.pause()}else if(k==='Escape'){$('modal').style.display='none'}else return;e.preventDefault()});
(async()=>{await loadState();const first=visible()[0]||S.speakers[0];if(first)openSpeaker(first.id);else toast('Chưa có speaker nào; chạy cut + identify trước')})();
</script></html>"""
