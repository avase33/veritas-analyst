"""Built-in chat UI (zero-build React via CDN)."""

from __future__ import annotations

CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Veritas — AI Document Analyst</title>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<style>
  :root{--bg:#0b0f1a;--panel:#131a2b;--panel2:#1b2540;--text:#e9eefb;--muted:#93a1bf;
        --accent:#7aa2ff;--good:#37d67a;--bad:#ff6b81;--line:#232f4a;}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       background:radial-gradient(1000px 500px at 70% -10%,#16213e,#0b0f1a);color:var(--text);height:100vh}
  header{padding:16px 24px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
  header h1{font-size:17px;margin:0}
  .badge{font-size:11px;color:var(--muted);border:1px solid var(--line);padding:3px 8px;border-radius:20px}
  .wrap{max-width:860px;margin:0 auto;padding:20px;display:flex;flex-direction:column;height:calc(100vh - 54px)}
  .msgs{flex:1;overflow-y:auto;padding-right:6px}
  .msg{margin-bottom:16px}
  .role{font-size:11px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
  .bubble{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;line-height:1.5}
  .bubble.user{background:var(--panel2);border-color:#2a3a5e}
  .refused{border-color:#4a2632;background:#231016}
  .cites{margin-top:8px;font-size:12px;color:var(--muted)}
  .cite{display:inline-block;background:var(--panel2);border:1px solid var(--line);border-radius:6px;
        padding:2px 7px;margin:2px 4px 0 0}
  .agents{margin-top:6px;font-size:11px;color:var(--muted)}
  .agent{display:inline-block;background:#122;border:1px solid var(--line);color:var(--accent);
         border-radius:6px;padding:1px 6px;margin-right:4px}
  form{display:flex;gap:10px;padding-top:12px;border-top:1px solid var(--line)}
  input{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--text);
        padding:12px 14px;font-size:14px;outline:none}
  button{background:var(--accent);color:#04101f;border:none;font-weight:700;border-radius:10px;
         padding:0 18px;cursor:pointer}
  .hint{font-size:12px;color:var(--muted);margin:6px 0 12px}
  .hint b{color:var(--text)}
</style>
</head>
<body>
<div id="root"></div>
<script>
const {useState,useRef,useEffect}=React;const e=React.createElement;
const SAMPLES=[
  "What was the total revenue growth in Q3, and what are the primary risk factors?",
  "What is the limitation of liability in the services agreement?",
  "Who is the current CEO and what is their salary?"
];
function Msg({m}){
  const cls="bubble"+(m.role==='user'?' user':'')+((m.grounded===false)?' refused':'');
  return e('div',{className:'msg'},
    e('div',{className:'role'},m.role),
    e('div',{className:cls},
      m.text,
      m.citations&&m.citations.length? e('div',{className:'cites'},'Sources: ',
        m.citations.map((c,i)=>e('span',{className:'cite',key:i},
          (c.doc_title? c.doc_title+' · ':'')+'Page '+c.page+(c.section? ' · '+c.section:'')))):null,
      m.agents&&m.agents.length? e('div',{className:'agents'},
        m.agents.map((a,i)=>e('span',{className:'agent',key:i},a))):null
    ));
}
function App(){
  const [msgs,setMsgs]=useState([]);const [q,setQ]=useState('');const [busy,setBusy]=useState(false);
  const end=useRef(null);
  useEffect(()=>{end.current&&end.current.scrollIntoView({behavior:'smooth'});},[msgs]);
  async function send(text){
    const query=(text||q).trim(); if(!query||busy) return;
    setMsgs(m=>[...m,{role:'user',text:query}]); setQ(''); setBusy(true);
    try{
      const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({query})});
      const d=await r.json();
      setMsgs(m=>[...m,{role:'analyst',text:d.answer,grounded:d.grounded,
        citations:d.citations,agents:d.agents_used}]);
    }catch(err){ setMsgs(m=>[...m,{role:'analyst',text:'Error: '+err}]); }
    setBusy(false);
  }
  return e('div',null,
    e('header',null,e('h1',null,'Veritas'),e('span',{className:'badge'},'multi-agent RAG · grounded answers')),
    e('div',{className:'wrap'},
      e('div',{className:'hint'},'Two sample documents are loaded (an annual report and a services agreement). Try: ',
        SAMPLES.map((s,i)=>e('span',{key:i},i>0?' · ':'',e('b',{style:{cursor:'pointer'},onClick:()=>send(s)},'“'+s.slice(0,38)+'…”')))),
      e('div',{className:'msgs'},
        msgs.map((m,i)=>e(Msg,{m,key:i})),
        busy? e('div',{className:'msg'},e('div',{className:'role'},'analyst'),e('div',{className:'bubble'},'Thinking…')):null,
        e('div',{ref:end})),
      e('form',{onSubmit:ev=>{ev.preventDefault();send();}},
        e('input',{value:q,onChange:ev=>setQ(ev.target.value),placeholder:'Ask about the documents…'}),
        e('button',{type:'submit'},'Ask'))
    ));
}
ReactDOM.createRoot(document.getElementById('root')).render(e(App));
</script>
</body>
</html>
"""
