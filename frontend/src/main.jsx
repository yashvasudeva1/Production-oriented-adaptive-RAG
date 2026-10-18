import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, ArrowDownRight, ArrowLeft, ArrowRight, ArrowUpRight, BookOpen,
  Check, CheckCircle2, ChevronDown, ChevronRight, CircleHelp, Clock3, Copy,
  Database, FileText, Filter, Gauge, Layers3, Library, LoaderCircle,
  Menu, MessageSquareText, MoreHorizontal, Network, PanelLeftClose, Plus,
  RefreshCw, Search, Send, Settings2, ShieldCheck, SlidersHorizontal,
  Sparkles, X, Zap, CircleAlert, BarChart3, Command, Upload, ExternalLink,
  GitBranch, HardDrive, Server, SearchCheck, Workflow, FileSearch
} from 'lucide-react';

function GithubIcon({ size = 16, className = '' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from 'recharts';
import './styles.css';

const API_BASE_DEFAULT = (import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
const REPO_URL = 'https://github.com/yashvasudeva1/Production_RAG';
const DEMO_DOCS = [
  {id:'doc-001',name:'Attention Is All You Need.pdf',type:'PDF',chunks:42,status:'INDEXED',size:'2.4 MB',updated:'Today, 10:42'},
  {id:'doc-002',name:'NITI Aayog Internship Guidelines.pdf',type:'PDF',chunks:28,status:'INDEXED',size:'86 KB',updated:'Today, 09:18'},
  {id:'doc-003',name:'Yash Vasudeva Resume.pdf',type:'PDF',chunks:12,status:'INDEXED',size:'110 KB',updated:'Yesterday'},
  {id:'doc-004',name:'Nexus Architecture Specification.md',type:'MD',chunks:31,status:'INDEXED',size:'112 KB',updated:'Oct 15'},
];
const DEMO_QUESTIONS = [
  'What is the Transformer architecture and self-attention mechanism?',
  'What is the maximum internship duration in NITI Aayog?',
  'What are the technical skills and projects of Yash Vasudeva?',
  'How does the system decide when to rerank?',
];
const BENCH = {
  cases:110, mrr:0.789, recall:0.8556, precision:0.575, hitRate:0.844, ndcg:0.7723,
  abstention:97.3, citation:96.7, fast:3.6, balanced:55.5, deep:40.9, escalation:15.5,
  rerank:87.3, multiQuery:28.2, parent:56.4, abstainRate:20.9,
  latency:{min:2.0,p50:863.9,p90:1423.8,p95:1504.4,p99:1851.0,mean:801.8,max:3179.5},
};
const BENCH_SERIES = [
  {label:'BM25 only',recall:86.17,p50:0.4,p95:0.7},
  {label:'Hybrid + filter',recall:88.30,p50:3.8,p95:8.0},
  {label:'Adaptive · no rerank',recall:88.30,p50:4.5,p95:6.7},
  {label:'Full system',recall:82.98,p50:6.0,p95:9.7},
  {label:'Warm cache',recall:82.98,p50:2.6,p95:4.7},
];
const latencyTrend = [
  {t:'00:00',retrieval:8.2,total:822},{t:'04:00',retrieval:7.1,total:762},{t:'08:00',retrieval:9.6,total:930},
  {t:'12:00',retrieval:8.5,total:854},{t:'16:00',retrieval:6.9,total:731},{t:'20:00',retrieval:7.8,total:799},{t:'Now',retrieval:6.4,total:784},
];

const fmt = (v, digits=1) => Number.isFinite(Number(v)) ? Number(v).toFixed(digits) : '-';
const pct = (v, digits=1) => `${fmt(Number(v)*100, digits)}%`;
const safeText = (v, fallback='-') => v === undefined || v === null || v === '' ? fallback : String(v);

function IconButton({children,label,onClick,className='',title}) {
  return <button className={`icon-button ${className}`} type="button" aria-label={label} title={title || label} onClick={onClick}>{children}</button>;
}
function Tag({children,tone='neutral',dot=false}) {return <span className={`tag tag-${tone}`}>{dot&&<i/>}{children}</span>}
function SectionHeading({eyebrow,title,description,action}) {return <div className="section-heading"><div>{eyebrow&&<div className="eyebrow">{eyebrow}</div>}<h2>{title}</h2>{description&&<p>{description}</p>}</div>{action}</div>}
function MetricCard({label,value,unit,detail,icon:Icon,accent='blue',footnote}) {return <article className="metric-card"><div className="metric-top"><span className="metric-label">{label}</span><span className={`metric-icon ${accent}`}><Icon size={16}/></span></div><div className="metric-value">{value}<small>{unit}</small></div>{detail&&<div className="metric-detail">{detail}</div>}{footnote&&<div className="metric-footnote">{footnote}</div>}</article>}

function App() {
  const [page,setPage] = useState('workspace');
  const [mode,setMode] = useState('adaptive');
  const [query,setQuery] = useState('');
  const [busy,setBusy] = useState(false);
  const [answer,setAnswer] = useState(null);
  const [error,setError] = useState('');
  const [health,setHealth] = useState(null);
  const [metrics,setMetrics] = useState(null);
  const [docs,setDocs] = useState(DEMO_DOCS);
  const [docsLoading,setDocsLoading] = useState(false);
  const [showLibrary,setShowLibrary] = useState(false);
  const [showSettings,setShowSettings] = useState(false);
  const [apiBase,setApiBase] = useState(API_BASE_DEFAULT);
  const [apiDraft,setApiDraft] = useState(API_BASE_DEFAULT);
  const [connectionChecked,setConnectionChecked] = useState(false);
  const [searchTerm,setSearchTerm] = useState('');
  const [searchStrategy,setSearchStrategy] = useState('hybrid');
  const [searchResults,setSearchResults] = useState(null);
  const [searchBusy,setSearchBusy] = useState(false);
  const [selectedDoc,setSelectedDoc] = useState(null);
  const [uploadBusy,setUploadBusy] = useState(false);
  const [notice,setNotice] = useState('');
  const [mobileNav,setMobileNav] = useState(false);
  const queryRef = useRef(null);
  const fileRef = useRef(null);
  const runCounter = useRef(0);

  const base = apiBase.replace(/\/$/, '');
  const isLive = Boolean(connectionChecked && health?.status === 'healthy');
  const totalDocs = metrics?.documents_total ?? docs.filter(d=>d.status==='INDEXED').length;
  const totalChunks = health?.chunk_count ?? docs.reduce((a,d)=>a+(d.chunks||0),0);

  async function request(path,options={}) {
    const response = await fetch(`${base}${path}`, options);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try { const body = await response.json(); detail = body.detail || detail; } catch {}
      throw new Error(detail);
    }
    return response;
  }

  async function refreshOverview(quiet=false) {
    try {
      const [h,m] = await Promise.all([request('/health').then(r=>r.json()),request('/metrics').then(r=>r.json())]);
      setHealth(h); setMetrics(m); setConnectionChecked(true);
      try {
        const d = await request('/documents').then(r=>r.json());
        const list = Array.isArray(d) ? d : (d.documents || d.items || []);
        if (list.length) setDocs(list.map((x,i)=>({id:x.document_id||x.id||`doc-${i}`,name:x.filename||x.name||'Untitled document',type:(x.filename||x.name||'FILE').split('.').pop().toUpperCase(),chunks:x.chunk_count||x.chunks_count||0,status:x.status||'INDEXED',size:x.size||'-',updated:x.updated_at||'-'})));
      } catch { /* Document endpoint paths can vary by deployed version; health/metrics remain useful. */ }
      if (!quiet) setNotice('Connected to the live RAG API.');
      return true;
    } catch (e) {
      setConnectionChecked(true); setHealth(null); setMetrics(null);
      if (!quiet) setNotice(`API unavailable: ${e.message}. Showing workspace preview data.`);
      return false;
    }
  }
  useEffect(()=>{ refreshOverview(true); const id=setInterval(()=>refreshOverview(true),30000); return ()=>clearInterval(id); },[apiBase]);
  useEffect(()=>{if(notice){const t=setTimeout(()=>setNotice(''),4500);return()=>clearTimeout(t)}},[notice]);
  useEffect(()=>{const onKey=e=>{if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();setPage('workspace');queryRef.current?.focus()}if(e.key==='Escape'){setShowLibrary(false);setShowSettings(false);setMobileNav(false)}};window.addEventListener('keydown',onKey);return()=>window.removeEventListener('keydown',onKey)},[]);

  async function runQuery(event) {
    event?.preventDefault();
    const prompt=query.trim(); if(!prompt||busy)return;
    setBusy(true);setError('');setAnswer(null);setNotice(''); const runId=++runCounter.current;
    if(!base){
      setError('Connect an API URL in Settings to run a live query. The workspace preview is available without a backend.');setBusy(false);return;
    }
    try {
      // The backend streams structured events before and during answer generation.
      const response=await request('/query/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:prompt,mode})});
      const reader=response.body.getReader();const decoder=new TextDecoder();let buffer='';let current={query:prompt,answer:'',sources:[],confidence:null,latency_ms:null,retrieval:null,citation_validation:null,abstained:false};setAnswer(current);
      while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const events=buffer.split('\n\n');buffer=events.pop()||'';
        for(const eventChunk of events){const line=eventChunk.split('\n').find(l=>l.startsWith('data: '));if(!line)continue;let data;try{data=JSON.parse(line.slice(6))}catch{continue}
          if(data.event==='retrieval'){current={...current,confidence:data.confidence,execution_mode:data.execution_mode,query_type:data.query_type,retrieval:{...(current.retrieval||{}),latency_retrieval_ms:data.latency_retrieval_ms}};}
          else if(data.event==='delta'){current={...current,answer:current.answer+(data.text||'')};}
          else if(data.event==='done'){current={...current,...data};}
          else if(data.event==='error'){throw new Error(data.detail||'Query failed');}
          if(runId===runCounter.current)setAnswer({...current});
        }
      }
      await refreshOverview(true);
    } catch(e){if(runId===runCounter.current)setError(`Could not complete query: ${e.message}. Check the API address and that the backend is running.`)}
    finally{if(runId===runCounter.current)setBusy(false)}
  }
  async function runSearch(e) {
    e?.preventDefault();if(!searchTerm.trim())return;
    if(!base){setNotice('Set the API URL in Settings to run document search.');return;}
    setSearchBusy(true);setSearchResults(null);
    try{const result=await request('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:searchTerm,strategy:searchStrategy,top_k:8})}).then(r=>r.json());setSearchResults(Array.isArray(result)?result:(result.results||result.items||[]));}
    catch(err){setNotice(`Search failed: ${err.message}`)}finally{setSearchBusy(false)}
  }
  async function uploadFiles(files) {
    if(!files?.length)return;
    if(!base){setNotice('Connect the API before uploading documents.');return;}
    setUploadBusy(true);
    try{for(const file of files){const body=new FormData();body.append('file',file);await request('/documents/upload',{method:'POST',body});}setNotice(`${files.length} file${files.length===1?'':'s'} uploaded.`);await refreshOverview(true)}
    catch(e){setNotice(`Upload failed: ${e.message}. Confirm the endpoint in your API version.`)}finally{setUploadBusy(false)}
  }
  async function saveApiSettings(e){e.preventDefault();setApiBase(apiDraft.trim().replace(/\/$/,''));setConnectionChecked(false);setShowSettings(false);setNotice('Testing API connection…');setTimeout(()=>refreshOverview(),100);}

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav?'sidebar-open':''}`}>
      <div className="brand-row"><div className="brand-mark"><Network size={19} strokeWidth={1.8}/></div><div className="brand-copy"><b>Nexus</b><span>Adaptive RAG</span></div><IconButton label="Close navigation" className="mobile-only" onClick={()=>setMobileNav(false)}><X size={17}/></IconButton></div>
      <div className="workspace-switcher"><div className="workspace-avatar">NX</div><div className="workspace-info"><b>Personal workspace</b><span>Research environment</span></div><ChevronDown size={15}/></div>
      <div className="nav-section-label">WORKSPACE</div>
      <nav className="main-nav" aria-label="Main navigation">
        <button className={page==='workspace'?'nav-item active':'nav-item'} onClick={()=>{setPage('workspace');setMobileNav(false)}}><MessageSquareText size={17}/><span>Query workspace</span><kbd>1</kbd></button>
        <button className={page==='search'?'nav-item active':'nav-item'} onClick={()=>{setPage('search');setMobileNav(false)}}><Search size={17}/><span>Search explorer</span></button>
        <button className={page==='library'?'nav-item active':'nav-item'} onClick={()=>{setPage('library');setMobileNav(false)}}><Library size={17}/><span>Document library</span><span className="nav-count">{totalDocs}</span></button>
        <button className={page==='metrics'?'nav-item active':'nav-item'} onClick={()=>{setPage('metrics');setMobileNav(false)}}><Activity size={17}/><span>Pipeline metrics</span><span className="nav-pill">NEW</span></button>
      </nav>
      <div className="nav-section-label nav-section-spaced">SYSTEM</div>
      <nav className="main-nav">
        <button className="nav-item" onClick={()=>setShowSettings(true)}><Settings2 size={17}/><span>API & settings</span></button>
        <a className="nav-item nav-link" href={REPO_URL} target="_blank" rel="noreferrer"><GithubIcon size={17}/><span>View source code</span><ExternalLink size={13} className="nav-external"/></a>
      </nav>
      <div className="sidebar-bottom"><div className="connection-card"><div className="connection-heading"><span className={`status-dot ${isLive?'online':'offline'}`}/><b>{isLive?'API connected':'Preview mode'}</b><button className="plain-icon" aria-label="Refresh connection" onClick={()=>refreshOverview()}><RefreshCw size={13}/></button></div><p>{isLive?`${safeText(health?.chunk_count,'-')} indexed chunks available`:'Connect your FastAPI backend to run live queries.'}</p><button className="connection-action" onClick={()=>setShowSettings(true)}>{isLive?'Configure connection':'Connect API'}<ArrowUpRight size={14}/></button></div><div className="profile-row"><div className="profile-avatar">Y</div><div className="profile-copy"><b>Yash Vasudeva</b><span>Developer</span></div><MoreHorizontal size={18} className="muted-icon"/></div></div>
    </aside>
    {mobileNav&&<button className="mobile-scrim" aria-label="Close navigation" onClick={()=>setMobileNav(false)}/>}
    <main className="main-panel">
      <header className="topbar"><div className="topbar-left"><IconButton label="Open navigation" className="mobile-only menu-trigger" onClick={()=>setMobileNav(true)}><Menu size={19}/></IconButton><div className="breadcrumb"><span>Nexus</span><ChevronRight size={14}/><b>{({workspace:'Workspace',search:'Search explorer',library:'Document library',metrics:'Pipeline metrics'})[page]}</b></div></div><div className="topbar-right"><div className="live-status"><span className={`status-dot ${isLive?'online':'offline'}`}/><span>{isLive?'Live API':connectionChecked?'Not connected':'Checking API'}</span></div><span className="topbar-divider"/><button className="topbar-help" onClick={()=>setShowSettings(true)}><CircleHelp size={16}/><span>Help & setup</span></button><IconButton label="Settings" onClick={()=>setShowSettings(true)}><Settings2 size={17}/></IconButton></div></header>
      <div className="page-content">
        {page==='workspace'&&<WorkspacePage {...{query,setQuery,mode,setMode,busy,answer,error,runQuery,queryRef,docs,totalDocs,totalChunks,isLive,setPage,setShowLibrary}}/>}
        {page==='metrics'&&<MetricsPage {...{isLive,health,metrics,refreshOverview}}/>}
        {page==='library'&&<LibraryPage {...{docs,setDocs,docsLoading,setDocsLoading,totalDocs,totalChunks,isLive,fileRef,uploadFiles,uploadBusy,selectedDoc,setSelectedDoc,apiBase}}/>}
        {page==='search'&&<SearchPage {...{searchTerm,setSearchTerm,searchStrategy,setSearchStrategy,searchResults,searchBusy,runSearch,docs}}/>}
      </div>
      <footer className="app-footer"><span>Nexus <span className="footer-dot">·</span> Adaptive retrieval, grounded answers</span><span><span className={`status-dot ${isLive?'online':'offline'}`}/>{isLive?'Connected to backend':'Demo data · Not connected'}</span></footer>
    </main>
    {showSettings&&<div className="modal-scrim" onMouseDown={e=>e.target===e.currentTarget&&setShowSettings(false)}><section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title"><div className="modal-header"><div><div className="eyebrow">CONFIGURATION</div><h2 id="settings-title">API connection</h2></div><IconButton label="Close settings" onClick={()=>setShowSettings(false)}><X size={18}/></IconButton></div><p className="modal-intro">Connect this interface to your running FastAPI service. For local development, the default Vite proxy is not configured: enter your backend origin and enable CORS or add a proxy in <code>vite.config.js</code>.</p><form onSubmit={saveApiSettings}><label className="field-label" htmlFor="api-url">Backend base URL</label><input className="text-input" id="api-url" placeholder="http://localhost:8000" value={apiDraft} onChange={e=>setApiDraft(e.target.value)}/><div className="field-hint">Leave blank to use the built-in preview. The UI expects endpoints such as <code>/health</code>, <code>/metrics</code>, and <code>/query/stream</code>.</div><div className="modal-actions"><button type="button" className="button button-secondary" onClick={()=>{setApiDraft('');setApiBase('');setShowSettings(false);setConnectionChecked(false);setNotice('Switched to preview mode.')}}>Use preview</button><button type="submit" className="button button-primary">Save & connect<ArrowRight size={15}/></button></div></form><div className="settings-divider"/><div className="setup-links"><a href={`${apiDraft.replace(/\/$/,'')||'http://localhost:8000'}/docs`} target="_blank" rel="noreferrer"><FileText size={15}/>Open API docs <ExternalLink size={12}/></a><a href={REPO_URL} target="_blank" rel="noreferrer"><GithubIcon size={15}/>Repository <ExternalLink size={12}/></a></div></section></div>}
    {showLibrary&&<div className="modal-scrim" onMouseDown={e=>e.target===e.currentTarget&&setShowLibrary(false)}><section className="settings-modal library-modal" role="dialog" aria-modal="true"><div className="modal-header"><div><div className="eyebrow">KNOWLEDGE BASE</div><h2>Document library</h2></div><IconButton label="Close library" onClick={()=>setShowLibrary(false)}><X size={18}/></IconButton></div><LibraryPage docs={docs} setDocs={setDocs} totalDocs={totalDocs} totalChunks={totalChunks} isLive={isLive} fileRef={fileRef} uploadFiles={uploadFiles} uploadBusy={uploadBusy} selectedDoc={selectedDoc} setSelectedDoc={setSelectedDoc} apiBase={apiBase}/></section></div>}
    {notice&&<div className="toast" role="status"><CheckCircle2 size={16}/><span>{notice}</span><button className="plain-icon" aria-label="Dismiss notification" onClick={()=>setNotice('')}><X size={14}/></button></div>}
  </div>;
}

function WorkspacePage({query,setQuery,mode,setMode,busy,answer,error,runQuery,queryRef,docs,totalDocs,totalChunks,isLive,setPage,setShowLibrary}) {
  const promptCards=[{icon:SearchCheck,title:'Explain a concept',text:'Give me a sourced explanation of a technical idea.'},{icon:GitBranch,title:'Compare approaches',text:'Trace trade-offs across relevant documents.'},{icon:ShieldCheck,title:'Verify a claim',text:'Check the evidence and call out uncertainty.'}];
  return <div className="workspace-page page-enter">
    <div className="workspace-hero"><div className="hero-overline"><span className="hero-overline-line"/>EVIDENCE-FIRST RESEARCH</div><h1>Good answers start<br/>with <em>better evidence.</em></h1><p className="hero-description">Ask across your knowledge base. Nexus chooses how to retrieve, when to rerank, and when the evidence isn't strong enough.</p><div className="hero-meta"><span><span className={`status-dot ${isLive?'online':'offline'}`}/>{isLive?'Live retrieval enabled':'Workspace preview'}</span><span className="hero-meta-divider"/><span><FileText size={14}/>{totalDocs} documents</span><span className="hero-meta-divider"/><span><Layers3 size={14}/>{totalChunks.toLocaleString()} chunks</span></div></div>
    <section className="query-composer"><div className="composer-topline"><div className="composer-title"><Sparkles size={16}/><span>Ask your knowledge base</span></div><span className="keyboard-hint"><Command size={12}/> K</span></div><form onSubmit={runQuery}><textarea ref={queryRef} value={query} onChange={e=>setQuery(e.target.value)} placeholder="Ask a technical question, compare a few ideas, or validate a claim…" rows={3} disabled={busy} onKeyDown={e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)){e.preventDefault();runQuery()}}}/><div className="composer-bottom"><div className="mode-control"><span className="mode-label">RETRIEVAL MODE</span><div className="segmented-control" role="group" aria-label="Retrieval mode">{['adaptive','fast','balanced','deep'].map(m=><button key={m} type="button" className={mode===m?'selected':''} onClick={()=>setMode(m)} disabled={busy}>{m[0].toUpperCase()+m.slice(1)}</button>)}</div><span className="mode-help" title="Adaptive mode lets the backend planner select retrieval depth"><CircleHelp size={13}/></span></div><button className="button button-primary ask-button" type="submit" disabled={!query.trim()||busy}>{busy?<><LoaderCircle size={16} className="spin"/>Retrieving</>:<>Ask Nexus<ArrowRight size={15}/></>}</button></div></form><div className="composer-footnote"><ShieldCheck size={13}/> Answers are grounded in retrieved context and include source references when available.</div></section>
    {error&&<div className="inline-alert"><CircleAlert size={17}/><span>{error}</span></div>}
    {busy&&<div className="run-progress"><div className="run-progress-head"><span className="progress-spinner"><LoaderCircle size={17}/></span><div><b>Following the evidence trail</b><span>Routing query · retrieving context · validating support</span></div><Tag tone="blue">IN PROGRESS</Tag></div><div className="progress-track"><span/></div><div className="pipeline-mini"><span className="done"><Check size={12}/> Plan</span><i/><span className="current"><LoaderCircle size={12}/> Retrieve</span><i/><span>Rerank if needed</span><i/><span>Generate</span></div></div>}
    {answer&&<AnswerPanel answer={answer} isBusy={busy}/>}
    {!answer&&!busy&&<section className="suggestion-section"><div className="suggestion-heading"><span>START WITH A QUESTION</span><span>SELECT A PROMPT TO BEGIN</span></div><div className="suggestion-list">{DEMO_QUESTIONS.map((q,i)=><button key={q} className="suggestion-row" onClick={()=>setQuery(q)}><span className="suggestion-index">0{i+1}</span><span>{q}</span><ArrowUpRight size={16}/></button>)}</div></section>}
    <section className="workspace-bottom"><div className="bottom-intro"><div className="eyebrow">BUILT FOR TRACEABILITY</div><h3>A little less guessing.<br/><span>A lot more grounding.</span></h3><p>Retrieval decisions and evidence stay inspectable, not hidden behind a chat bubble.</p><button className="text-link" onClick={()=>setPage('metrics')}>Explore pipeline metrics<ArrowRight size={15}/></button></div><div className="principle-list"><div className="principle-row"><span className="principle-icon"><Network size={17}/></span><div><b>Hybrid by design</b><p>Dense vectors and lexical signals work together.</p></div><ChevronRight size={15}/></div><div className="principle-row"><span className="principle-icon"><Gauge size={17}/></span><div><b>Confidence-aware routing</b><p>Spend more compute only when evidence calls for it.</p></div><ChevronRight size={15}/></div><div className="principle-row"><span className="principle-icon"><ShieldCheck size={17}/></span><div><b>Grounded by default</b><p>Source references, citation checks, and abstention paths.</p></div><ChevronRight size={15}/></div></div></section>
  </div>;
}

function AnswerPanel({answer,isBusy}) {
  const sources=answer.sources||[];const confidence=answer.confidence;const retrieval=answer.retrieval||{};const breakdown=retrieval.latency_breakdown_ms||{};
  const [tab,setTab]=useState('answer');const [copied,setCopied]=useState(false);
  async function copyAnswer(){try{await navigator.clipboard.writeText(answer.answer||'');setCopied(true);setTimeout(()=>setCopied(false),1400)}catch{setCopied(false)}}
  return <section className="answer-panel page-enter"><div className="answer-panel-top"><div className="answer-label"><span className="answer-spark"><Sparkles size={14}/></span><span><b>Response</b><small>{isBusy?'Generating answer':'Completed response'}</small></span></div><div className="answer-actions"><Tag tone={answer.abstained?'amber':'green'} dot>{answer.abstained?'Abstained':(answer.citation_validation?.valid===false?'Review citations':'Evidence returned')}</Tag><IconButton label="Copy answer" onClick={copyAnswer}>{copied?<Check size={16}/>:<Copy size={16}/>}</IconButton></div></div><div className="answer-stats"><span><span className="stat-label">ROUTE</span><b>{safeText(answer.execution_mode||retrieval.execution_mode,'Adaptive').toUpperCase()}</b></span><span><span className="stat-label">CONFIDENCE</span><b>{confidence==null?'-':fmt(confidence,2)}</b></span><span><span className="stat-label">LATENCY</span><b>{answer.latency_ms==null?'-':`${fmt(answer.latency_ms,0)} ms`}</b></span><span><span className="stat-label">SOURCES</span><b>{sources.length}</b></span></div><div className="answer-tabs"><button className={tab==='answer'?'active':''} onClick={()=>setTab('answer')}>Answer</button><button className={tab==='sources'?'active':''} onClick={()=>setTab('sources')}>Sources <span>{sources.length}</span></button><button className={tab==='trace'?'active':''} onClick={()=>setTab('trace')}>Execution trace</button></div>{tab==='answer'&&<div className="answer-body">{answer.answer?<p className="answer-text">{answer.answer}</p>:<span className="subtle">Waiting for answer tokens…</span>}{answer.abstention_reason&&<div className="abstention-note"><ShieldCheck size={16}/><div><b>Evidence gate abstention</b><p>{answer.abstention_reason}</p></div></div>}{sources.length>0&&<div className="inline-sources"><span>SUPPORTED BY</span>{sources.map((s,i)=><span key={`${s.citation_id}-${i}`} className="inline-source-chip">[{s.citation_id||i+1}] {s.filename||s.document_id||'Source'}</span>)}</div>}</div>}{tab==='sources'&&<div className="answer-body source-detail-list">{sources.length?sources.map((s,i)=><article className="source-detail" key={`${s.chunk_id||s.document_id}-${i}`}><span className="source-number">[{s.citation_id||i+1}]</span><div><b>{s.filename||s.document_id||'Source document'}</b><p>{s.snippet||'Source returned without a preview snippet.'}</p><small>{[s.page&&`Page ${s.page}`,s.section,s.chunk_id&&`Chunk ${s.chunk_id}`].filter(Boolean).join(' · ')||'Retrieved evidence'}</small></div></article>):<div className="empty-state small-empty"><FileSearch size={21}/><b>No source records returned</b><p>The backend did not attach source references to this answer.</p></div>}</div>}{tab==='trace'&&<div className="answer-body"><div className="trace-list">{[['Query planning',breakdown.planning_ms??breakdown.query_planning_ms],['Hybrid retrieval',breakdown.hybrid_retrieval_ms??breakdown.fast_retrieval_ms],['Fusion & confidence',breakdown.fusion_ms??breakdown.confidence_ms],['Reranking',breakdown.rerank_ms],['Parent expansion',breakdown.parent_expansion_ms],['Generation',breakdown.generation_ms]].map(([name,val],i)=><div className="trace-row" key={name}><span className={`trace-dot ${val==null?'muted':''}`}/><span>{name}</span><b>{val==null?'Not reported':`${fmt(val,1)} ms`}</b></div>)}</div><p className="trace-footnote">Stage timings are shown only when the API returns them. Missing values are not inferred.</p></div>}</section>;
}

function MetricsPage({isLive,health,metrics,refreshOverview}) {
  const [range,setRange]=useState('Benchmark');const [busy,setBusy]=useState(false);
  async function refresh(){setBusy(true);await refreshOverview();setBusy(false)}
  const cards=[
    {label:'Mean reciprocal rank',value:fmt(BENCH.mrr,3),detail:'Rank-sensitive retrieval quality',icon:SearchCheck,accent:'blue'},
    {label:'Recall @ K',value:pct(BENCH.recall),detail:'Relevant evidence recovered',icon:Database,accent:'green'},
    {label:'Citation accuracy',value:pct(BENCH.citation/100),detail:'Reported benchmark result',icon:ShieldCheck,accent:'green'},
    {label:'P50 end-to-end latency',value:fmt(BENCH.latency.p50,0),unit:'ms',detail:'Includes LLM generation',icon:Clock3,accent:'amber'},
  ];
  return <div className="metrics-page page-enter"><SectionHeading eyebrow="OBSERVABILITY" title="Pipeline metrics" description="Inspect retrieval quality, latency, routing decisions, and the live service state." action={<div className="heading-actions"><button className="button button-secondary" onClick={refresh} disabled={busy}>{busy?<LoaderCircle size={15} className="spin"/>:<RefreshCw size={15}/>}Refresh status</button><button className="button button-primary" onClick={()=>setRange(range==='Benchmark'?'Live API':'Benchmark')}>{range==='Benchmark'?'View live status':'View benchmark'}<ArrowRight size={15}/></button></div>}/>
    <div className={`data-notice ${isLive?'live-notice':''}`}><span className={`notice-icon ${isLive?'ok':''}`}>{isLive?<CheckCircle2 size={17}/>:<CircleAlert size={17}/>}</span><div><b>{isLive?'Live service connected':'Benchmark data and live status are separate'}</b><p>{isLive?`Health endpoint reports ${safeText(health?.status)}; operational counts below are from this API session.`:'Quality and latency charts show the repository’s recorded benchmark (110 test cases), not current traffic. Connect the API to fetch live health and index counts.'}</p></div><Tag tone={isLive?'green':'amber'}>{isLive?'LIVE HEALTH':'RECORDED BENCHMARK'}</Tag></div>
    <div className="metric-grid">{(range==='Benchmark'?cards:[{label:'API health',value:isLive?'Healthy':'Offline',detail:'GET /health',icon:Server,accent:isLive?'green':'amber'},{label:'Indexed documents',value:safeText(metrics?.indexed_documents,'-'),detail:'GET /metrics',icon:FileText,accent:'blue'},{label:'Qdrant points',value:metrics?.qdrant_points_count==null?'-':Number(metrics.qdrant_points_count).toLocaleString(),detail:'Vector collection size',icon:Database,accent:'green'},{label:'Active jobs',value:safeText(metrics?.active_jobs_count,'-'),detail:'Reported by the API',icon:Activity,accent:'amber'}]).map(c=><MetricCard key={c.label} {...c}/>)}</div>
    <div className="metrics-main-grid"><section className="surface chart-surface"><div className="surface-heading"><div><h3>Latency distribution</h3><p>Full execution · adaptive mode · 110 cases</p></div><span className="chart-legend"><i className="legend-blue"/>Latency (ms)</span></div><div className="latency-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={[{p:'Min',v:2},{p:'P50',v:863.9},{p:'P90',v:1423.8},{p:'P95',v:1504.4},{p:'P99',v:1851},{p:'Max',v:3179.5}]} margin={{top:12,right:8,left:-18,bottom:0}}><defs><linearGradient id="latencyFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#3478f6" stopOpacity={0.18}/><stop offset="100%" stopColor="#3478f6" stopOpacity={0.015}/></linearGradient></defs><CartesianGrid stroke="#e9e9ed" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="p" tickLine={false} axisLine={false} tick={{fill:'#8a8a93',fontSize:11}} dy={8}/><YAxis tickLine={false} axisLine={false} tick={{fill:'#8a8a93',fontSize:11}}/><Tooltip content={<ChartTooltip suffix=" ms"/>}/><Area type="monotone" dataKey="v" name="Latency" stroke="#3478f6" strokeWidth={2.4} fill="url(#latencyFill)" activeDot={{r:4,strokeWidth:0}}/></AreaChart></ResponsiveContainer></div><div className="chart-footline"><span>Mean <b>{fmt(BENCH.latency.mean,1)} ms</b></span><span>P95 <b>{fmt(BENCH.latency.p95,1)} ms</b></span><span>Maximum <b>{fmt(BENCH.latency.max,0)} ms</b></span></div><p className="chart-caveat">End-to-end wall-clock timings include generation. Retrieval-only measurements are much lower and are not comparable to this chart.</p></section>
      <section className="surface route-surface"><div className="surface-heading"><div><h3>Adaptive routing</h3><p>Share of benchmark queries</p></div><span className="tiny-pill">110 CASES</span></div><div className="route-bar"><span style={{width:`${BENCH.fast}%`}}/><span style={{width:`${BENCH.balanced}%`}}/><span style={{width:`${BENCH.deep}%`}}/></div><div className="route-legend"><div><i className="route-fast"/><span>Fast path</span><b>{fmt(BENCH.fast)}%</b></div><div><i className="route-balanced"/><span>Balanced path</span><b>{fmt(BENCH.balanced)}%</b></div><div><i className="route-deep"/><span>Deep path</span><b>{fmt(BENCH.deep)}%</b></div></div><div className="route-divider"/><div className="route-stat"><span>Cross-encoder rerank rate</span><b>{fmt(BENCH.rerank)}%</b></div><div className="route-stat"><span>Multi-query rate</span><b>{fmt(BENCH.multiQuery)}%</b></div><div className="route-stat"><span>Early abstention rate</span><b>{fmt(BENCH.abstainRate)}%</b></div><p className="micro-note">Fast path bypasses reranking when confidence crosses the configured threshold. Benchmark fast-path frequency is low; this should remain visible rather than overstated.</p></section></div>
    <section className="surface ablation-surface"><div className="surface-heading"><div><h3>Retrieval configuration comparison</h3><p>Recall and retrieval latency from the repository ablation study</p></div><Tag tone="blue">RETRIEVAL ONLY</Tag></div><div className="ablation-table-wrap"><table className="data-table"><thead><tr><th>Configuration</th><th>Recall @ 5</th><th>P50 latency</th><th>P95 latency</th><th>Interpretation</th></tr></thead><tbody>{BENCH_SERIES.map((row,i)=><tr key={row.label}><td><span className={`table-rank rank-${i}`}>{String.fromCharCode(65+i)}</span><b>{row.label}</b></td><td><div className="table-bar-cell"><span className="table-bar"><i style={{width:`${row.recall}%`}}/></span><span>{row.recall.toFixed(2)}%</span></div></td><td className="mono">{row.p50.toFixed(1)} ms</td><td className="mono">{row.p95.toFixed(1)} ms</td><td><span className={`table-insight ${i===2?'good':''}`}>{i===0?'Lowest overhead':i===1?'Higher recall':'Cache / routing trade-off'}</span></td></tr>)}</tbody></table></div><p className="chart-caveat">These measurements compare retrieval stages and should not be directly compared with end-to-end timings including LLM generation.</p></section>
    <div className="metrics-lower-grid"><section className="surface"><div className="surface-heading"><div><h3>Quality signals</h3><p>Recorded adaptive benchmark</p></div><ShieldCheck size={17} className="surface-heading-icon"/></div><QualityRow label="Hit rate @ K" value={BENCH.hitRate}/><QualityRow label="nDCG @ K" value={BENCH.ndcg}/><QualityRow label="Abstention accuracy" value={BENCH.abstention/100}/><QualityRow label="Citation accuracy" value={BENCH.citation/100}/><QualityRow label="Precision @ K" value={BENCH.precision}/></section><section className="surface"><div className="surface-heading"><div><h3>Live index status</h3><p>Values retrieved from your API</p></div><span className={`status-dot ${isLive?'online':'offline'}`}/></div><div className="live-stat-list"><LiveStat icon={FileText} label="Total documents" value={metrics?.documents_total}/><LiveStat icon={CheckCircle2} label="Indexed documents" value={metrics?.indexed_documents}/><LiveStat icon={Database} label="Qdrant points" value={metrics?.qdrant_points_count}/><LiveStat icon={Layers3} label="BM25 chunks" value={metrics?.bm25_chunks_count}/><LiveStat icon={Activity} label="Active jobs" value={metrics?.active_jobs_count}/></div>{!isLive&&<div className="connect-empty"><CircleAlert size={15}/><span>Connect your backend to populate operational values.</span></div>}</section></div>
    <div className="metrics-methodology"><div className="methodology-icon"><CircleHelp size={16}/></div><div><b>How to read these numbers</b><p>Benchmark quality is measured offline against a fixed evaluation set. Live operational metrics describe index state, not answer quality. RAGAS metrics, retrieval latency, and end-to-end latency measure different things; avoid combining them into one score.</p></div></div>
  </div>;
}
function ChartTooltip({active,payload,label,suffix=''}){if(!active||!payload?.length)return null;return <div className="chart-tooltip"><span>{label}</span><b>{fmt(payload[0].value,1)}{suffix}</b></div>}
function QualityRow({label,value}){return <div className="quality-row"><span>{label}</span><span className="quality-track"><i style={{width:`${value*100}%`}}/></span><b>{pct(value)}</b></div>}
function LiveStat({icon:Icon,label,value}){return <div className="live-stat"><span><Icon size={15}/>{label}</span><b>{value==null?'-':Number(value).toLocaleString()}</b></div>}

function LibraryPage({docs,setDocs,totalDocs,totalChunks,isLive,fileRef,uploadFiles,uploadBusy,selectedDoc,setSelectedDoc}) {
  const [filter,setFilter]=useState('All');const [term,setTerm]=useState('');
  const filtered=docs.filter(d=>(filter==='All'||(filter==='Indexed'?d.status==='INDEXED':d.status!=='INDEXED'))&&d.name.toLowerCase().includes(term.toLowerCase()));
  return <div className="library-page page-enter"><SectionHeading eyebrow="KNOWLEDGE BASE" title="Document library" description="Manage the source material your retrieval pipeline can search." action={<><input ref={fileRef} type="file" multiple hidden accept=".pdf,.txt,.md,.docx,.html,.csv,.json" onChange={e=>{uploadFiles(e.target.files);e.target.value=''}}/><button className="button button-primary" onClick={()=>fileRef.current?.click()} disabled={uploadBusy}>{uploadBusy?<LoaderCircle size={15} className="spin"/>:<Plus size={16}/>}Add documents</button></>}/>
    <div className="library-stats"><div><span>Documents</span><b>{totalDocs}</b><small>Available to the workspace</small></div><div><span>Indexed chunks</span><b>{totalChunks.toLocaleString()}</b><small>Source passages across documents</small></div><div><span>Index status</span><b className="status-value"><i className={`status-dot ${isLive?'online':'offline'}`}/>{isLive?'Connected':'Preview'}</b><small>{isLive?'Live API status':'Illustrative document list'}</small></div></div>
    <section className="surface library-table-surface"><div className="library-table-toolbar"><div className="segmented-control compact">{['All','Indexed','Processing'].map(f=><button key={f} className={filter===f?'selected':''} onClick={()=>setFilter(f)}>{f}</button>)}</div><div className="library-search"><Search size={15}/><input value={term} onChange={e=>setTerm(e.target.value)} placeholder="Find a document"/><kbd>/</kbd></div><button className="button button-secondary button-small" onClick={()=>setTerm('')}><Filter size={14}/>Filter</button></div>
      <div className="library-list"><div className="library-list-head"><span>DOCUMENT</span><span>CHUNKS</span><span>STATUS</span><span>LAST UPDATED</span><span/></div>{filtered.map(doc=><button className={`library-row ${selectedDoc?.id===doc.id?'selected':''}`} key={doc.id} onClick={()=>setSelectedDoc(selectedDoc?.id===doc.id?null:doc)}><div className="document-name"><span className={`file-type ${doc.type==='PDF'?'pdf':doc.type==='MD'?'md':'other'}`}><FileText size={16}/></span><span><b>{doc.name}</b><small>{doc.type} <i>·</i> {doc.size||'-'}</small></span></div><span className="chunk-count">{doc.chunks||'-'}</span><span><Tag tone={doc.status==='INDEXED'?'green':'amber'} dot>{doc.status||'UNKNOWN'}</Tag></span><span className="doc-updated">{doc.updated||'-'}</span><ChevronRight size={15} className="row-chevron"/></button>)}{filtered.length===0&&<div className="empty-state"><FileSearch size={25}/><b>No matching documents</b><p>Try a different search term or filter.</p></div>}</div>
      {selectedDoc&&<div className="selected-document"><div><b>{selectedDoc.name}</b><p>{selectedDoc.chunks||0} indexed chunks · {selectedDoc.status}</p></div><button className="button button-secondary button-small" onClick={()=>setSelectedDoc(null)}>Close details<X size={14}/></button></div>}
      <div className="library-table-foot"><span>Showing {filtered.length} of {docs.length} documents</span><span><Database size={13}/>{isLive?'Synced from API':'Preview data'}</span></div>
    </section>
    <div className="upload-note"><Upload size={17}/><div><b>Bring your own knowledge base</b><p>Upload supported files to the running API. Document processing and indexing depend on your backend configuration and may take a moment.</p></div><button className="text-link" onClick={()=>fileRef.current?.click()}>Choose files<ArrowRight size={14}/></button></div>
  </div>;
}

function SearchPage({searchTerm,setSearchTerm,searchStrategy,setSearchStrategy,searchResults,searchBusy,runSearch,docs}) {
  return <div className="search-page page-enter"><SectionHeading eyebrow="RETRIEVAL INSPECTOR" title="Search explorer" description="Inspect retrieved passages before generation. Compare lexical, dense, and hybrid retrieval."/>
    <form className="search-explorer-form surface" onSubmit={runSearch}><div className="search-explorer-input"><Search size={19}/><input value={searchTerm} onChange={e=>setSearchTerm(e.target.value)} placeholder="Search your indexed knowledge…"/><button className="button button-primary" type="submit" disabled={!searchTerm.trim()||searchBusy}>{searchBusy?<LoaderCircle size={15} className="spin"/>:<Search size={15}/>}Search</button></div><div className="search-options"><span className="option-label">RETRIEVER</span><div className="segmented-control compact">{['hybrid','dense','keyword'].map(x=><button type="button" key={x} className={searchStrategy===x?'selected':''} onClick={()=>setSearchStrategy(x)}>{x==='keyword'?'BM25':x[0].toUpperCase()+x.slice(1)}</button>)}</div><span className="search-option-hint">Top 8 candidates <i>·</i> Raw retrieval scores</span></div></form>
    {searchBusy&&<div className="empty-state"><LoaderCircle size={24} className="spin"/><b>Retrieving candidates</b><p>Running {searchStrategy} retrieval against your index.</p></div>}
    {searchResults&&<section className="search-result-section"><div className="search-results-header"><div><h3>Retrieved passages</h3><p>{searchResults.length} results · {searchStrategy.toUpperCase()} strategy</p></div><Tag tone="blue">LIVE SEARCH</Tag></div>{searchResults.length?searchResults.map((r,i)=><article className="search-result surface" key={r.chunk_id||i}><div className="result-top"><div className="result-source"><span className="result-rank">{String(i+1).padStart(2,'0')}</span><div><b>{r.filename||r.document_id||r.metadata?.filename||'Retrieved document'}</b><small>{r.retriever||searchStrategy} <i>·</i> {r.page?`Page ${r.page}`:'Passage'}</small></div></div><div className="result-score"><span>SCORE</span><b>{typeof r.score==='number'?fmt(r.score,4):'-'}</b></div></div><p className="result-excerpt">{r.text||r.snippet||'No passage text returned.'}</p><div className="result-bottom"><span><Layers3 size={13}/>{r.chunk_id||'Chunk ID unavailable'}</span><span>Rank {r.rank||i+1}</span></div></article>):<div className="empty-state"><FileSearch size={25}/><b>No passages found</b><p>Try a different query or retrieval strategy.</p></div>}</section>}
    {!searchResults&&!searchBusy&&<div className="search-empty-layout"><div className="search-empty-graphic"><div className="search-orbit orbit-one"/><div className="search-orbit orbit-two"/><div className="search-core"><Database size={28}/></div><span className="orbit-node node-a"/><span className="orbit-node node-b"/><span className="orbit-node node-c"/></div><h3>Look closer at retrieval.</h3><p>Search results show which passages the retriever considers relevant before the generation step begins.</p><div className="search-methods"><span><Network size={15}/>Hybrid fusion</span><span><Zap size={15}/>Dense similarity</span><span><BookOpen size={15}/>BM25 lexical</span></div></div>}
  </div>;
}

function createStylesPlaceholder(){return null}

createRoot(document.getElementById('root')).render(<App/>);
