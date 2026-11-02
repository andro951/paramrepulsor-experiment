// Minimal-DOM control/geometry/cadence checks; not browser layout QA.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const fixtures=JSON.parse(html.match(/const previewDatasets=(\{[^\n]+\});/)[1]);
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.listeners={};this.style={};this.disabled=false;this.checked=false;this.hidden=false;this.textContent='';}
 setAttribute(k,v){this.attrs[k]=String(v);}getAttribute(k){return this.attrs[k]??null;}
 append(...children){this.children.push(...children);}replaceChildren(...children){this.children=[...children];}
 addEventListener(name,fn){this.listeners[name]=fn;}
}
const elements={};for(const id of ['variant','n','seed','seconds','start','reset','labels','error','progress','status','detail','phase-note','epoch-count','input','output','output-title','embedding-caption','training-key','selection-key','legend'])elements[id]=new Element(id);
elements.variant.value='baseline';elements.n.value='36';elements.seed.value='42';elements.seconds.value='90';
const points=fixtures['36'];let state={status:'idle'},posts=0,resets=0,failNetwork=false,pairReady=false,availableFrames=0,clock=0;
let trace=null;
function mockTrace(n){return{n,calls:[{sequence:0,kind:'annoy_query',source:0,returned:[0,1]},{sequence:1,kind:'distance_check',source:0,target:1,distance:1}],pairs:Object.fromEntries([['NN',10],['MN',5],['FP',20]].map(([family,count])=>[family,Array.from({length:n},(_,i)=>Array.from({length:count},(_,j)=>[i,(i+j+1)%n])).flat()]))};}
const context=vm.createContext({document:{getElementById:id=>elements[id],createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>({tag:'text',textContent:text,attrs:{},children:[]})},setInterval:()=>{},requestAnimationFrame:()=>{},performance:{now:()=>clock},console,fetch:async(url,options)=>{
 if(failNetwork)throw Error('Disconnected');
 if(options?.method==='POST'){
  if(url==='/api/reset'){resets++;state={status:'idle'};pairReady=false;availableFrames=0;return{ok:true,json:async()=>state};}
  posts++;const req=JSON.parse(options.body);assert.ok(['baseline','modified'].includes(req.variant));assert.equal(req.n,36);assert.equal(req.seed,42);
  pairReady=false;availableFrames=0;trace=mockTrace(req.n);trace.pairs.MN[0]=[0,1];trace.pairs.FP[0]=[0,1];trace.pairs.FP[1]=[0,1];trace.pairs.FP[2]=[0,0];state={status:'preparing',stage:'loading_runtime',replay_version:1,variant:req.variant,n:req.n,seed:req.seed,epoch:0,points:fixtures[String(req.n)],coordinates:null,run_id:`run${posts}`,elapsed_seconds:0};
  return{ok:true,json:async()=>({status:'preparing',run_id:state.run_id})};
 }
 if(url.startsWith('/api/pairs'))return{ok:true,status:pairReady?200:202,json:async()=>({ready:pairReady,run_id:state.run_id,trace})};
 if(url.startsWith('/api/frames')){const after=Number(new URL('http://local'+url).searchParams.get('after')),frames=[];for(let epoch=after+1;epoch<Math.min(availableFrames,after+129);epoch++)frames.push({epoch,coordinates:points.map(p=>[p.x/10+epoch/1000,p.y/10-epoch/1000])});return{ok:true,json:async()=>({run_id:state.run_id,frames})};}
 return{ok:true,json:async()=>state};
}});
vm.runInContext(html.split('<script>')[1].split('</script>')[0],context);
const evaluate=code=>vm.runInContext(code,context);
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const inputFingerprint=()=>JSON.stringify(elements.input.children.filter(n=>n.attrs['data-role']==='input-point').map(n=>n.attrs));
function advance(){clock=Math.max(clock,Number(evaluate('demo.nextAt')));evaluate(`tick(${clock})`);}
async function main(){
 assert.ok(html.indexOf('id="input-panel"')<html.indexOf('id="demo-status-block"'));
 assert.ok(html.indexOf('id="demo-status-block"')<html.indexOf('id="output-panel"'));
 assert.ok(html.indexOf('id="epoch-count"')<html.indexOf('<svg id="output"'));
 assert.ok(html.includes('grid-template-columns:minmax(0,1fr)'));assert.ok(html.includes('main{max-width:none;width:100%}'));assert.ok(html.includes('calc(100dvh - 180px)'));assert.ok(html.includes("'--plot-ratio'"));
 // The real input is drawn synchronously before even the first status response.
 assert.equal(elements.input.children.filter(n=>n.attrs['data-role']==='input-point').length,36);assert.equal(posts,0);
 assert.equal(role('selection-point').length,36);assert.ok(role('selection-point').every(n=>n.attrs.fill==='#242b28'));
 const fixed=inputFingerprint(),inputNode=elements.input.children[0];await flush();
 await elements.start.listeners.click();assert.equal(posts,1);assert.equal(elements.start.disabled,true);assert.equal(inputFingerprint(),fixed);assert.equal(elements.input.children[0],inputNode);
 await elements.start.listeners.click();assert.equal(posts,1);
 clock=100;evaluate('tick(100)');assert.equal(role('active-source').length,0);assert.equal(inputFingerprint(),fixed);
 elements.seconds.value='60';elements.seconds.listeners.change();assert.equal(evaluate('demo.target'),60);assert.equal(elements.seconds.disabled,false);
 pairReady=true;availableFrames=451;state={...state,status:'completed',epoch:450,coordinates:points.map(p=>[p.x/10+.45,p.y/10-.45]),elapsed_seconds:5};
 for(let i=0;i<4;i++)await evaluate('poll()');
 clock=1000;evaluate('tick(1000)');assert.equal(evaluate('demo.events.length'),36);assert.deepEqual(Array.from(evaluate('demo.events.map(e=>e.source)')),Array.from({length:36},(_,i)=>i));assert.equal(role('active-source').length,1);assert.equal(role('active-source')[0].attrs.fill,'rgb(147,89,190)');assert.equal(inputFingerprint(),fixed);checkAnchor(trace,0);
 const fullState=state;state={status:'preparing',run_id:fullState.run_id};await evaluate('poll()');advance();
 assert.equal(role('checking-line').length,0);assert.match(elements.status.textContent,/Anchor/);checkAnchor(trace,1);state=fullState;await evaluate('poll()');
 advance();checkAnchor(trace,2);assert.equal(inputFingerprint(),fixed);
 const step=evaluate('demo.step');clock+=3000;evaluate(`tick(${clock})`);assert.equal(evaluate('demo.step'),step+1,'no burst catch-up after a delayed tick');evaluate(`tick(${clock})`);assert.equal(evaluate('demo.step'),step+1);
 const oldInterval=evaluate('demo.interval');elements.seconds.value='20';elements.seconds.listeners.change();assert.ok(evaluate('demo.interval')<oldInterval,'duration can be shortened during playback');
 const seen=[];for(let safety=0;safety<1000&&evaluate('demo.mode')!=='complete';safety++){
  const before=evaluate('demo.nextEpoch');advance();const after=evaluate('demo.nextEpoch');if(after>before)seen.push(after-1);
  assert.equal(inputFingerprint(),fixed);assert.equal(elements.input.children[0],inputNode);
 }
 assert.equal(evaluate('demo.mode'),'complete');assert.deepEqual(seen,Array.from({length:451},(_,i)=>i));assert.equal(elements.progress.value,450);assert.equal(elements['epoch-count'].textContent,'Epoch 450 / 450');assert.equal(elements.start.disabled,false);
 await elements.reset.listeners.click();assert.equal(resets,1);assert.equal(elements['epoch-count'].textContent,'Epoch 0 / 450');assert.equal(role('active-source').length,0);assert.equal(role('selection-point').length,36);assert.equal(inputFingerprint(),fixed);
 elements.n.value='50';elements.n.listeners.change();assert.equal(elements.input.children.filter(n=>n.attrs['data-role']==='input-point').length,50);assert.equal(posts,1);elements.n.value='36';elements.n.listeners.change();
 await elements.start.listeners.click();assert.equal(posts,2);state={...state,status:'error',error:'Real backend error'};await evaluate('poll()');assert.equal(elements.error.textContent,'Real backend error');assert.equal(elements.start.disabled,false);
 elements.seed.value='-1';await elements.start.listeners.click();assert.equal(posts,2);assert.match(elements.error.textContent,/integer seed/);elements.seed.value='42';elements.seconds.value='0';await elements.start.listeners.click();assert.equal(posts,2);assert.match(elements.error.textContent,/positive/);elements.seconds.value='90';
 elements.variant.value='modified';await elements.start.listeners.click();assert.equal(posts,3);assert.equal(state.variant,'modified');await elements.reset.listeners.click();
 failNetwork=true;await evaluate('poll()');assert.match(elements.error.textContent,/Disconnected/);failNetwork=false;
 checkCorrespondences();if(process.env.PARAMREP_QA_STATE)exportActualPlot();if(process.env.PARAMREP_QA_TRACE)exportSelection();
 console.log('PASS: synchronous real input, invariant input geometry/colors, neutral output, grouped per-anchor NN/MN/FP roles and exact slide RGB, overlapping/repeated memberships, ordered 451-frame cadence, no burst catch-up, editable duration, reset/restart/variants/errors; true-input overlay geometry retained; full center-to-center lines without arrowheads; stacked full-width DOM order and epoch label passed. No full browser QA.');
}
function role(name){return elements.output.children.filter(n=>n.attrs['data-role']===name);}
function number(n,k){return Number(n.attrs[k]);}
function project(x,y){const a=elements.output.attrs;return[Number(a['data-left'])+(x-Number(a['data-xmin']))/(Number(a['data-xmax'])-Number(a['data-xmin']))*(Number(a['data-right'])-Number(a['data-left'])),Number(a['data-bottom'])-(y-Number(a['data-ymin']))/(Number(a['data-ymax'])-Number(a['data-ymin']))*(Number(a['data-bottom'])-Number(a['data-top']))];}
function near(a,b){assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);}
function renderCase(s){context.caseState=s;vm.runInContext('render(caseState)',context);}
function checkGeometry(s){
 const origins=role('original-input'),current=role('current-output'),lines=role('correspondence'),heads=role('arrowhead');
 assert.equal(origins.length,s.points.length);assert.equal(current.length,s.points.length);assert.equal(lines.length,s.points.length);assert.equal(heads.length,0);
 for(let i=0;i<s.points.length;i++){
  const p=s.points[i],o=origins[i],c=current[i],origin=project(p.x,p.y),target=project(...s.coordinates[i]);
  assert.equal(o.attrs['data-point-id'],p.point_id);assert.equal(c.attrs['data-point-id'],p.point_id);
  near(number(o,'data-x'),p.x);near(number(o,'data-y'),p.y);near(number(o,'cx'),origin[0]);near(number(o,'cy'),origin[1]);
  near(number(c,'data-x'),s.coordinates[i][0]);near(number(c,'data-y'),s.coordinates[i][1]);near(number(c,'cx'),target[0]);near(number(c,'cy'),target[1]);
  assert.ok(number(o,'fill-opacity')<=.05&&number(o,'stroke-opacity')<=.25);assert.equal(number(c,'fill-opacity'),1);assert.ok(number(c,'r')<3&&number(c,'r')<number(o,'r'));
  for(const circle of[o,c]){assert.ok(number(circle,'cx')-number(circle,'r')>=number(elements.output,'data-left'));assert.ok(number(circle,'cx')+number(circle,'r')<=number(elements.output,'data-right'));assert.ok(number(circle,'cy')-number(circle,'r')>=number(elements.output,'data-top'));assert.ok(number(circle,'cy')+number(circle,'r')<=number(elements.output,'data-bottom'));}
  const l=lines.find(l=>l.attrs['data-point-id']===p.point_id);assert.ok(l);
  near(number(l,'x1'),origin[0]);near(number(l,'y1'),origin[1]);near(number(l,'x2'),target[0]);near(number(l,'y2'),target[1]);
  near(number(l,'data-origin-x'),p.x);near(number(l,'data-origin-y'),p.y);near(number(l,'data-target-x'),s.coordinates[i][0]);near(number(l,'data-target-y'),s.coordinates[i][1]);
  assert.ok(number(l,'stroke-opacity')<=.25);assert.equal(l.attrs['marker-end'],undefined);

 }
 const a=elements.output.attrs,sx=(Number(a['data-right'])-Number(a['data-left']))/(Number(a['data-xmax'])-Number(a['data-xmin'])),sy=(Number(a['data-bottom'])-Number(a['data-top']))/(Number(a['data-ymax'])-Number(a['data-ymin']));near(sx,sy);
 const all=[...s.points.map(p=>[p.x,p.y]),...s.coordinates],xs=all.map(v=>v[0]),ys=all.map(v=>v[1]);
 const fitX=(Math.max(...xs)-Math.min(...xs))/(Number(a['data-xmax'])-Number(a['data-xmin'])),fitY=(Math.max(...ys)-Math.min(...ys))/(Number(a['data-ymax'])-Number(a['data-ymin']));near(Math.max(fitX,fitY),1/1.055);
 const labels=elements.output.children.filter(n=>n.attrs.class==='ids'),boxes=labels.map(l=>({x:number(l,'x'),y:number(l,'y')-9,w:l.textContent.length*4.5+3,h:10}));
 const overlap=(a,b)=>a.x<b.x+b.w&&a.x+a.w>b.x&&a.y<b.y+b.h&&a.y+a.h>b.y;
 boxes.forEach((box,i)=>{for(let j=0;j<i;j++)assert.equal(overlap(box,boxes[j]),false);for(const c of current)assert.equal(overlap(box,{x:number(c,'cx')-4,y:number(c,'cy')-4,w:8,h:8}),false);});
}
function checkCorrespondences(){
 assert.doesNotMatch(html,/<input id="labels"[^>]*checked/);assert.ok(!html.includes('s.initial_coordinates'));
 let s={points:[points[1],points[0]],coordinates:[[-4,3],[2,5]],initial_coordinates:[[1e9,-1e9],[-1e9,1e9]]};
 renderCase(s);checkGeometry(s);const bounds=JSON.stringify(elements.output.attrs);
 renderCase({...s,initial_coordinates:[[0,0],[0,0]]});assert.equal(JSON.stringify(elements.output.attrs),bounds,'obsolete network starts cannot affect bounds');
 s={...s,coordinates:[[10,11],[-8,4]]};renderCase(s);checkGeometry(s);
 const coincident=points.slice(0,2);renderCase({points:coincident,coordinates:coincident.map(p=>[p.x,p.y])});assert.equal(role('correspondence').length,2);assert.equal(role('original-input').length,2);assert.equal(role('current-output').length,2);
}
function xml(value){return String(value).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');}
function serialize(n){const attrs=Object.entries(n.attrs||{}).map(([k,v])=>` ${k}="${xml(v)}"`).join('');return `<${n.tag}${attrs}>${xml(n.textContent||'')}${n.children.map(serialize).join('')}</${n.tag}>`;}
function exportActualPlot(){
 const s=JSON.parse(fs.readFileSync(process.env.PARAMREP_QA_STATE,'utf8'));assert.equal(s.n,100);assert.equal(s.seed,42);assert.equal(s.epoch,450);
 const dir=process.env.PARAMREP_QA_OUTPUT||__dirname;fs.mkdirSync(dir,{recursive:true});
 for(const enabled of[false,true]){
  elements.labels.checked=enabled;renderCase(s);checkGeometry(s);
  const a=elements.output.attrs,all=[...s.points.map(p=>[p.x,p.y]),...s.coordinates],xs=all.map(v=>v[0]),ys=all.map(v=>v[1]);
  assert.ok((Math.max(...xs)-Math.min(...xs))/(Number(a['data-xmax'])-Number(a['data-xmin']))>.94);
  assert.ok((Math.max(...ys)-Math.min(...ys))/(Number(a['data-ymax'])-Number(a['data-ymin']))>.94);
  const suffix=enabled?'ids-on':'ids-off';
  for(const id of['input','output']){
   const svg=elements[id];svg.tag='svg';svg.setAttribute('xmlns','http://www.w3.org/2000/svg');
   const css='.raw{font-family:Arial;font-size:10px;fill:#5e7768}.axis{font-family:Arial;font-size:12px;fill:#496657}.ids{font-family:Consolas;font-size:7.5px;fill:#344b40;paint-order:stroke;stroke:white;stroke-width:1.6px;stroke-linejoin:round}';
   const svgText=serialize(svg).replace('>',`><rect width="${svg.attrs.width}" height="${svg.attrs.height}" fill="white"/><style>${css}</style>`);
   fs.writeFileSync(path.join(dir,`baseline100-seed42-${id}-${suffix}.svg`),svgText);
  }
 }
}

function checkAnchor(trace,source){
 const circles=role('selected-target'),links=role('selected-pair');
 for(const [family,color]of Object.entries({NN:'rgb(93,205,170)',MN:'rgb(255,171,56)',FP:'rgb(110,181,232)'})){
  const expected=trace.pairs[family].filter(p=>p[0]===source).map(p=>p[1]),id=index=>fixtures['36'][index].point_id;
  const actual=circles.filter(n=>n.attrs['data-family']===family);
  assert.deepEqual(actual.map(n=>n.attrs['data-point-id']).sort(),[...new Set(expected.map(id))].sort());
  assert.ok(actual.every(n=>n.attrs.stroke===color));assert.equal(links.filter(n=>n.attrs['data-family']===family).reduce((sum,n)=>sum+Number(n.attrs['data-slots']),0),expected.length);
 }
 assert.equal(role('active-source')[0].attrs.fill,'rgb(147,89,190)');
}
function exportSelection(){
 const s=JSON.parse(fs.readFileSync(process.env.PARAMREP_QA_STATE,'utf8')),trace=JSON.parse(fs.readFileSync(process.env.PARAMREP_QA_TRACE,'utf8'));
 const groups=Object.fromEntries(['NN','MN','FP'].map(f=>[f,trace.pairs[f].filter(p=>p[0]===0).map(p=>p[1])]));
 neutralOutputForTest(s.points,{kind:'anchor',source:0,groups});
 const svg=elements.output;svg.tag='svg';svg.setAttribute('xmlns','http://www.w3.org/2000/svg');
 fs.writeFileSync(path.join(process.env.PARAMREP_QA_OUTPUT||__dirname,'selection-anchor.svg'),serialize(svg));
}
function neutralOutputForTest(points,event){context.selectionPoints=points;context.selectionEvent=event;evaluate('neutralOutput(selectionPoints,selectionEvent)');}
main().catch(error=>{console.error(error);process.exitCode=1;});
