/* A real live worker stream delivered by parent; no recording, server or fetch. */
window.NIM_TRANSPORT=new Promise(resolve=>{
 let info,cb,last;
 const t={advance(){},start(fn){cb=fn;if(last)cb(last);},control(body){parent.postMessage({nim:'control',body},'*');return Promise.resolve({});}};
 addEventListener('message',e=>{if(e.source!==parent)return;const d=e.data;if(d.nim==='info'&&!info){info=d.info;t.info=info;resolve(t);}if(d.nim==='tick'){last=d.tick;if(cb)cb(last);}if(d.nim==='view'&&window.NIM)window.NIM.view(d.view);if(d.nim==='photo'&&window.NIM)window.NIM.setLook(d.photo?'photo':'stylized');});
 parent.postMessage({nim:'surface-ready'},'*');
});
