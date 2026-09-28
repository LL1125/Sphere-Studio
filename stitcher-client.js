(function(global){
'use strict';

const DB_NAME='sphere-studio-local';
const DB_VERSION=1;
const STORE_NAME='panoramas';

function openDB(){
  return new Promise((resolve,reject)=>{
    const req=indexedDB.open(DB_NAME,DB_VERSION);
    req.onupgradeneeded=()=>{
      const db=req.result;
      if(!db.objectStoreNames.contains(STORE_NAME)) db.createObjectStore(STORE_NAME);
    };
    req.onsuccess=()=>resolve(req.result);
    req.onerror=()=>reject(req.error||new Error('IndexedDB open failed'));
  });
}

async function savePanoramaBlob(key,blob){
  const db=await openDB();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE_NAME,'readwrite');
    tx.objectStore(STORE_NAME).put(blob,key);
    tx.oncomplete=()=>{db.close();resolve(true)};
    tx.onerror=()=>{const e=tx.error;db.close();reject(e)};
  });
}

async function loadPanoramaBlob(key){
  const db=await openDB();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE_NAME,'readonly');
    const req=tx.objectStore(STORE_NAME).get(key);
    req.onsuccess=()=>{const v=req.result||null;db.close();resolve(v)};
    req.onerror=()=>{const e=req.error;db.close();reject(e)};
  });
}

async function removePanoramaBlob(key){
  const db=await openDB();
  return new Promise((resolve,reject)=>{
    const tx=db.transaction(STORE_NAME,'readwrite');
    tx.objectStore(STORE_NAME).delete(key);
    tx.oncomplete=()=>{db.close();resolve(true)};
    tx.onerror=()=>{const e=tx.error;db.close();reject(e)};
  });
}

function compile(gl,type,src){
  const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)||'Shader compile failed');
  return s;
}
function program(gl,vs,fs){
  const p=gl.createProgram();gl.attachShader(p,compile(gl,gl.VERTEX_SHADER,vs));gl.attachShader(p,compile(gl,gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)||'Program link failed');
  return p;
}

function yawPitchBasis(yaw,pitch){
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  return {
    right:[cy,0,-sy],
    up:[-sy*sp,cp,-cy*sp],
    forward:[sy*cp,sp,-cy*cp]
  };
}

function normalizedBasis(frame,index){
  if(frame&&frame.basis&&frame.basis.right&&frame.basis.up&&frame.basis.forward) return frame.basis;
  const target=frame&&frame.target?frame.target:{};
  let yaw=Number(target.yaw);
  let pitch=Number(target.pitch);
  if(!Number.isFinite(yaw)||!Number.isFinite(pitch)){
    const i=index+1;
    if(i<=12){yaw=(i-1)*Math.PI/6;pitch=0}
    else if(i<=24){yaw=(i-13+.5)*Math.PI/6;pitch=Math.PI/4}
    else if(i<=36){yaw=(i-25+.5)*Math.PI/6;pitch=-Math.PI/4}
    else if(i===37){yaw=0;pitch=Math.PI/2}
    else{yaw=0;pitch=-Math.PI/2}
  }
  return yawPitchBasis(yaw,pitch);
}

function blobToImage(blob){
  return new Promise((resolve,reject)=>{
    const url=URL.createObjectURL(blob);const im=new Image();
    im.onload=()=>{URL.revokeObjectURL(url);resolve(im)};
    im.onerror=e=>{URL.revokeObjectURL(url);reject(e||new Error('Image decode failed'))};
    im.src=url;
  });
}

function canvasToBlob(canvas,type,quality){
  return new Promise((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(new Error('Panorama encode failed')),type,quality));
}

async function stitch(frames,opts){
  opts=opts||{};
  if(!frames||frames.length<3) throw new Error('Not enough frames');
  const width=opts.width||4096;
  const height=Math.round(width/2);
  const hfov=(opts.hfovDeg||76)*Math.PI/180;
  const onProgress=typeof opts.onProgress==='function'?opts.onProgress:function(){};

  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const gl=canvas.getContext('webgl',{alpha:false,antialias:false,preserveDrawingBuffer:true,premultipliedAlpha:false});
  if(!gl) throw new Error('WebGL is required for automatic stitching');

  const vs='attribute vec2 aPos;varying vec2 vUV;void main(){vUV=(aPos+1.0)*0.5;gl_Position=vec4(aPos,0.0,1.0);}';
  const fs='precision highp float;varying vec2 vUV;uniform sampler2D uTex;uniform vec3 uRight,uUp,uForward;uniform float uTanH,uTanV;const float PI=3.141592653589793;void main(){float lon=(vUV.x*2.0-1.0)*PI;float lat=(0.5-vUV.y)*PI;float cl=cos(lat);vec3 d=vec3(sin(lon)*cl,sin(lat),-cos(lon)*cl);float cx=dot(d,uRight);float cy=dot(d,uUp);float cz=dot(d,uForward);if(cz<=0.0001)discard;float nx=cx/(cz*uTanH);float ny=cy/(cz*uTanV);if(abs(nx)>1.0||abs(ny)>1.0)discard;vec2 uv=vec2((nx+1.0)*0.5,(ny+1.0)*0.5);vec3 color=texture2D(uTex,uv).rgb;float edge=min(1.0-abs(nx),1.0-abs(ny));float a=smoothstep(0.015,0.28,edge);a=pow(a,0.72);gl_FragColor=vec4(color,a);}';
  const p=program(gl,vs,fs);gl.useProgram(p);
  const pos=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pos);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
  const aPos=gl.getAttribLocation(p,'aPos');gl.enableVertexAttribArray(aPos);gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,0,0);
  const loc=n=>gl.getUniformLocation(p,n);
  const tex=gl.createTexture();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.uniform1i(loc('uTex'),0);
  gl.viewport(0,0,width,height);gl.clearColor(0.03,0.035,0.045,1);gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  gl.disable(gl.DEPTH_TEST);

  const order=[];
  for(let i=0;i<frames.length;i++) order.push(i);
  order.sort((a,b)=>{
    const ra=frames[a]&&frames[a].target?frames[a].target.row:'';
    const rb=frames[b]&&frames[b].target?frames[b].target.row:'';
    const rank=x=>x==='水平'?0:(x==='上方'||x==='下方'?1:2);
    return rank(ra)-rank(rb)||a-b;
  });

  for(let n=0;n<order.length;n++){
    const i=order[n],frame=frames[i];
    const source=await blobToImage(frame.blob);
    const basis=normalizedBasis(frame,i);
    const aspect=source.naturalHeight/Math.max(1,source.naturalWidth);
    const tanH=Math.tan(hfov/2);
    const tanV=tanH*aspect;
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,source);
    gl.uniform3fv(loc('uRight'),new Float32Array(basis.right));
    gl.uniform3fv(loc('uUp'),new Float32Array(basis.up));
    gl.uniform3fv(loc('uForward'),new Float32Array(basis.forward));
    gl.uniform1f(loc('uTanH'),tanH);gl.uniform1f(loc('uTanV'),tanV);
    gl.drawArrays(gl.TRIANGLES,0,6);
    gl.flush();
    onProgress({stage:'stitching',current:n+1,total:order.length,progress:Math.round((n+1)/order.length*88)});
    await new Promise(r=>requestAnimationFrame(r));
  }

  gl.finish();
  onProgress({stage:'encoding',current:frames.length,total:frames.length,progress:94});
  const blob=await canvasToBlob(canvas,'image/jpeg',0.95);
  onProgress({stage:'done',current:frames.length,total:frames.length,progress:100});
  return {blob,width,height};
}

global.SphereClientStitch={stitch,savePanoramaBlob,loadPanoramaBlob,removePanoramaBlob};
})(window);
