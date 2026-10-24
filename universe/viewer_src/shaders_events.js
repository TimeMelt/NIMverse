// ---- high-energy event shaders (v4) ----
const EV_N = NOISE + `
float efbm(vec3 p){float s=0.,a=.5;for(int i=0;i<5;i++){s+=a*snoise(p);p=p*2.07+vec3(3.1,1.7,5.3);a*=.5;}return s;}`;
const VS_EV = `varying vec3 vP; varying vec3 vN; varying vec3 vV; varying vec2 vUv;
void main(){vP=position; vUv=uv; vN=normalize(normalMatrix*normal); vec4 mv=modelViewMatrix*vec4(position,1.); vV=normalize(-mv.xyz); gl_Position=projectionMatrix*mv;}`;
// supernova remnant shell: filamentary noise on a thin expanding shell, limb-brightened (fresnel), colour cools with age
const FS_SHELL = EV_N + `uniform float uAge; uniform vec3 uSeed; uniform float uAlpha; varying vec3 vP; varying vec3 vN; varying vec3 vV;
void main(){float f=1.-abs(dot(vN,vV)); float limb=pow(f,2.2);
 float n=efbm(vP*2.6+uSeed+uAge*.15); float fil=pow(1.-abs(snoise(vP*5.+uSeed*1.3)),6.);
 float m=clamp(.55+.8*n,0.,1.)*(limb*1.6+.15)+fil*limb*1.5;
 vec3 hot=vec3(.75,.85,1.), oiii=vec3(.2,.95,.8), ha=vec3(1.,.25,.35);
 vec3 c=mix(hot, mix(oiii, ha, clamp(n*.9+.5,0.,1.)), clamp(uAge*2.5,0.,1.));
 gl_FragColor=vec4(c*m*uAlpha*.75, 1.);}`;
// accretion disk: temperature falls with radius, Doppler-beamed on the approaching side, turbulent rotating noise
const FS_DISK = EV_N + `uniform float uTime; uniform float uIn; uniform float uAct; varying vec3 vP; varying vec2 vUv;
void main(){float r=length(vP.xy); float x=(r-uIn)/(1.-uIn); if(x<0.||x>1.) discard; float ang=atan(vP.y,vP.x);
 float n=efbm(vec3(cos(ang-uTime*1.5/(.3+x))*r*4., sin(ang-uTime*1.5/(.3+x))*r*4., x*6.));
 float T=pow(1.-x,1.6); float dop=1.+.65*sin(ang); vec3 c=mix(vec3(1.,.35,.08), vec3(1.,.95,.85), T)*(.55+.6*n)*dop;
 float a=smoothstep(0.,.08,x)*smoothstep(1.,.55,x); gl_FragColor=vec4(c*a*(2.6+2.4*uAct)*(.5+T), 1.);}`;
// relativistic jet / pulsar beam: additive cone, bright core, knots flowing outward
const FS_JET = EV_N + `uniform float uTime; uniform vec3 uCol; uniform float uAlpha; varying vec2 vUv; varying vec3 vN; varying vec3 vV;
void main(){float along=vUv.y; float core=pow(abs(dot(vN,vV)),2.5); float knots=.6+.4*snoise(vec3(along*9.-uTime*3., vUv.x*3., 0.));
 float a=core*knots*smoothstep(1.,.15,along)*smoothstep(0.,.05,along)*uAlpha; gl_FragColor=vec4(uCol*a,1.);}`;
// expanding gravitational-wave ripple ring (SMBH merger)
const FS_RING = `uniform float uAge; varying vec2 vUv; void main(){float r=vUv.x; float a=exp(-pow((r-.5)*8.,2.))*(1.-uAge); gl_FragColor=vec4(vec3(.6,.75,1.)*a*.8,1.);}`;
// screen-space gravitational lensing + shadow around up to 8 black holes (Einstein-ring style deflection)
const VS_Q = `varying vec2 vUv; void main(){vUv=uv; gl_Position=vec4(position.xy,0.,1.);}`;
const FS_LENS = `uniform sampler2D tDiffuse; uniform vec4 uL[8]; uniform int uNL; uniform float uAsp; varying vec2 vUv;
void main(){vec2 uv=vUv; float shadow=1.;
 for(int i=0;i<8;i++){if(i>=uNL)break; vec2 d=uv-uL[i].xy; d.x*=uAsp; float r=length(d), re=uL[i].z;
  if(r<re*3.){float k=re*re/max(r*r,1e-6); vec2 off=d/max(r,1e-6)*re*min(k,1.5)*.3; off.x/=uAsp; uv+=off*smoothstep(re*3.,re,r)*uL[i].w;
   shadow*=smoothstep(re*.26,re*.34,r);}}
 vec3 c=texture2D(tDiffuse,uv).rgb; gl_FragColor=vec4(c*shadow,1.);}`;
