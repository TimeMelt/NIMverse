// ---- GLSL: 3D simplex noise (Ashima / Stefan Gustavson, MIT) + fBm/ridged + Worley ----
const NOISE = `
vec3 mod289(vec3 x){return x-floor(x*(1./289.))*289.;}
vec4 mod289(vec4 x){return x-floor(x*(1./289.))*289.;}
vec4 permute(vec4 x){return mod289(((x*34.)+1.)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-.85373472095314*r;}
float snoise(vec3 v){const vec2 C=vec2(1./6.,1./3.);const vec4 D=vec4(0.,.5,1.,2.);
 vec3 i=floor(v+dot(v,C.yyy));vec3 x0=v-i+dot(i,C.xxx);vec3 g=step(x0.yzx,x0.xyz);vec3 l=1.-g;
 vec3 i1=min(g.xyz,l.zxy);vec3 i2=max(g.xyz,l.zxy);vec3 x1=x0-i1+C.xxx;vec3 x2=x0-i2+C.yyy;vec3 x3=x0-D.yyy;
 i=mod289(i);vec4 p=permute(permute(permute(i.z+vec4(0.,i1.z,i2.z,1.))+i.y+vec4(0.,i1.y,i2.y,1.))+i.x+vec4(0.,i1.x,i2.x,1.));
 float n_=.142857142857;vec3 ns=n_*D.wyz-D.xzx;vec4 j=p-49.*floor(p*ns.z*ns.z);vec4 x_=floor(j*ns.z);vec4 y_=floor(j-7.*x_);
 vec4 x=x_*ns.x+ns.yyyy;vec4 y=y_*ns.x+ns.yyyy;vec4 h=1.-abs(x)-abs(y);vec4 b0=vec4(x.xy,y.xy);vec4 b1=vec4(x.zw,y.zw);
 vec4 s0=floor(b0)*2.+1.;vec4 s1=floor(b1)*2.+1.;vec4 sh=-step(h,vec4(0.));vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
 vec3 p0=vec3(a0.xy,h.x);vec3 p1=vec3(a0.zw,h.y);vec3 p2=vec3(a1.xy,h.z);vec3 p3=vec3(a1.zw,h.w);
 vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
 vec4 m=max(.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.);m=m*m;
 return 42.*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));}
uniform float uOct;
float fbm(vec3 p,float gain){float s=0.,a=.5;for(int i=0;i<14;i++){if(float(i)>=uOct)break;s+=a*snoise(p);p=p*2.03+vec3(1.7,9.2,3.1);a*=gain;}return s;}
float ridged(vec3 p,float gain){float s=0.,a=.5,w=1.;for(int i=0;i<14;i++){if(float(i)>=uOct)break;float n=1.-abs(snoise(p));n*=n;n*=w;w=clamp(n*1.5,0.,1.);s+=a*n;p=p*2.1+vec3(5.3,1.1,7.7);a*=gain;}return s;}
vec3 hash3(vec3 p){p=vec3(dot(p,vec3(127.1,311.7,74.7)),dot(p,vec3(269.5,183.3,246.1)),dot(p,vec3(113.5,271.9,124.6)));return fract(sin(p)*43758.5453);}
vec2 worley(vec3 p){vec3 i=floor(p),f=fract(p);float d1=9.;float id=0.;for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++){vec3 g=vec3(x,y,z);vec3 o=hash3(i+g);float d=length(g+o-f);if(d<d1){d1=d;id=o.x;}}return vec2(d1,id);}
`;
const VS_BODY = `varying vec3 vObj; varying vec3 vN; varying vec3 vW;
void main(){vObj=position; vN=normalize(mat3(modelMatrix)*normal); vec4 w=modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w;}`;
const LIGHT = `uniform vec3 uSun; uniform vec3 uSunCol; uniform vec3 uOff;
varying vec3 vObj; varying vec3 vN; varying vec3 vW;
float lam(vec3 n){return max(dot(n,normalize(uSun-vW)),0.);} `;
// Rocky planets: continents (fBm) + ridged mountains + biomes by height/latitude/moisture + caps + bump lighting
const FS_ROCKY = NOISE + LIGHT + `
uniform float uSea,uRough,uCS,uMtn,uMoist,uCap; uniform vec3 uC0,uC1,uC2,uC3,uC4,uC5; uniform float uLava; uniform mat3 uRot; uniform float uWet;   // uWet (life viewers only; 0=gallery look): ocean fresnel sheen + broad glint
float height(vec3 p){vec3 q=p*uCS+uOff; float c=fbm(q,uRough)*.9+.5; float m=ridged(q*2.3+3.1,.5);
 return c + uMtn*.45*m*smoothstep(uSea-.05,uSea+.25,c);}
void main(){vec3 p=normalize(vObj); float h=height(p);
 vec3 t1=normalize(cross(p,vec3(0.,1.,.001))), t2=cross(p,t1); float e=.0015;
 float hx=height(normalize(p+e*t1)), hy=height(normalize(p+e*t2));
 float lat=abs(p.y); float moist=clamp(.5+uMoist+.5*fbm(p*3.+uOff.zxy,.5)-.35*abs(lat-.0),0.,1.);
 vec3 col; bool water=h<uSea; float spec=0.;
 if(water){col=mix(uC0,uC1,smoothstep(uSea-.25,uSea,h)); spec=.6;}
 else{float t=(h-uSea)/max(1.15-uSea,.05);
  vec3 low=mix(uC2*vec3(1.,.95,.8),uC3,smoothstep(.2,.7,moist)); // dry plains vs forest by moisture
  col = t<.03?uC2: t<.45?mix(low,uC3*.8,smoothstep(.05,.45,t)) : t<.75?mix(uC3*.8,uC4,smoothstep(.45,.75,t)) : mix(uC4,uC5,smoothstep(.75,.95,t));
  col=mix(col,vec3(.75,.78,.7),smoothstep(.55,.85,lat)*(1.-uLava)*.6);}
 float cap=uCap+.06*snoise(p*6.+uOff); if(lat>cap){col=mix(col,vec3(.93,.96,1.),smoothstep(cap,cap+.03,lat)); spec=.1;}
 vec3 no=p; if(!water){vec3 g=(hx-h)*t1+(hy-h)*t2; no=normalize(p-(g/e)*.22);} vec3 n=normalize(uRot*no);
 vec3 L=normalize(uSun-vW); float d=max(dot(n,L),0.); vec3 V=normalize(cameraPosition-vW);
 float s=pow(max(dot(reflect(-L,normalize(vN)),V),0.),40.)*spec;
 vec3 c=col*(.03+d*uSunCol)+s*uSunCol;
 if(uWet>0.){float fr=pow(1.-max(dot(normalize(vN),V),0.),5.); float wet=water?1.:.0; float g2=pow(max(dot(reflect(-L,normalize(vN)),V),0.),8.);
  c+=uWet*wet*(fr*vec3(.25,.35,.5)*(.3+.7*d)+g2*.18*uSunCol*d); if(!water) c*=1.+uWet*.12*smoothstep(.0,.6,d)*(1.-moist)*vec3(1.,.92,.78)-uWet*.06;}
 if(uLava>.5){float lv=smoothstep(uSea,uSea-.08,h); c+=vec3(1.,.35,.05)*lv*1.4*(.6+.4*snoise(p*20.+uOff));}
 gl_FragColor=vec4(c,1.);}`;
const FS_CLOUD = NOISE + LIGHT + ` uniform float uCover, uTime;
void main(){vec3 p=normalize(vObj); float n=fbm(p*3.+uOff+vec3(uTime*.02,0.,0.),.55)+.35*fbm(p*9.-uOff,.5);
 float a=smoothstep(1.-uCover-.15,1.-uCover+.25,n*.5+.5); float d=lam(normalize(vN)); gl_FragColor=vec4(vec3(1.)*(.05+d*.95),a*.9);}`;
const FS_ATM = `varying vec3 vObj; varying vec3 vN; varying vec3 vW; uniform vec3 uAtm; uniform vec3 uSun;
void main(){vec3 V=normalize(cameraPosition-vW); vec3 n=normalize(vN); float f=pow(1.-abs(dot(V,n)),3.);
 float d=clamp(dot(n,normalize(uSun-vW))+.3,0.,1.); gl_FragColor=vec4(uAtm*f*d*1.6,f*d);}`;
// Gas giants: latitude bands warped by turbulence + an oval storm
const FS_GAS = NOISE + LIGHT + `
uniform float uBands,uTurb,uTime; uniform vec3 uC0,uC1,uC2,uStorm;
void main(){vec3 p=normalize(vObj); float lat=asin(clamp(p.y,-1.,1.)); float lon=atan(p.z,p.x);
 vec2 sd=vec2((lon-uStorm.y)*cos(lat)*.6,(lat-uStorm.x)); float sr=length(sd/vec2(1.6,1.)) / uStorm.z;
 float sw=exp(-sr*sr)*5.; float ca=cos(sw),sa=sin(sw); vec3 q=vec3(ca*p.x-sa*p.z,p.y,sa*p.x+ca*p.z);
 float w=fbm(vec3(q.x*2.,q.y*9.,q.z*2.)+uOff+vec3(uTime*.01,0.,0.),.55)*uTurb;
 float b=sin((lat+w*.15)*uBands); float b2=sin((lat+w*.1)*uBands*2.7+1.3);
 vec3 col=mix(uC0,uC1,b*.5+.5); col=mix(col,uC2,smoothstep(.5,1.,b2)*.5);
 col=mix(col,vec3(.75,.35,.2)*1.1,smoothstep(1.,.6,sr)*.8);
 vec3 n=normalize(vN); float mu=max(dot(n,normalize(cameraPosition-vW)),0.);
 gl_FragColor=vec4(col*(.02+lam(n)*uSunCol)*(.6+.4*mu),1.);}`;
// Moons: Worley craters (bowl + rim) over fBm regolith
const FS_MOON = NOISE + LIGHT + ` uniform float uCr,uDepth; uniform vec3 uCol; uniform mat3 uRot;
float hgt(vec3 p){float h=fbm(p*4.+uOff,.5)*.3; for(int k=0;k<3;k++){float sc=uCr*pow(2.3,float(k)); vec2 w=worley(p*sc+uOff*float(k+1));
 float r=.45; float d=w.x/r; h+= (d<1.? (d*d-1.)*uDepth : exp(-(d-1.)*(d-1.)*30.)*.25*uDepth)/(1.+float(k));} return h;}
void main(){vec3 p=normalize(vObj); float h=hgt(p); vec3 t1=normalize(cross(p,vec3(0.,1.,.001))),t2=cross(p,t1); float e=.004;
 vec3 g=(hgt(normalize(p+e*t1))-h)*t1+(hgt(normalize(p+e*t2))-h)*t2; vec3 n=normalize(uRot*normalize(p-g/e*.04));
 vec3 col=uCol*(.75+.35*h); gl_FragColor=vec4(col*(.02+lam(n)*uSunCol),1.);}`;
// Stars: limb darkening + animated granulation + spots
const FS_STAR = NOISE + `varying vec3 vObj; varying vec3 vN; varying vec3 vW; uniform vec3 uCol,uOff; uniform float uGran,uSpots,uTime;
void main(){vec3 p=normalize(vObj); vec3 V=normalize(cameraPosition-vW); float mu=max(dot(normalize(vN),V),0.);
 vec2 w=worley(p*uGran+uOff+vec3(0.,uTime*.05,0.)); float gr=.75+.35*smoothstep(0.,.8,w.x)*-1.+.3;
 float sp=smoothstep(.55,.75,snoise(p*2.5+uOff+uTime*.003))*uSpots; float ld=1.-.6*(1.-pow(mu,.8));
 gl_FragColor=vec4(uCol*ld*(.85+.3*(1.-w.x))*(1.-sp*.8)*1.25,1.);}`;
const VS_PTS = `attribute float size; attribute vec3 color; varying vec3 vC; uniform float uScale;
void main(){vC=color; vec4 mv=modelViewMatrix*vec4(position,1.); gl_PointSize=clamp(size*uScale/-mv.z,1.,64.); gl_Position=projectionMatrix*mv;}`;
const FS_PTS = `varying vec3 vC; uniform float uAlpha; void main(){vec2 d=gl_PointCoord-.5; float r=dot(d,d)*4.; if(r>1.)discard; gl_FragColor=vec4(vC*exp(-r*3.)*uAlpha,1.);}`;
