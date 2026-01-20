aPOSITION_LOC = 0
aTEXCOORD_LOC = 1
aCOLOR_LOC = 2
aIDTRIANGLE_LOC = 3

vertex_shader = """
#version 430 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec3 aColor;

out vec2 vTexCoord;
out vec3 vColor;
out float vDepth;

uniform float uClickableId;
uniform mat4 uChunk;
uniform mat4 uProj; 
uniform mat4 uView; 
uniform mat4 uTrack;
uniform mat4 uViewCam;


uniform  int resolution_width;
uniform  int resolution_height;

 // Properties
uniform    float pixel_width;
uniform    float pixel_height;
uniform    float focal_length;

// Calibration
uniform    float f;
uniform    float cx; // this is the offset w.r.t. the center
uniform    float cy; // this is the offset w.r.t. the center
uniform    float k1;
uniform    float k2;
uniform    float k3;
uniform    float k4;
uniform    float p1;
uniform    float p2;
uniform    float b1;
uniform    float b2;
uniform    int uMode; // mode: 0-distorted, 1-undistorted 2-distorted project to texture
uniform    int uModeProj;

// near far
uniform float uNear;
uniform float uFar;


vec2 xyz_to_uv(vec3 p){
    float x = p.x/p.z;
    float y = -p.y/p.z;
    float r = sqrt(x*x+y*y);
    float r2 = r*r;
    float r4 = r2*r2;
    float r6 = r4*r2;
    float r8 = r6*r2;

    float A = (1.0+k1*r2+k2*r4+k3*r6   +k4*r8  ); 
    float B = (1.0 /* +p3*r2+p4*r4 */ );

    float xp = x * A+ (p1*(r2+2*x*x)+2*p2*x*y) * B;
    float yp = y * A+ (p2*(r2+2*y*y)+2*p1*x*y) * B;

    float u = resolution_width*0.5+cx+xp*f + xp*b1 + yp*b2;
    float v = resolution_height*0.5+cy+yp*f;

    u /= resolution_width;
    v /= resolution_height;

    return vec2(u,v);
}
void main(void)
{
    //gl_Position = uProj*uView * uModel*uRot*vec4(aPosition, 1.0);
   
    vec4 pos =   uProj*uView*uTrack*uChunk*vec4(aPosition, 1.0) ;
    vTexCoord = aTexCoord;
    vColor = aColor;

    vec3 pos_vs;
    if(uMode == 0){ // metashape projection
        // todo: the depth value is taken from the uProj just to make zbuffering
        // work. to be cleaned up
        pos_vs = (uView*uChunk*vec4(aPosition, 1.0)).xyz;
        vec4 pr_p = uProj*vec4(pos_vs,1.0);
        float focmm = f / resolution_width;    
        if( abs(pos_vs.z ) < uNear)
            gl_Position = vec4(0, 0, 2, 1); // outside clip volume
        else
            gl_Position = vec4(xyz_to_uv(pos_vs)*2.0-1.0, (pos_vs.z-uNear)/(uFar-uNear)*2.0-1.0,1.0);   //to be fixed
        
    }
    else    // opengl projection
    { 
        pos_vs = (uView*uTrack*uChunk*vec4(aPosition, 1.0)).xyz;
        gl_Position = uProj*vec4(pos_vs,1.0);
        if(uModeProj == 1)
            vTexCoord = xyz_to_uv((uViewCam*vec4(aPosition, 1.0)).xyz);
    }
}
"""

fragment_shader = """
#version 460 core
layout(location = 0) out vec4 color;

in vec2 vTexCoord;
in vec3 vColor;
in float vDepth;


uniform sampler2D uColorTex;
uniform int uWriteModelTexCoords;
uniform sampler2D uMasks;
uniform vec3 uColor;

uniform int uColorMode; // 0: vertex color, 1: uniform color, 2: texture

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 col(float t) {
    // Scramble t to make close values result in different hues
    float scatteredT = fract(sin(t * 1234.567 + 4.321) * 43758.5453123);
    
    // Map the scattered value to the hue range
    float hue = fract(scatteredT);
    return hsv2rgb(vec3(hue, 1.0, 1.0));
}

void main()
{
    if(uColorMode == 0)
        color  = vec4(vColor,1.0);
    else
    if(uColorMode == 1)
        color  = vec4(uColor,1.0);
    else
    if(uColorMode == 2)
        color  = vec4(texture(uColorTex,vTexCoord.xy).rgb,1.0);

}
"""

vertex_shader_frame = """
#version 430 core
layout(location = 0) in vec3 aPosition;
layout(location = 2) in vec3 aColor;

out vec3 vColor;

uniform mat4 uProj; 
uniform mat4 uView; 
uniform mat4 uTrack;
uniform float uScale;
void main(void)
{
    vColor = aColor;
    gl_Position = uProj*uView*uTrack*vec4(aPosition*uScale, 1.0);
}
"""

fragment_shader_frame = """
#version 460 core
layout(location = 0) out vec4 color;

in vec3 vColor;

void main()
{
    color  = vec4(vColor,1.0);
}
"""

vertex_shader_clickable = """
#version 430 core
layout(location = 0) in vec3 aPosition;

uniform float uSca;
uniform vec2 uTra;

void main(void)
{
    gl_Position = vec4(aPosition*uSca+vec3(uTra,0.0), 1.0);
}
"""
fragment_shader_clickable = """  
#version 430 core
layout(location = 0) out vec4 ClickableId;
uniform float uClickableId;

void main()
{
    ClickableId = vec4(vec3(uClickableId),1.0);        
}
"""




vertex_shader_fsq = """
#version 430 core
layout(location = 0) in vec3 aPosition;
out vec2 vTexCoord;
uniform float uSca;
uniform vec2 uTra;

void main(void)
{
    if(uSca == 0.0)  
        gl_Position = vec4(aPosition, 1.0);
     else
         gl_Position = vec4(aPosition*uSca+vec3(uTra,0.0), 1.0);
    vTexCoord = aPosition.xy *0.5+0.5;
}
"""
fragment_shader_fsq = """  
#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uColorTex;
uniform sampler2D uMask;

layout(binding = 14)  uniform sampler2D uFluoTex1;  
layout(binding = 15)  uniform sampler2D uFluoTex2;  

uniform ivec2 uOff;
uniform ivec2 uSize;
uniform float uSca;

uniform  int resolution_width;
uniform  int resolution_height;


// colorRamp.glsl
// Valore di input expected in [0.0, 0.2] (valori esterni vengono clampati)
// contrastFactor: 1.0 = nessuna modifica, >1 aumenta contrasto, 0..1 riduce contrasto

vec3 applyContrast(vec3 col, float contrastFactor) {
    // contrast around 0.5: (col - 0.5)*factor + 0.5
    return clamp((col - 0.5) * contrastFactor + 0.5, 0.0, 1.0);
}

vec3 colorRamp01(float t) {
    // t in [0,1] -> più stop: blu -> ciano -> verde -> giallo -> rosso
    // puoi cambiare i colori come preferisci
    vec3 c0 = vec3(0.0, 0.0, 0.6); // blu scuro
    vec3 c1 = vec3(0.0, 0.7, 0.9); // ciano
    vec3 c2 = vec3(0.0, 0.9, 0.3); // verde chiaro
    vec3 c3 = vec3(1.0, 0.85, 0.0); // giallo
    vec3 c4 = vec3(0.85, 0.05, 0.05); // rosso

    // segmenti: [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]
    float s = smoothstep(0.0, 1.0, t); // optional, qui serve solo se vuoi smussare ulteriormente il t
    if (t < 0.25) {
        float u = smoothstep(0.0, 0.25, t);
        return mix(c0, c1, u);
    } else if (t < 0.5) {
        float u = smoothstep(0.25, 0.5, t);
        return mix(c1, c2, u);
    } else if (t < 0.75) {
        float u = smoothstep(0.5, 0.75, t);
        return mix(c2, c3, u);
    } else {
        float u = smoothstep(0.75, 1.0, t);
        return mix(c3, c4, u);
    }
}


vec3 rampForRange(float v, float contrastFactor) {
   // return  vec3(v,v,v)*10.0; // placeholder per debug
    // normalizza v nell'intervallo [0,1]
    float t = clamp((v - 0.0) / (0.1 - 0.0), 0.0, 1.0);

    // ottieni colore dalla rampa
    vec3 col = colorRamp01(t);

    // applica contrasto
   // col = applyContrast(col, contrastFactor);

    return col;
}

void main()
{
   FragColor = vec4(texture(uColorTex, vTexCoord).rgb,1.0);

   ivec2 texel_coord = ivec2(vTexCoord.xy*ivec2(resolution_width,resolution_height)) ;
   texel_coord.y = resolution_height - texel_coord.y; // flip y coordinate
   texel_coord = texel_coord - uOff;
   texel_coord.y = uSize.y - texel_coord.y; // flip y coordinate

   float v = texelFetch(uMask, texel_coord, 0).x; 

   if (uSca != 0.0) 
       FragColor+= vec4(1,1,1, 0.0)*v/uSca; 
    
}
"""

bbox_shader_str = """
#version 460

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in; 


layout(binding = 12)  uniform sampler2D uColorTexture;    // for each pixel of the image, the coordinates in parametric space

layout(std430, binding = 6) buffer bBbox {
    uint bbox[];  
};


uniform  int resolution_width;
uniform  int resolution_height;

void main() {

    if (gl_GlobalInvocationID.x >= resolution_width || gl_GlobalInvocationID.y >= resolution_height)
        return;

    vec4 texel = texelFetch(uColorTexture, ivec2(gl_GlobalInvocationID.xy), 0);

    if (texel.r < 1.0) {
        // bbox[0]: min_x, bbox[1]: min_y, bbox[2]: max_x, bbox[3]: max_y
        atomicMin(bbox[0], gl_GlobalInvocationID.x);
        atomicMin(bbox[1], gl_GlobalInvocationID.y);
        atomicMax(bbox[2], gl_GlobalInvocationID.x);
        atomicMax(bbox[3], gl_GlobalInvocationID.y);
    }
  
}
"""



vertex_shader_basic = """
#version 430 core
layout(location = 0) in vec3 aPosition;

uniform vec2 uPos;

void main(void)
{
    gl_Position =vec4(aPosition + vec3(uPos,0.0),1.0);
    
}
"""

fragment_shader_basic = """
#version 460 core
layout(location = 0) out vec4 color;

uniform vec3 uColor;

void main()
{
    color  = vec4(uColor,1.0);
}
"""
