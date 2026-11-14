// Foam: the white that broken water leaves on its surface, for the eye only (src/world/foam.js).
//
// One number per texel of the map a scene has anyway: of the height field, or of the map of
// waves beyond the solved water. It is made where the water breaks, moves with the water (over
// a height field), and fades. The solver knows nothing of it.
//
// The map: r = how much foam (0 to 1), gb = the velocity of the water it lies on (cells per
// second; the renderer carries its pattern along with it).

const HEAD = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D uF;      // the foam so far
uniform float uDt;         // s
uniform float uKeep;       // what is left of it after a step
uniform float uGain;       // what broken water adds per second
out vec4 o;
`;

export function foamShaders() {
  const S = {};

  // Over a height field: depth h and the velocities u, w through the low faces of each texel
  // (cells per second), also where the 3D solver has the column (there they are the sums over
  // its cells). Water breaks
  //   where the flow runs together faster than a wave would even it out (the foot of a fall, a
  //     jump, a front that has gone over): -div v against sqrt(g / h)
  //   where a steep face moves (a bore)
  //   a little where it shoots over the bed, several times faster than its waves (a chute)
  S.hf = `${HEAD}
uniform sampler2D uH;
uniform sampler2D uBed;    // r = bed (cells), g = 1 where the texel is no part of the height field
uniform float uDx;         // cells per texel
uniform float uG;          // cells / s^2
uniform float uDry;
vec4 st(ivec2 t) {
  ivec2 n = textureSize(uH, 0);
  if (t.x < 0 || t.y < 0 || t.x >= n.x || t.y >= n.y) return vec4(0.0);
  return texelFetch(uH, t, 0);
}
// the surface in texel t; where it is dry or no part of the map: that of the texel asked from
float top(ivec2 t, float mine) {
  ivec2 n = textureSize(uH, 0);
  if (t.x < 0 || t.y < 0 || t.x >= n.x || t.y >= n.y) return mine;
  vec2 b = texelFetch(uBed, t, 0).rg;
  float h = texelFetch(uH, t, 0).r;
  return b.g > 0.5 || h < uDry ? mine : b.r + h;
}
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  vec4 c = texelFetch(uH, t, 0);
  vec2 b = texelFetch(uBed, t, 0).rg;
  if (c.r < uDry || b.g > 0.5) { o = vec4(0.0); return; }
  vec4 ex = st(t + ivec2(1, 0)), ez = st(t + ivec2(0, 1));
  vec2 v = 0.5 * vec2(c.g + ex.g, c.b + ez.b);
  // the foam on the water that is here now
  vec2 n = vec2(textureSize(uH, 0));
  float f = texture(uF, (vec2(t) + 0.5 - v * uDt / uDx) / n).r * uKeep;
  float div = ((ex.g - c.g) + (ez.b - c.b)) / uDx;
  float eta = b.r + c.r;
  vec2 sl = vec2(top(t + ivec2(1, 0), eta) - top(t - ivec2(1, 0), eta), top(t + ivec2(0, 1), eta) - top(t - ivec2(0, 1), eta)) / (2.0 * uDx);
  float speed = length(v), wave = sqrt(uG * c.r);
  // (not a film on a bank: it has no waves to break)
  float deep = smoothstep(0.4, 1.2, c.r);
  float a = deep * (smoothstep(0.25, 1.0, -div / sqrt(uG / max(c.r, 0.5)))
                    + smoothstep(0.25, 0.7, length(sl)) * smoothstep(0.3, 1.0, speed / wave)
                    + 0.3 * smoothstep(2.5, 5.0, speed / wave));
  o = vec4(min(f + uGain * uDt * a, 1.0), v, 1.0);
}`;

  // Over the map of waves: where the water is solved, the map holds the height of its surface
  // and how fast that rises and falls. Water breaks where its surface moves up or down faster
  // than a wave moves it, or stands steeper than a wave does. (It stays where it was made: on
  // an open sea the water goes up and down, not along.)
  S.swell = `${HEAD}
uniform sampler2D uZ;      // h (cells above rest), dh/dt (cells / s)
uniform sampler2D uE;      // r = surface of the solved water, a = 1 where the water is solved
uniform float uDx;
float solvedTop(ivec2 t, float mine) {
  ivec2 n = textureSize(uE, 0);
  if (t.x < 0 || t.y < 0 || t.x >= n.x || t.y >= n.y) return mine;
  vec4 e = texelFetch(uE, t, 0);
  return e.a > 0.5 ? e.r : mine;
}
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  float f = texelFetch(uF, t, 0).r * uKeep;
  vec4 e = texelFetch(uE, t, 0);
  float a = 0.0;
  if (e.a > 0.5) {
    vec2 sl = vec2(solvedTop(t + ivec2(1, 0), e.r) - solvedTop(t - ivec2(1, 0), e.r), solvedTop(t + ivec2(0, 1), e.r) - solvedTop(t - ivec2(0, 1), e.r)) / (2.0 * uDx);
    a = smoothstep(4.0, 16.0, abs(texelFetch(uZ, t, 0).g)) + smoothstep(0.35, 0.9, length(sl));
  }
  o = vec4(min(f + uGain * uDt * a, 1.0), 0.0, 0.0, 1.0);
}`;
  return S;
}
