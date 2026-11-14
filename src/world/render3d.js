// World renderer: ray-marches the liquid stored in bricks. Empty space is skipped on three
// levels: a group of 16 bricks per side without any brick (one fetch of the page table's top
// level), a brick that is not stored, and the part of a stored brick that holds no liquid (the
// per-brick statistics the solver keeps anyway). The ground is traced as a height field,
// obstacles and bodies are sphere-traced analytically.
//
// Water at rest that is not solved (a body of water, src/world/water.js; its bricks are held or
// not stored at all) is drawn from its description: the liquid is what lies below the body's
// surface inside its extent. At rest the surface is where the stored density of the same water
// crosses one half, so a brick that is stored or dropped does not move it. On top of that comes
// the map of the small waves that run on over such water (src/world/swell.js). Solved water is
// drawn from its density, moved up or down by what the map says its surface should differ by:
// nothing inside the solved water, and at its edge whatever makes the two surfaces meet.
//
// The water of a height field (src/world/heightfield.js) that is not solved is traced as what it
// is: a surface over its map (hfSurf). Where the 3D solver has a column, the same water is drawn
// from its density, and the surface of the height field is put where that density would cross
// one half if the column were stored as it stands, so a column that changes hands does not move.
//
// The density is first copied into a pool of its own in which every brick carries a border of
// one cell from its neighbours: inside a brick the GPU's own filtering then gives a continuous
// trilinear density from two fetches.
import { header, prims, TERRAIN, FAR as FAR_FIELD, SWELL, BRICK, MAX_PRIMS, MAX_WATER } from '../shaders/common.js';
import { exactUniforms } from './hull.js';

export { Camera } from '../render3d.js';

const B = BRICK[3];
const AP = B + 2;          // a brick with its border, per axis
const AT = AP * AP;        // width of its tile: AP slices side by side

export class WorldRenderer {
  constructor(glc) {
    this.glc = glc;
    const H = header(3, false, 'sparse');
    this.perRow = Math.min(80, Math.floor(glc.maxTextureSize / AT));
    this.vol = null;
    this.volRows = 0;
    // what the shaders read where a scene has no map of the surface (Swell)
    this.noSwell = { uSwell: glc.texture(1, 1, 'rgba16f', { linear: true }), uSwellMap: [0, 0, 0, 0], uSwellAmp: 0 };
    this.noSwell.uSwell2 = this.noSwell.uSwell;
    glc.clear(this.noSwell.uSwell);

    // density of every stored brick with a border of one cell, tile by tile
    this.pApron = glc.program('worldApron', `${H}
uniform Grid G;
uniform sampler2D uLiq;
uniform int uVolRow;
${FAR_FIELD}
out vec4 o;
// density of cell c; if it is not stored: of the water at rest that belongs there, if any
float cellDens(ivec3 c) {
  vec4 q = fetchC(uLiq, G, c);
  return q.g < 0.0 && FAR_ON ? farState(c).x : max(0.5 - q.r, 0.0);
}
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  ivec2 tile = ivec2(t.x / ${AT}, t.y / ${AP});
  int slot = tile.x + tile.y * uVolRow;
  o = vec4(0.0);
  if (tile.x >= uVolRow || slot >= G.tiles.x * G.tiles.y) return;
  enterSlot(G, slot);
  if (!g_live) return;
  int lx = t.x - tile.x * ${AT};
  ivec3 l = ivec3(lx % ${AP}, t.y - tile.y * ${AP}, lx / ${AP}) - 1;
  ivec3 c = g_frame + l;
  float d = cellDens(c);
  if (FAR_ON && (l.x < 0 || l.x > BM || l.z < 0 || l.z > BM)) {
    // A border cell beside the brick whose water is not solved (held, or not stored): the surface
    // of that water is the map's, not what is stored there. The border continues the brick's own
    // density in a straight line, so its surface runs on to the face of the brick as it arrives.
    int sn = slotOf(G, c);
    if (sn >= G.tiles.x * G.tiles.y || texelFetch(uSl0, ivec2(sn % G.tiles.x, sn / G.tiles.x), 0).w > 1.5) {
      ivec3 e = ivec3(clamp(l.x, 0, BM), l.y, clamp(l.z, 0, BM));     // the brick's cell next to it
      ivec3 i = e + (e - l);                                          // and the one after that
      if (e.x != l.x && e.z != l.z) {
        // (a corner: in a straight line along both axes)
        d = 4.0 * cellDens(g_frame + e) - 2.0 * (cellDens(g_frame + ivec3(e.x, l.y, i.z)) + cellDens(g_frame + ivec3(i.x, l.y, e.z))) + cellDens(g_frame + i);
      } else {
        d = 2.0 * cellDens(g_frame + e) - cellDens(g_frame + i);
      }
      d = max(d, 0.0);
    }
  }
  o = vec4(d, 0.0, 0.0, 0.0);
}`);

    const COMMON = `${H}
uniform Grid G;
uniform sampler2D uVol;       // density with borders (pass worldApron)
uniform vec2 uVolInv;
uniform int uVolRow;
uniform sampler2D uS1;        // per slot: lowest wet cell, number of wet cells
uniform sampler2D uS2;        // per slot: highest wet cell
uniform float uStep;
uniform float uTerMax;        // highest point of the ground
${TERRAIN}
${FAR_FIELD}
${SWELL}
${prims('S', true, true)}
uniform float uKind[${MAX_PRIMS}];    // per primitive: 0 fixed obstacle, 1 scripted body, 2..5 rigid body colours, 6 rock (as the ground where it is steep), 7 hull
const float FAR = 20000.0;

// slot of brick b, -1 if it is not stored (the top level of the page table wraps: the answer
// is checked against the brick the slot belongs to); g_state: its state (uSl0.w)
float g_state = 0.0;
int brickSlot(ivec3 b) {
  int s = pageLookup(G, b);
  if (s < 0) return -1;
  vec4 sl = texelFetch(uSl0, ivec2(s % G.tiles.x, s / G.tiles.x), 0);
  g_state = sl.w;
  return sl.w > 0.5 && ivec3(sl.xyz) == b ? s : -1;
}
// density at l (cells from the corner of the brick in slot s; valid from -0.5 to BS + 0.5)
float densIn(int s, vec3 l) {
  vec2 org = vec2(float((s % uVolRow) * ${AT}), float((s / uVolRow) * ${AP}));
  float u = clamp(l.z + 0.5, 0.0, ${(AP - 1).toFixed(1)});
  float k = min(floor(u), ${(AP - 2).toFixed(1)});
  vec2 xy = clamp(l.xy + 1.0, vec2(0.5), vec2(${(AP - 0.5).toFixed(1)}));
  vec2 uv = (org + vec2(k * ${AP.toFixed(1)}, 0.0) + xy) * uVolInv;
  return mix(texture(uVol, uv).r, texture(uVol, uv + vec2(${AP.toFixed(1)} * uVolInv.x, 0.0)).r, u - k);
}
// the same with the solved water moved to where the map has its surface (org: corner of the brick)
float densLifted(int s, vec3 l, vec3 org) {
  if (uSwellMap.w > 0.5) l.y -= swellAt(org.xz + l.xz).g;
  return densIn(s, l);
}
// Density at p (cells) next to the brick in slot s0 (corner org0): from that brick where p lies in
// it or in its border, else from the brick p is in. Where that one is not solved: 0 above the
// brick of s0 and 1 below it (air, and the water at rest), -1 beside it.
float densNear(int s0, vec3 org0, vec3 p) {
  vec3 l = p - org0;
  if (all(greaterThanEqual(l, vec3(-0.5))) && all(lessThanEqual(l, vec3(float(BS) + 0.5)))) return densIn(s0, l);
  ivec3 b = ivec3(floor(p / float(BS)));
  int s = brickSlot(b);
  if (s >= 0 && g_state < 1.5) return densIn(s, p - vec3(b * BS));
  ivec3 b0 = ivec3(floor(org0 / float(BS) + 0.5));
  if (b.xz != b0.xz) return -1.0;
  return b.y > b0.y ? 0.0 : 1.0;
}
// How much liquid the column at xz holds in the five rows around the height yc (the middle of a
// row): x = cells of it, y = 1 where the column is one that a height describes (full at the
// bottom of the five rows, empty at the top), 0 where it is not or cannot be read.
vec2 columnHeight(int s0, vec3 org0, vec2 xz, float yc) {
  float sum = 0.0, lo = 0.0, hi = 0.0;
  for (int k = -2; k <= 2; k++) {
    float d = densNear(s0, org0, vec3(xz.x, yc + float(k), xz.y));
    if (d < 0.0) return vec2(0.0);
    d = min(d, 1.0);
    if (k == -2) lo = d;
    if (k == 2) hi = d;
    sum += d;
  }
  return vec2(sum, smoothstep(0.7, 0.95, lo) * (1.0 - smoothstep(0.05, 0.3, hi)));
}
vec2 boxHit(vec3 ro, vec3 inv, vec3 lo, vec3 hi) {
  vec3 t0 = (lo - ro) * inv, t1 = (hi - ro) * inv;
  vec3 a = min(t0, t1), b = max(t0, t1);
  return vec2(max(max(a.x, a.y), a.z), min(min(b.x, b.y), b.z));
}
// Where the ray is in water at rest inside the box lo..hi: x = entry, y = exit (x >= y: nowhere).
// A body of water is everything below its surface within its extent; the surface at rest (surf)
// is where the density of the same water in stored cells, interpolated between their centres,
// crosses 0.5. With the map of the waves in use the span is that of the highest the surface can
// be (uSwellAmp above rest); waterEnter() finds the surface itself.
vec2 waterSpan(vec3 ro, vec3 inv, vec3 lo, vec3 hi, out float surf) {
  vec2 best = vec2(1.0e9, -1.0e9);
  vec3 an = vec3(uAnchor);
  float amp = uSwellMap.w > 0.5 ? uSwellAmp : 0.0;
  surf = 0.0;
  for (int i = 0; i < ${MAX_WATER}; i++) {
    if (i >= uWaterN) break;
    vec4 b = uWaterBox[i];
    vec4 L = uWaterLvl[i];
    float fa = min(L.x - L.y, 1.0), fb = max(L.x - L.y - 1.0, 0.0);     // fill of the top liquid row and of the row above
    float top = L.y + 0.5 + (fa - 0.5) / (fa - fb);
    vec3 l = max(lo, vec3(b.x, -1.0e9, b.y) + an);
    vec3 h = min(hi, vec3(b.z, top + amp, b.w) + an);
    if (any(greaterThanEqual(l, h))) continue;
    vec2 t = boxHit(ro, inv, l, h);
    if (t.x < t.y && t.x < best.x) { best = t; surf = top + an.y; }
  }
  return best;
}
// Where a ray that is in the air at t enters the water at rest inside the box lo..hi, which it
// leaves at tOut; -1 if it does not. The surface of that water is the map's.
float waterEnter(vec3 ro, vec3 rd, vec3 inv, vec3 lo, vec3 hi, float t, float tOut) {
  float surf;
  vec2 w = waterSpan(ro, inv, lo, hi, surf);
  float ta = max(w.x, t), tb = min(w.y, tOut);
  if (ta >= tb) return -1.0;
  if (uSwellMap.w < 0.5) return ta;
  // (f: height above the surface)
  vec3 p = ro + rd * ta;
  float fa = p.y - surf - swellAt(p.xz).r;
  if (fa <= 0.0) return ta;
  p = ro + rd * tb;
  float fb = p.y - surf - swellAt(p.xz).r;
  if (fb >= 0.0) return -1.0;
  for (int k = 0; k < 6; k++) {
    float tm = ta + (tb - ta) * fa / (fa - fb);
    p = ro + rd * tm;
    float fm = p.y - surf - swellAt(p.xz).r;
    if (fm > 0.0) { ta = tm; fa = fm; } else { tb = tm; fb = fm; }
  }
  return ta + (tb - ta) * fa / (fa - fb);
}
// ---- the water of a height field where it is not solved
uniform sampler2D uHfGrp;     // per group of 16 x 16 columns of bricks: lowest bed, highest level of its water
uniform ivec4 uHfGrpMap;      // group x and z of texel (0, 0) of uHfGrp, its size
// The sea beyond the map (World.sea): for the eye only, nothing is solved or kept count of
// there. Its surface is level + amp sin(phase - k (xz - origin) . dir), the swell that the
// strip at the edge of the map sends in.
uniform vec4 uHfSea;          // level, amp, k, phase
uniform vec4 uHfSeaDir;       // dir, origin . dir, 1 if there is such a sea
float hfSea(vec2 xz) {
  return uHfSea.x + uHfSea.y * sin(uHfSea.w - uHfSea.z * (dot(xz - vec2(uAnchor.xz), uHfSeaDir.xy) - uHfSeaDir.z)) + float(uAnchor.y);
}
// where the density of a column filled to lv crosses one half between the centres of its cells
float hfIso(float lv) {
  float top = floor(lv - 0.5);
  float a = min(lv - top, 1.0), b = max(lv - top - 1.0, 0.0);
  return top + 0.5 + (a - 0.5) / (a - b);
}
// The surface of the height field's water at xz: x = its height, y > 0 if there is water (the
// share of the texels around that are wet). Between the centres of the texels it runs straight;
// dry ground below the water next to it counts with its own height (a front runs out on the
// ground it runs over), higher dry ground does not (a shore: the water is level up to it, and
// the ground itself is what hides it). Without water: the height of the bed.
// iso: as the density would show it (see hfIso), else the level as it is.
vec2 hfSurf(vec2 xz, bool iso) {
  ivec2 n = textureSize(uHf, 0);
  vec2 q = (xz - vec2(uAnchor.xz) - uHfMap.xy) * uHfMap.z;
  if (q.x < 0.0 || q.y < 0.0 || q.x > float(n.x) || q.y > float(n.y)) return uHfSeaDir.w > 0.5 ? vec2(hfSea(xz), 1.0) : vec2(-1.0e9, 0.0);
  if (iso) {
    // (only where bricks are stored: next to solved water, where the two surfaces have to meet.
    // Elsewhere the level as it is: hfIso() has a kink at every cell of height, which a surface
    // seen at a low angle shows as stripes)
    ivec2 k = ivec2(floor(xz / float(BS))) - uHfColsMap.xy;
    vec4 c = k.x >= 0 && k.y >= 0 && k.x < uHfColsMap.z && k.y < uHfColsMap.w ? texelFetch(uHfCols, k, 0) : vec4(0.0);
    iso = c.a > c.b;
  }
  q -= 0.5;
  vec2 f0 = floor(q), f = q - f0;
  float lv[4], wt[4];
  int kind[4];              // 0: no part of the map, 1: dry, 2: wet
  float sw = 0.0, sl = 0.0;
  for (int k = 0; k < 4; k++) {
    ivec2 t = ivec2(f0) + ivec2(k & 1, k >> 1);
    wt[k] = ((k & 1) == 1 ? f.x : 1.0 - f.x) * ((k >> 1) == 1 ? f.y : 1.0 - f.y);
    kind[k] = 0;
    lv[k] = 0.0;
    if (t.x < 0 || t.y < 0 || t.x >= n.x || t.y >= n.y) continue;
    vec2 b = texelFetch(uHfBed, t, 0).rg;
    if (b.y > 0.5) continue;
    float h = texelFetch(uHf, t, 0).r;
    if (h > 1.0e-3) { kind[k] = 2; lv[k] = iso ? hfIso(b.x + h) : b.x + h; sw += wt[k]; sl += wt[k] * lv[k]; }
    else { kind[k] = 1; lv[k] = b.x; }
  }
  float an = float(uAnchor.y);
  if (sw <= 0.0) {
    float sb = 0.0, wb = 0.0;
    for (int k = 0; k < 4; k++) if (kind[k] == 1) { sb += wt[k] * lv[k]; wb += wt[k]; }
    return vec2(wb > 0.0 ? sb / wb + an : -1.0e9, 0.0);
  }
  float l0 = sl / sw, w = sw;
  for (int k = 0; k < 4; k++) if (kind[k] == 1 && lv[k] < l0) { sl += wt[k] * lv[k]; w += wt[k]; }
  return vec2(sl / w + an, sw);
}
// lowest bed and highest level of the height field's water in the block at lo (a brick, or a
// group of bricks none of which is stored); x > y: it has none
vec2 hfRange(vec3 lo, float size) {
  ivec2 b = ivec2(floor(lo.xz / float(BS) + 0.5));
  vec2 q = vec2(1.0, 0.0);
  if (size > float(BS) + 0.5) {
    ivec2 g = (b >> LS) - uHfGrpMap.xy;
    if (g.x >= 0 && g.y >= 0 && g.x < uHfGrpMap.z && g.y < uHfGrpMap.w) q = texelFetch(uHfGrp, g, 0).rg;
  } else {
    ivec2 k = b - uHfColsMap.xy;
    if (k.x >= 0 && k.y >= 0 && k.x < uHfColsMap.z && k.y < uHfColsMap.w) q = texelFetch(uHfCols, k, 0).rg;
  }
  vec2 r = q.y > q.x ? q + float(uAnchor.y) : vec2(1.0, 0.0);
  if (uHfSeaDir.w > 0.5) {
    // (a block that is not all over the map has the sea beyond it too)
    vec2 a = (lo.xz - vec2(uAnchor.xz) - uHfMap.xy) * uHfMap.z, c = a + size * uHfMap.z;
    vec2 n = vec2(textureSize(uHf, 0));
    if (a.x < 0.0 || a.y < 0.0 || c.x > n.x || c.y > n.y) {
      vec2 sea = vec2(uHfSea.x - uHfSea.y - 1.0e4, uHfSea.x + uHfSea.y) + float(uAnchor.y);
      r = r.x < r.y ? vec2(min(r.x, sea.x), max(r.y, sea.y)) : sea;
    }
  }
  return r;
}
// for a ray in the air: how far p is above the water (<= 0: in it); for one in the water: how
// far below its surface (<= 0: out of it)
float hfSide(vec3 p, bool inside) {
  vec2 s = hfSurf(p.xz, true);
  if (inside) return s.y > 0.0 ? s.x - p.y : -1.0;
  return s.y > 0.0 ? p.y - s.x : max(p.y - s.x, 0.0) + 0.75;
}
// Where between ta and tb the ray crosses the surface of the height field's water (or, inside,
// leaves the water): -1 if it does not. Steps as far as a surface no steeper than 1 allows.
// g_hfLeft: the steps ran out at this distance, before tb (-1: they did not).
float g_hfLeft = -1.0;
float hfTrace(vec3 ro, vec3 rd, float ta, float tb, bool inside) {
  g_hfLeft = -1.0;
  float per = 1.0 / (length(rd.xz) + abs(rd.y));
  float t = ta;
  float g = hfSide(ro + rd * t, inside);
  // (inside: the caller has made sure of it; the ray may start on the surface itself)
  if (g <= 0.0) { if (!inside) return ta; g = 0.05; }
  for (int k = 0; k < 40; k++) {
    if (t >= tb) return -1.0;
    float tn = min(t + clamp(g * per, 0.3 + 0.003 * t, 12.0 + 0.01 * t), tb);
    float gn = hfSide(ro + rd * tn, inside);
    if (gn <= 0.0) {
      float a = t, b = tn, ga = g, gb = gn;
      for (int j = 0; j < 5; j++) {
        // (where the water just ends there is no slope to go by)
        float m = gb > -0.5 ? a + (b - a) * ga / (ga - gb) : 0.5 * (a + b);
        float gm = hfSide(ro + rd * m, inside);
        if (gm > 0.0) { a = m; ga = gm; } else { b = m; gb = gm; }
      }
      return gb > -0.5 ? a + (b - a) * ga / (ga - gb) : b;
    }
    t = tn;
    g = gn;
  }
  if (t < tb) g_hfLeft = t;
  return -1.0;
}
// First place after t0 where the density crosses 0.5, away from the side the ray starts on
// (inside: in liquid); -1 if there is none before tEnd. Adds up the density met in air (mist)
// and, with uOverlay, lines along the edges of the stored bricks the ray enters.
// g_mapped: the place found is on water that is not solved (its surface is the map's).
//
// Density in the air is spray, except the little there is of it on every surface: between the
// cell that is full and the one that is empty the density passes through everything below one
// half. Two rules keep that out of the mist. Density with liquid a cell and a half below it is
// the top of that liquid. And what is met in one stretch right up to the place where the ray
// enters the liquid is kept apart (g_run, begun at g_runFrom): the caller counts it only where
// the stretch is longer than a surface is thick. (One allowance for every surface, taken off
// the sum, left a grain on all solved water that had moved: the sum is taken in steps, and came
// out above the allowance for one pixel and below it for the next.)
uniform int uOverlay;
bool g_mapped = false;
float g_run = 0.0, g_runFrom = 0.0;
float crossing(vec3 ro, vec3 rd, float t0, float tEnd, bool inside, float jitter, inout float mist, inout vec4 lines) {
  vec3 inv = 1.0 / rd;
  float t = t0;
  float prev = inside ? 1.0 : 0.0;
  float amp = uSwellMap.w > 0.5 ? uSwellAmp : 0.0;
  float run = 0.0, runFrom = t0;
  g_mapped = false;
  g_run = 0.0;
  g_runFrom = t0;
  for (int i = 0; i < 220; i++) {
    if (t >= tEnd) break;
    vec3 p = ro + rd * (t + 2.0e-3);
    ivec3 b = ivec3(floor(p / float(BS)));
    float size = float(BS);
    int s = -1;
    if (texelFetch(uPg0, topTexel(b, G.top, G.topTile), 0).r < 0.0) size = float(BS << LS);
    else s = brickSlot(b);
    vec3 lo = floor(p / size) * size;
    float tOut = max(boxHit(ro, inv, lo, lo + size).y, t + 2.0e-3);
    float tA = t, tB = tOut;
    bool stored = s >= 0;
    if (s >= 0) {
      ivec2 st = ivec2(s % G.tiles.x, s / G.tiles.x);
      vec4 w1 = texelFetch(uS1, st, 0);
      bool heldB = g_state > 1.5;
      if (uOverlay == 1 && lines.a < 0.9) {
        vec3 q = abs(fract((ro + rd * t) / size + 0.5) - 0.5) * size;      // distance to the brick's faces
        float w = 0.05 + 0.0012 * t;
        float e = float(q.x < w) + float(q.y < w) + float(q.z < w);
        if (e > 1.5) {
          // solved with liquid, solved without (the margin), held (water at rest)
          vec3 c = heldB ? vec3(1.0, 0.66, 0.22) : w1.w > 0.5 ? vec3(0.25, 0.85, 1.0) : vec3(0.75, 0.75, 0.8);
          float a = (heldB ? 0.30 : w1.w > 0.5 ? 0.40 : 0.16) * (1.0 - lines.a);
          lines += vec4(c * a, a);
        }
      }
      // (a held brick holds the water at rest and nothing else: it is drawn as the body of water it is part of)
      if (heldB) { s = -1; stored = false; }
      else if (w1.w < 0.5) s = -1;
      else {
        vec2 tw = boxHit(ro, inv, lo + w1.xyz - vec3(1.5, 1.5 + amp, 1.5), lo + texelFetch(uS2, st, 0).xyz + vec3(2.5, 2.5 + amp, 2.5));
        tA = max(t, tw.x); tB = min(tOut, tw.y);
        if (tA >= tB) s = -1;
      }
    }
    if (s < 0) {
      mist += run;
      run = 0.0;
      if (!stored && uHfMap.w > 0.5) {
        // nothing solved here: the water of the height field, if it has any in this block
        vec2 yr = hfRange(lo, size);
        // (the range is a few steps old: a bore that has come into the block since stands
        // higher than it says, and was drawn with holes in its crest)
        float y0 = max(lo.y, yr.x - 1.0), y1 = min(lo.y + size, yr.y + 8.0);
        bool any = yr.x < yr.y && y0 < y1;
        vec2 tw = any ? boxHit(ro, inv, vec3(lo.x, y0, lo.z), vec3(lo.x + size, y1, lo.z + size)) : vec2(1.0, 0.0);
        float wa = max(tw.x, t), wb = min(min(tw.y, tOut), tEnd);
        if (inside) {
          // (the block was picked 2e-3 further along the ray; and the ray may start on the surface itself)
          if (wa >= wb || wa > t + 3.0e-3 || hfSide(ro + rd * (t + 0.05), true) <= -0.1) return t;
          float te = hfTrace(ro, rd, t + 0.05, wb, true);
          if (te >= 0.0) return te;
          if (g_hfLeft >= 0.0) { t = g_hfLeft; continue; }
          if (wb < min(tOut, tEnd) - 1.0e-3) return wb;
          prev = 1.0;
          t = tOut;
          continue;
        }
        if (wa < wb) {
          float te = hfTrace(ro, rd, wa, wb, false);
          if (te >= 0.0) { g_mapped = true; return te; }
          if (g_hfLeft >= 0.0) { prev = 0.0; t = g_hfLeft; continue; }
        }
      } else if (!stored && uWaterN > 0) {
        // nothing solved here: the water at rest of the bodies of water, if this is in one
        if (inside) {
          float surf;
          vec2 w = waterSpan(ro, inv, lo, lo + size, surf);
          float wa = max(w.x, t), wb = min(w.y, tOut);
          // (the block was picked 2e-3 further along the ray: past a corner it is entered that much later)
          if (wa >= wb || wa > t + 3.0e-3) return t;
          if (wb < min(tOut, tEnd) - 1.0e-3) return wb;      // out through the surface or the side of the body of water
          prev = 1.0;
          t = tOut;
          continue;
        }
        float te = waterEnter(ro, rd, inv, lo, lo + size, t, tOut);
        if (te >= 0.0 && te < tEnd) { g_mapped = true; return te; }
      }
      if (inside) return t;       // liquid ends where the wet cells do
      prev = 0.0;
      t = tOut;
      continue;
    }
    if (inside && tA > t + 1.0e-3) return t;
    if (tA > t) { prev = 0.0; mist += run; run = 0.0; }
    float tt = tA + (i == 0 ? jitter * uStep : 0.0);
    float tStop = min(tB, tEnd);
    for (int k = 0; k < 48; k++) {
      float tn = min(tt + uStep, tStop);
      vec3 l = ro + rd * tn - lo;
      float d = densLifted(s, l, lo);
      if ((d > 0.5) != inside) {
        g_run = run;
        g_runFrom = runFrom;
        return tt + (tn - tt) * clamp((0.5 - prev) / (d - prev), 0.0, 1.0);
      }
      if (!inside) {
        if (d > 0.01) {
          if (run <= 0.0) runFrom = tt;
          if (densLifted(s, l - vec3(0.0, 1.5, 0.0), lo) < 0.5) run += d * (tn - tt);
        } else {
          mist += run;
          run = 0.0;
        }
      }
      prev = d;
      tt = tn;
      if (tt >= tStop - 1.0e-4) break;
    }
    if (tB < tOut) { if (inside) return tB; prev = 0.0; mist += run; run = 0.0; }
    t = tOut;
  }
  mist += run;
  if (!inside && uWaterN > 0) {
    // out of steps (a long way over stored bricks): what is left is the water at rest
    float te = waterEnter(ro, rd, inv, vec3(-1.0e9), vec3(1.0e9), t, tEnd);
    if (te >= 0.0 && te < tEnd) { g_mapped = true; return te; }
  }
  if (!inside && uHfMap.w > 0.5 && t < tEnd) {
    // out of steps (a long way low over the water of a height field, brick by brick where
    // some are stored): what is left is that water, and if the steps run out once more the
    // ray is as good as on it. (Without this: holes behind the crests, with the sea bed in them.)
    float te = hfTrace(ro, rd, t, tEnd, false);
    if (te < 0.0 && g_hfLeft >= 0.0 && hfSide(ro + rd * g_hfLeft, false) < 4.0) te = g_hfLeft;
    if (te >= 0.0) { g_mapped = true; return te; }
  }
  return -1.0;
}
// the ground: distance along the ray, -1 if it is not hit before tmax
float traceTerrain(vec3 ro, vec3 rd, float tmax) {
  if (uTerMap.w < 0.5) return -1.0;
  float t = 0.0;
  if (ro.y > uTerMax + 1.0) {
    if (rd.y >= -1.0e-5) return -1.0;
    t = (uTerMax + 1.0 - ro.y) / rd.y;
  }
  float ta = t, d = sdTerrain(ro + rd * t);
  if (d < 0.0) return t;
  for (int i = 0; i < 140; i++) {
    t += clamp(d * 0.8, 0.2, 40.0) + t * 0.004;
    if (t > tmax) return -1.0;
    vec3 p = ro + rd * t;
    if (rd.y > 0.0 && p.y > uTerMax + 1.0) return -1.0;
    d = sdTerrain(p);
    if (d < 0.0) {
      float tb = t;
      for (int k = 0; k < 7; k++) {
        float m = 0.5 * (ta + tb);
        if (sdTerrain(ro + rd * m) < 0.0) tb = m; else ta = m;
      }
      return 0.5 * (ta + tb);
    }
    ta = t;
  }
  return -1.0;
}
// the primitive idx, if it is a hull that is drawn from its drawing (else -1)
int drawnHull(float idx) {
  for (int i = 0; i < ${MAX_PRIMS}; i++) {
    if (i >= uSN) break;
    if (float(i) != idx) continue;
    float w = abs(uSA[i].w);
    if (w > 2.5 && w < 3.5 && uSXN[int(uSB[i].w + 0.5)].x > 0) return i;
  }
  return -1;
}
float traceSolid(vec3 ro, vec3 rd, float tmax, out float idx) {
  idx = -1.0;
  if (uSN == 0) return -1.0;
  float t = 0.0, tp = 0.0, dp = -1.0;
  for (int i = 0; i < 80; i++) {
    vec2 s = sdS(ro + rd * t);
    if (s.x < 0.02) { idx = s.y; return t; }
    float st = max(s.x, 0.02);
    // Two samples in a row that come down on a face: the line through them meets it. (No further
    // than two and a half times the distance: past an edge it is not a line. A hull drawn from
    // its drawing is the reason: there every sample counts, see exactCode in src/shaders/common.js.)
    if (dp > s.x && s.x < 0.6) st = min(s.x * (t - tp) / (dp - s.x), 2.5 * s.x);
    tp = t;
    dp = s.x;
    t += st;
    if (t > tmax) break;
  }
  return -1.0;
}
// nearest thing that is not liquid: x = distance (-1: none), y = -1 for the ground, else the primitive
vec2 traceOpaque(vec3 ro, vec3 rd) {
  float idx;
  float tS = traceSolid(ro, rd, FAR, idx);
  // (0: the ray starts on a solid, as one does that goes on from the liquid that lies against it)
  float tT = traceTerrain(ro, rd, tS >= 0.0 ? tS : FAR);
  if (tT >= 0.0) return vec2(tT, -1.0);
  return vec2(tS, idx);
}`;

    // 1x1 picking pass: distances along the pointer ray to the liquid surface and to the ground / an obstacle
    this.pPick = glc.program('worldPick', `${COMMON}
uniform vec3 uRo;
uniform vec3 uRd;
out vec4 o;
void main() {
  vec3 rd = uRd + vec3(equal(uRd, vec3(0.0))) * 1.0e-6;
  vec2 op = traceOpaque(uRo, rd);
  float mist = 0.0;
  vec4 lines = vec4(0.0);
  o = vec4(crossing(uRo, rd, 0.0, op.x > 0.0 ? op.x : FAR, false, 0.0, mist, lines), op.x, 0.0, 1.0);
}`);
    this.pickTex = glc.texture(1, 1, 'rgba32f');
    this.pickBusy = false;

    this.pShade = glc.program('worldShade', `${COMMON}
uniform vec2 uRes;
uniform vec3 uCamPos;
uniform mat3 uCamRot;         // columns: right, up, forward
uniform float uFov;
uniform vec3 uLight;
uniform float uMist;
uniform vec4 uCursor;         // xyz position, w radius (0 = hidden)
out vec4 o;

const float IOR = 1.33;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec3 sky(vec3 d) {
  float t = clamp(d.y, -1.0, 1.0);
  vec3 c = t > 0.0 ? mix(vec3(0.66, 0.74, 0.84), vec3(0.16, 0.32, 0.62), pow(t, 0.5))
                   : mix(vec3(0.66, 0.74, 0.84), vec3(0.30, 0.32, 0.34), pow(-t, 0.3));
  float s = max(dot(d, uLight), 0.0);
  return c + vec3(1.0, 0.93, 0.80) * (pow(s, 900.0) * 6.0 + pow(s, 24.0) * 0.18);
}
// ---- what no cell is small enough to hold: ripples, the sun on them, wet ground along the water
uniform float uTime;          // s
uniform float uDetail;        // 1: draw these, 0: the water and the ground as the solver has them
float hashI(ivec2 p) {
  uint h = uint(p.x) * 0x27d4eb2dU ^ uint(p.y) * 0x165667b1U;
  h ^= h >> 15; h *= 0x2c1b3c6dU; h ^= h >> 12; h *= 0x297a2d39U; h ^= h >> 15;
  return float(h >> 8) * (1.0 / 16777216.0);
}
// smooth noise between the points of a lattice: x = value (0 to 1), yz = its slope
vec3 noiseD(vec2 x) {
  vec2 fl = floor(x), f = x - fl;
  ivec2 i = ivec2(fl);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
  float a = hashI(i), b = hashI(i + ivec2(1, 0)), c = hashI(i + ivec2(0, 1)), d = hashI(i + ivec2(1, 1));
  return vec3(a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y, du * (vec2(b - a, c - a) + (a - b - c + d) * u.yx));
}
// Slope (d/dx, d/dz) of the ripples at xz: three sets, 10, 4.5 and 2 cells long, each two trains
// that cross (one alone slides over the water like a pattern on a cloth). foot: the size of a
// pixel on the water there, in cells: a set that a pixel cannot show is left out.
vec2 ripples(vec2 xz, float foot) {
  vec2 sl = vec2(0.0);
  for (int k = 0; k < 3; k++) {
    float len = k == 0 ? 10.0 : k == 1 ? 4.5 : 2.0;
    float w = 1.0 - smoothstep(0.25 * len, 0.7 * len, foot);
    if (w <= 0.0) continue;
    float a = k == 0 ? 0.6 : k == 1 ? 1.9 : 3.7;
    mat2 R = mat2(cos(a), sin(a), -sin(a), cos(a));
    vec2 q = R * xz * (2.0 / len);
    float speed = (k == 0 ? 0.55 : k == 1 ? 0.8 : 1.1) * uTime;
    vec2 g = noiseD(q + vec2(speed, 0.3 * speed)).yz + noiseD(q * 1.13 + vec2(17.0 - 0.7 * speed, 5.0 + 0.9 * speed)).yz * 1.13;
    sl += (k == 0 ? 0.030 : k == 1 ? 0.026 : 0.020) * w * (g * R);
  }
  return sl;
}
// The sun on ripples too small to draw. A lattice of facets, one to a square of grain cells,
// each with a slope of its own (spread like those of a sea in a light breeze: variance 0.012)
// that turns slowly: a facet sends the sun this way while its slope is the one that takes
// (need). Two lattices, the grain a few pixels, so that a point of light is never smaller than a
// pixel. Mean over many facets: exp(-need.need / 0.014).
float facets(vec2 xz, float grain, vec2 need) {
  vec2 q = xz / grain, fl = floor(q);
  ivec2 i = ivec2(fl);
  float h1 = hashI(i), h2 = hashI(i + ivec2(91, 37));
  float spot = 1.0 - smoothstep(0.10, 0.30, length(q - fl - 0.3 - 0.4 * vec2(h1, h2)));
  if (spot <= 0.0) return 0.0;
  float h3 = hashI(i + ivec2(-53, 113)), h4 = hashI(i + ivec2(7, -201));
  float a = 6.2832 * (h4 + uTime * (0.1 + 0.25 * h1));
  vec2 d = sqrt(-0.012 * log(max(h3, 1.0e-4))) * vec2(cos(a), sin(a)) - need;
  // (mean of the spot: 0.1319, of the flash: 0.002 / 0.014)
  return spot * exp(-dot(d, d) / 0.002) / 0.01885;
}
float sparkle(vec2 xz, float foot, vec2 need) {
  float lod = log2(max(foot * 4.0 / 0.6, 1.0)), l0 = floor(lod);
  float g = 0.6 * exp2(l0);
  return mix(facets(xz, g, need), facets(xz, 2.0 * g, need), lod - l0);
}
// ---- foam (src/world/foam.js): r = how much, gb = the velocity of the water it lies on
uniform sampler2D uFoam;
uniform vec4 uFoamMap;        // x, z of the corner of the map (cells), 1 / its size along x and z (0: none)
float foamPattern(vec2 q) {
  return 0.40 * noiseD(q * 0.21).x + 0.36 * noiseD(q * 0.6 + 7.0).x + 0.24 * noiseD(q * 1.7 + 31.0).x;
}
// The share of the water at xz that foam covers. A little foam is flecks, more of it grows
// together. The pattern goes with the water: two copies, each carried along for a while and
// then set back, one fading in as the other fades out. foot: the size of a pixel on the water
// there (cells); where it is larger than the flecks, what they cover on the average.
float foamCover(vec2 xz, float foot) {
  if (uFoamMap.z <= 0.0) return 0.0;
  vec2 uv = (xz - uFoamMap.xy) * uFoamMap.zw;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 0.0;
  vec3 f = texture(uFoam, uv).rgb;
  if (f.r < 0.02) return 0.0;
  float far = smoothstep(0.5, 2.0, foot);
  if (far >= 1.0) return 0.85 * f.r;
  const float T = 1.8;
  float a = fract(uTime / T), b = fract(a + 0.5);
  float p = mix(foamPattern(xz - f.gb * ((a - 0.5) * T)), foamPattern(xz - f.gb * ((b - 0.5) * T) + 13.7), abs(2.0 * a - 1.0));
  float th = mix(0.80, 0.34, f.r);
  // (thin where there is little of it)
  return mix(smoothstep(th - 0.10, th + 0.12, p) * mix(0.6, 1.0, f.r), 0.85 * f.r, far);
}
// mapped: p is on water that is not solved
vec3 liquidNormal(vec3 p, bool mapped) {
  ivec3 b = ivec3(floor(p / float(BS)));
  int s = mapped ? -1 : brickSlot(b);
  if (s < 0 || g_state > 1.5) {
    if (uHfMap.w > 0.5) {
      // (the level as it is: hfIso() has a ripple of its own, one per cell of height)
      float e = 0.5 / uHfMap.z;
      float c = hfSurf(p.xz, false).x;
      vec2 xa = hfSurf(p.xz - vec2(e, 0.0), false), xb = hfSurf(p.xz + vec2(e, 0.0), false);
      vec2 za = hfSurf(p.xz - vec2(0.0, e), false), zb = hfSurf(p.xz + vec2(0.0, e), false);
      // (towards dry ground: the slope on the side that has water)
      float dx = xa.y > 0.0 && xb.y > 0.0 ? (xb.x - xa.x) / (2.0 * e) : xb.y > 0.0 ? (xb.x - c) / e : xa.y > 0.0 ? (c - xa.x) / e : 0.0;
      float dz = za.y > 0.0 && zb.y > 0.0 ? (zb.x - za.x) / (2.0 * e) : zb.y > 0.0 ? (zb.x - c) / e : za.y > 0.0 ? (c - za.x) / e : 0.0;
      return normalize(vec3(-dx, 1.0, -dz));
    }
    vec4 m = swellAt(p.xz);
    return normalize(vec3(-m.b, 1.0, -m.a));
  }
  vec3 org = vec3(b * BS);
  vec3 l = p - org;
  float own = swellOwn(p.xz);
  // Where the liquid lies as a layer (full below, empty above), its slope is that of the amount
  // of liquid per column: the height of the water. The gradient of the density is not: between
  // the middles of two rows the surface drawn rises at half to twice the rate the water does,
  // row after row, and a gentle slope was lit in steps, one per cell of height. (Solved water
  // had a grain that the water around it, drawn from a height, has not.) Elsewhere (spray, a
  // wave that turns over, next to a body) the gradient is what there is.
  bool lifted = own < 0.999;      // (the map moves the density only over the last solved brick)
  float yc = floor(p.y - (lifted ? swellAt(p.xz).g : 0.0)) + 0.5;
  const float ec = 1.0;
  vec2 xa = columnHeight(s, org, p.xz - vec2(ec, 0.0), yc), xb = columnHeight(s, org, p.xz + vec2(ec, 0.0), yc);
  vec2 za = columnHeight(s, org, p.xz - vec2(0.0, ec), yc), zb = columnHeight(s, org, p.xz + vec2(0.0, ec), yc);
  float layer = min(min(xa.y, xb.y), min(za.y, zb.y));
  if (layer > 0.0 && uSN > 0) layer *= smoothstep(1.5, 3.0, sdS(p).x);
  vec3 nd = vec3(0.0, 1.0, 0.0);
  if (layer > 0.0) {
    vec2 sl = vec2(xb.x - xa.x, zb.x - za.x);
    if (lifted) sl += vec2(swellAt(p.xz + vec2(ec, 0.0)).g - swellAt(p.xz - vec2(ec, 0.0)).g, swellAt(p.xz + vec2(0.0, ec)).g - swellAt(p.xz - vec2(0.0, ec)).g);
    nd = normalize(vec3(-sl.x, 2.0 * ec, -sl.y));
  }
  if (layer < 0.999) {
    const float e = 0.5;
    vec3 g = vec3(densLifted(s, l + vec3(e, 0.0, 0.0), org) - densLifted(s, l - vec3(e, 0.0, 0.0), org),
                  densLifted(s, l + vec3(0.0, e, 0.0), org) - densLifted(s, l - vec3(0.0, e, 0.0), org),
                  densLifted(s, l + vec3(0.0, 0.0, e), org) - densLifted(s, l - vec3(0.0, 0.0, e), org));
    float n = length(g);
    nd = normalize(mix(n > 1.0e-5 ? -g / n : vec3(0.0, 1.0, 0.0), nd, layer));
  }
  // Over the last brick before water that is not solved the surface drawn is the map's (the
  // density, moved to where the map has it), and so is its slope: the two sides of the edge are
  // then lit by the same rule. (Only where the surface lies flat: spray is no part of the map.)
  if (own < 0.999) {
    vec4 m = swellAt(p.xz);
    nd = normalize(mix(nd, normalize(vec3(-m.b, 1.0, -m.a)), (1.0 - own) * smoothstep(0.5, 0.9, nd.y)));
  }
  return nd;
}
// how much of the sun reaches p past the ground (obstacles and liquid cast no shadow)
float groundShadow(vec3 p) {
  if (uTerMap.w < 0.5) return 1.0;
  float res = 1.0;
  float t = 1.5;
  for (int i = 0; i < 24; i++) {
    vec3 q = p + uLight * t;
    if (q.y > uTerMax) break;
    float d = sdTerrain(q);
    res = min(res, 5.0 * d / t);
    if (res < 0.02) break;
    t += clamp(d, 1.0, 24.0);
  }
  return clamp(res, 0.0, 1.0);
}
// How wet the ground at p is (0 to 1): under water, and for a cell or two of height above it
// along the shore. slope: that of the ground at p (the water is looked for downhill of p).
float wetGround(vec3 p, vec2 slope) {
  float wet = 0.0;
  if (uHfMap.w > 0.5) {
    float g = max(length(slope), 1.0e-4);
    for (int k = 0; k < 4; k++) {
      // (here, then where the ground is 0.7, 1.4 and 2.1 cells lower, as far as 64 cells away)
      float d = min(0.7 * float(k) / g, 64.0);
      vec2 sf = hfSurf(p.xz - slope * (d / g), false);
      if (sf.y > 0.0) wet = max(wet, (1.0 - smoothstep(0.3, 2.0, p.y - sf.x)) * (1.0 - smoothstep(40.0, 64.0, d)));
      if (wet >= 1.0 || d >= 64.0) break;
    }
  } else {
    vec3 an = vec3(uAnchor);
    for (int i = 0; i < ${MAX_WATER}; i++) {
      if (i >= uWaterN) break;
      vec4 b = uWaterBox[i];
      vec3 r = p - an;
      if (r.x < b.x || r.x > b.z || r.z < b.y || r.z > b.w) continue;
      wet = max(wet, 1.0 - smoothstep(0.3, 2.0, r.y - uWaterLvl[i].x - swellAt(p.xz).r));
    }
  }
  if (wet >= 1.0) return wet;
  // solved water: liquid (or spray that lands) on the ground here, or liquid beside it
  ivec3 b = ivec3(floor(p / float(BS)));
  int sb = brickSlot(b);
  if (sb < 0 || g_state > 1.5) return wet;
  vec3 org = vec3(b * BS);
  float d = densNear(sb, org, p + vec3(0.0, 0.7, 0.0));
  for (int k = 0; k < 4; k++) {
    vec2 o = k == 0 ? vec2(2.0, 0.0) : k == 1 ? vec2(-2.0, 0.0) : k == 2 ? vec2(0.0, 2.0) : vec2(0.0, -2.0);
    vec3 q = p + vec3(o.x, 0.0, o.y);
    // (beside, at the height of the ground there or just below this point, whichever is higher)
    q.y = max(terrainAt(q).x + 0.6, p.y - 1.5);
    if (q.y < p.y + 0.5) d = max(d, 0.8 * densNear(sb, org, q));
  }
  return max(wet, smoothstep(0.05, 0.5, d));
}
// colour of the ground or of an obstacle at p
vec3 shadeOpaque(vec3 p, float idx, float t) {
  vec3 n, base;
  if (idx < 0.0) {
    vec3 h = terrainAt(p);
    n = normalize(vec3(-h.y, 1.0, -h.z));
    float steep = 1.0 - n.y;
    float patchy = fract(sin(dot(floor(p.xz / 24.0), vec2(12.9898, 78.233))) * 43758.5453);
    vec3 soil = mix(vec3(0.21, 0.17, 0.10), vec3(0.10, 0.19, 0.06), 0.35 + 0.4 * patchy);
    // rock where it is steep, in layers
    float layer = fract(sin(floor(p.y / 5.0 + 0.6 * sin(p.x * 0.05) + 0.6 * sin(p.z * 0.07)) * 12.9898) * 43758.5453);
    base = mix(soil, vec3(0.17, 0.155, 0.14) * (0.8 + 0.4 * layer), smoothstep(0.10, 0.40, steep));
    // contour lines every 8 cells of height, a faint grid every 8 cells (one brick)
    float c = abs(fract(p.y / 8.0 + 0.5) - 0.5) * 8.0;
    base *= 1.0 - 0.16 * (1.0 - smoothstep(0.0, 0.25 + 0.0015 * t, c)) * smoothstep(0.03, 0.2, steep);
    vec2 g = abs(fract(p.xz / 8.0 + 0.5) - 0.5) * 8.0;
    base *= 1.0 - 0.10 * (1.0 - smoothstep(0.0, 0.12 + 0.0015 * t, min(g.x, g.y))) * (1.0 - smoothstep(0.1, 0.4, steep));
    // wet ground is darker
    if (uDetail > 0.5) base *= 1.0 - 0.36 * wetGround(p, h.yz);
  } else {
    // a hull that is drawn from its drawing has its normal from there
    int at = drawnHull(idx);
    for (int i = 0; i < ${MAX_PRIMS}; i++) {
      if (i != at) continue;
      n = sdSExactN(int(uSB[i].w + 0.5), uSR[i] * (p - uSA[i].xyz)) * uSR[i];
      if (uSA[i].w < 0.0) n = -n;
    }
    if (at < 0) {
      const float e = 0.05;
      n = normalize(vec3(sdS(p + vec3(e, 0.0, 0.0)).x - sdS(p - vec3(e, 0.0, 0.0)).x,
                         sdS(p + vec3(0.0, e, 0.0)).x - sdS(p - vec3(0.0, e, 0.0)).x,
                         sdS(p + vec3(0.0, 0.0, e)).x - sdS(p - vec3(0.0, 0.0, e)).x));
    }
    float kind = uKind[int(idx)];
    base = kind < 0.5 ? vec3(0.50, 0.44, 0.40)
      : kind < 1.5 ? vec3(0.62, 0.20, 0.14)
      : kind < 2.5 ? vec3(0.95, 0.60, 0.20)
      : kind < 3.5 ? vec3(0.90, 0.33, 0.35)
      : kind < 4.5 ? vec3(0.42, 0.78, 0.50)
      : kind < 5.5 ? vec3(0.68, 0.55, 0.93)
      : kind > 6.5 ? vec3(0.46, 0.21, 0.16)
      : vec3(0.17, 0.155, 0.14) * (0.8 + 0.4 * fract(sin(floor(p.y / 5.0 + 0.6 * sin(p.x * 0.05) + 0.6 * sin(p.z * 0.07)) * 12.9898) * 43758.5453));
  }
  float dif = max(dot(n, uLight), 0.0) * groundShadow(p + n * 0.3);
  vec3 c = base * (0.26 + 0.20 * (n.y * 0.5 + 0.5) + 1.0 * dif);
  return mix(c, sky(vec3(0.0, 0.02, 1.0)), 1.0 - exp(-t / 12000.0));
}

void main() {
  vec2 ndc = (gl_FragCoord.xy / uRes) * 2.0 - 1.0;
  vec3 ro = uCamPos;
  vec3 rd = normalize(uCamRot * vec3(ndc.x * uFov * uRes.x / uRes.y, ndc.y * uFov, 1.0));
  rd += vec3(equal(rd, vec3(0.0))) * 1.0e-6;
  float jitter = hash(gl_FragCoord.xy);
  float mist = 0.0, none = 0.0;
  vec4 lines = vec4(0.0), noLines = vec4(1.0);

  vec2 op = traceOpaque(ro, rd);
  float tEnd = op.x > 0.0 ? op.x : FAR;
  float tw = crossing(ro, rd, 0.0, tEnd, false, jitter, mist, lines);
  vec3 col;
  if (tw >= 0.0) {
    vec3 ph = ro + rd * tw;
    vec3 n = liquidNormal(ph, g_mapped);
    // the size of a pixel on the water, along the line of sight
    float foot = tw * uFov * 2.0 / uRes.y / max(abs(dot(rd, n)), 0.08);
    if (uDetail > 0.5 && n.y > 0.3) {
      // ripples: on water that lies flat (a sheet that falls has none)
      vec2 rp = ripples(ph.xz, foot) * smoothstep(0.3, 0.7, n.y);
      n = normalize(vec3(n.x - rp.x * n.y, n.y, n.z - rp.y * n.y));
    }
    if (dot(n, rd) > 0.0) n = -n;
    float c = clamp(dot(-rd, n), 0.0, 1.0);
    float F = 0.02 + 0.98 * pow(1.0 - c, 5.0);
    col = F * sky(reflect(rd, n));
    if (uDetail > 0.5 && n.y > 0.3) {
      // the sun on the ripples that are too small to draw: see sparkle()
      vec3 hv = normalize(uLight - rd);
      vec2 need = hv.xz / max(hv.y, 0.05) - n.xz / n.y;
      if (dot(need, need) < 0.09) {
        float fa = tw * uFov * 2.0 / uRes.y / sqrt(max(abs(dot(rd, n)), 0.08));
        col += vec3(1.0, 0.93, 0.80) * (F * 8.0 * sparkle(ph.xz, fa, need) * groundShadow(ph));
      }
    }
    // (the density met right up to the surface is the surface, unless there is more of it than
    // the three cells a surface is thick at the most: see crossing())
    float skin = 3.0 / max(c, 0.15) + uStep, met = tw - g_runFrom;
    if (!g_mapped && met > skin) mist += g_run * (1.0 - skin / met);
    // into the liquid: how far the refracted ray travels in it, and what it lands on
    vec3 rr = refract(rd, n, 1.0 / IOR);
    rr = normalize(rr + vec3(equal(rr, vec3(0.0))) * 1.0e-6);
    // (0: the surface was met at, or just inside, what the liquid lies against)
    vec2 op2 = traceOpaque(ph, rr);
    float tEnd2 = op2.x >= 0.0 ? op2.x : 400.0;
    float te = crossing(ph, rr, 0.05, tEnd2, true, 0.0, none, noLines);
    // (liquid that ends within two cells of what the ray lands on reaches it: in the cells the
    // ground cuts the density is below the threshold, and where bricks that are stored lie next
    // to water that is not, the sea bed showed their outlines)
    float thick = te >= 0.0 && te < tEnd2 - 2.0 ? te : tEnd2;
    if (te >= 0.0 && rr.y < 0.0 && uHfMap.w > 0.5) {
      // Out of the back of a crest of the height field, with its water below: the ray goes into
      // that next, and what it lands on is seen through it. (One stretch of water is all that
      // is followed: without this the sea bed showed behind every steep crest as if it were dry.)
      vec3 pe = ph + rr * te;
      vec2 sf = hfSurf(pe.xz, false);
      if (sf.y > 0.0 && pe.y < sf.x + 12.0) thick = tEnd2;
    }
    vec3 behind = op2.x >= 0.0 ? shadeOpaque(ph + rr * op2.x, op2.y, tw + op2.x) : sky(rr);
    vec3 trans = exp(-vec3(0.075, 0.020, 0.010) * thick);
    float body = 1.0 - exp(-thick * 0.035);
    col += (1.0 - F) * trans * mix(behind, vec3(0.03, 0.30, 0.52), body * 0.75);
    // deep water: what the light scattered in it sends back (nothing to speak of in a pool)
    col += (1.0 - F) * vec3(0.010, 0.065, 0.120) * (1.0 - exp(-thick * 0.012));
    float fc = foamCover(ph.xz, foot);
    if (fc > 0.0) col = mix(col, vec3(0.86, 0.90, 0.93) * (0.40 + 0.20 * n.y + 0.75 * max(dot(n, uLight), 0.0) * groundShadow(ph)), 0.9 * fc);
  } else if (op.x > 0.0) {
    col = shadeOpaque(ro + rd * op.x, op.y, op.x);
  } else {
    col = sky(rd);
  }
  // spray: density below the surface threshold, met on the way
  float m = 1.0 - exp(-mist * uMist);
  col = mix(col, vec3(0.90, 0.95, 0.99), m * 0.9);
  if (uCursor.w > 0.0) {
    // the brush: a ring where the pointer acts
    vec3 oc = ro - uCursor.xyz;
    float bq = dot(oc, rd), disc = bq * bq - dot(oc, oc) + uCursor.w * uCursor.w;
    if (disc > 0.0 && -bq > 0.0) {
      float rim = 1.0 - sqrt(disc) / uCursor.w;
      col = mix(col, vec3(1.0, 0.85, 0.35), 0.35 * smoothstep(0.75, 1.0, rim) + 0.05);
    }
  }
  col = col * (1.0 - lines.a) + lines.rgb;
  // tone map + gamma
  col = col / (1.0 + 0.35 * col);
  col = pow(col * 1.25, vec3(1.0 / 2.2));
  o = vec4(col, 1.0);
}`);
  }

  _uniforms(world, o = {}) {
    const sim = world.sim;
    const L0 = sim.levels[0];
    const set = L0.set;
    // the density pool with borders: as many rows of tiles as the slots in use need
    const rows = Math.max(1, Math.ceil(set.high / this.perRow));
    if (!this.vol || rows > this.volRows) {
      if (this.vol) this.glc.deleteTexture(this.vol);
      this.volRows = Math.ceil(rows * 1.5);
      this.vol = this.glc.texture(this.perRow * AT, this.volRows * AP, 'r16f', { linear: true });
    }
    const kinds = new Float32Array(MAX_PRIMS);
    kinds.set(world.solidKinds());
    const grp = sim.hf ? sim.hfGroups() : null;
    const sea = world.sea ? world.sea() : null;
    // the hulls as they were drawn (o.exact = false: as the lattices the solver has them on)
    const solids = world.solidPrims();
    const X = (this._exact = exactUniforms(solids, 'S', this._exact, o.exact !== false));
    return {
      ...X,
      uHfSea: sea ? [sea.level, sea.amp, sea.k, sea.phase] : [0, 0, 0, 0],
      uHfSeaDir: sea ? [sea.dir[0], sea.dir[1], sea.origin[0] * sea.dir[0] + sea.origin[1] * sea.dir[1], 1] : [0, 0, 0, 0],
      ...sim._U(L0), ...sim._solidUniforms(), ...(sim.swell && o.swell !== false ? sim.swell.uniforms : this.noSwell),
      ...(grp ? { uHfGrp: grp.tex, uHfGrpMap: grp.map } : { uHfGrp: this.noSwell.uSwell, uHfGrpMap: [0, 0, 0, 0] }),
      uVol: this.vol, uVolInv: [1 / this.vol.w, 1 / this.vol.h], uVolRow: this.perRow,
      uS1: sim.R.s[1], uS2: sim.R.s[2],
      uTerMax: world.terrain ? world.terrain.max : 0,
      ...sim._prims(solids, 'S'),
      uKind: kinds,
      _rows: rows,
    };
  }

  /** Bring the density pool and the per-brick statistics up to date with the solver's state. */
  _prepare(world, o = {}) {
    const sim = world.sim;
    sim._syncBricks();
    if (sim.statsFrame !== sim.frame || sim._statsVersion !== sim.levels[0].set.version) sim.refreshSlotStats();
    const U = this._uniforms(world, o);
    this.glc.draw(this.pApron, this.vol, { ...U, uLiq: sim.liq }, { viewport: [0, 0, this.vol.w, U._rows * AP] });
    return U;
  }

  render(world, camera, o = {}) {
    const glc = this.glc;
    const U = this._prepare(world, o);
    this.U = U;
    const { pos, right, up, fwd } = camera.basis();
    const canvas = glc.canvas;
    const l = [0.45, 0.80, 0.35];
    const ll = Math.hypot(...l);
    glc.draw(this.pShade, null, {
      ...U,
      uRes: [canvas.width, canvas.height],
      uCamPos: pos,
      uCamRot: [right[0], right[1], right[2], up[0], up[1], up[2], fwd[0], fwd[1], fwd[2]],
      uFov: camera.fov,
      uStep: o.step ?? 0.6,
      uLight: [l[0] / ll, l[1] / ll, l[2] / ll],
      uMist: o.mist === false ? 0 : 0.22,
      uTime: world.time,
      uDetail: o.detail === false ? 0 : 1,
      ...(world.sim.foam && o.foam !== false ? world.sim.foam.uniforms : { uFoam: this.noSwell.uSwell, uFoamMap: [0, 0, 0, 0] }),
      uOverlay: o.overlay ? 1 : 0,
      uCursor: o.cursor || [0, 0, 0, 0],
    });
  }

  /**
   * Distances along a ray to the liquid surface and to the ground or an obstacle (asynchronous
   * 1-pixel readback): { liquid, solid }, each -1 if there is none; null if busy or unavailable.
   */
  async pick(world, origin, dir) {
    if (this.pickBusy || !this.U || this.glc.readback === 'scrambled') return null;
    this.pickBusy = true;
    try {
      this.glc.draw(this.pPick, this.pickTex, { ...this.U, uRo: origin, uRd: dir, uStep: 0.7, uOverlay: 0 });
      const px = await this.glc.readAsync(this.pickTex);
      return { liquid: px[0], solid: px[1] };
    } finally {
      this.pickBusy = false;
    }
  }

  dispose() {
    if (this.vol) this.glc.deleteTexture(this.vol);
    this.glc.deleteTexture(this.pickTex);
    this.glc.deleteTexture(this.noSwell.uSwell);
  }
}
