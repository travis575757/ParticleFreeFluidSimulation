// HeightField: water that is not solved in 3D, as shallow water over the ground.
//
// A map of n[0] x n[1] texels over x and z (a 2D world: n[1] = 1), each `cell` cells of the 3D
// grid on a side, with a depth and two face velocities per texel (src/shaders/heightfield.js).
// On its own it is a shallow-water solver; in a world, SparseSim hands liquid over to it and
// takes liquid from it where the two meet (src/world/hfcouple.js).
//
// Coordinates are cells in the solver's shape space, as the primitives and the terrain.
import { heightfieldShaders } from '../shaders/heightfield.js';

export const HF_PARAMS = {
  dry: 1e-3,       // a texel with less water than this is dry (cells)
  cfl: 0.5,        // the fastest wave crosses this share of a texel per sub-step, at most
  maxSub: 12,      // sub-steps per step at most (beyond it the velocity limit holds the scheme together)
  friction: 0,     // drag of the bed: dv/dt = -friction |v| v / h. The 3D solver has none; with it the two disagree on a slope
  look: 4,         // steps between looks at how fast the waves are (they decide the number of sub-steps)
  sync: 0,         // 1: those looks block (tests)
};

export class HeightField {
  /**
   * @param o.origin   [x, z] of the corner of texel (0, 0), cells
   * @param o.n        [nx, nz] texels
   * @param o.cell     cells per texel
   * @param o.gravity  cells / s^2
   */
  constructor(glc, { origin = [0, 0], n = [256, 256], cell = 2, gravity = 100, params = {} } = {}) {
    this.glc = glc;
    this.origin = origin;
    this.n = n;
    this.cell = cell;
    this.gravity = gravity;
    this.params = { ...HF_PARAMS, ...params };
    const S = heightfieldShaders();
    this.P = {};
    for (const k of Object.keys(S)) this.P[k] = glc.program('hf.' + k, S[k]);
    const [nx, nz] = n;
    this.H = [glc.texture(nx, nz, 'rgba32f'), glc.texture(nx, nz, 'rgba32f')];
    this.Bd = glc.texture(nx, nz, 'rgba32f');
    this.dS = glc.texture(nx, nz, 'rgba32f');          // what the sources of the step added and took per texel
    // reductions of the map, 8 x 8 to one, down to a single texel
    this.pyr = [];
    for (let w = nx, h = nz; w > 1 || h > 1;) {
      w = Math.ceil(w / 8); h = Math.ceil(h / 8);
      this.pyr.push(glc.texture(w, h, 'rgba32f'));
    }
    if (!this.pyr.length) this.pyr.push(glc.texture(1, 1, 'rgba32f'));
    this.ledger = glc.texture(1, 1, 'rgba32f');         // sum of dS over the steps since it was last read
    glc.clear(this.ledger);
    // water of a texel that is not in its depth yet: r = handed over by another solver, g = lost
    // by the sums in single precision (pass "depth")
    this.X = [glc.texture(nx, nz, 'rgba32f'), glc.texture(nx, nz, 'rgba32f')];
    glc.clear(this.X[0]);
    this.bed = new Float32Array(nx * nz);
    this.blocked = new Uint8Array(nx * nz);
    this.sources = [];     // { box: [x0, z0, x1, z1] (cells), rate (cells of depth / s) } or { box, level, pull (share per step) }
    // Ways under the ground, at most four: { a: box, b: box, area (cells^2 of pipe; the flow is
    // 0.6 area sqrt(2 g dH)) or seep (cells^2 / s; the flow is seep dH), sill (the highest point
    // of the way, cells) }. Pass "link" of the shaders moves the water.
    this.links = [];
    this.linkStat = [glc.texture(4, 1, 'rgba32f'), glc.texture(4, 1, 'rgba32f')];
    this.linkState = [];   // per link, when last read: { levelA, levelB, flow (cells^3 / s, from a to b) }
    this.added = 0;        // what the sources have put in and taken out (cells^3; 2D world: cells^2), as far as read
    this.removed = 0;
    this.linked = 0;       // what the links have moved ...
    this.linkLost = 0;     // ... and what went in at one end without coming out at the other (rounding)
    this.speed = 0;        // the fastest a wave travelled when last looked at (cells / s)
    this.deepest = 0;
    this.frame = 0;
    this.subSteps = 1;
    this._run = 0;
    glc.clear(this.H[0]);
    glc.clear(this.Bd);
  }

  /** Area of a texel (2D world: its width). */
  get area() { return this.n[1] > 1 ? this.cell * this.cell : this.cell; }

  /** Centre of texel (i, j), cells. */
  center(i, j = 0) { return [this.origin[0] + (i + 0.5) * this.cell, this.origin[1] + (j + 0.5) * this.cell]; }

  /**
   * The bed: height(x, z) of the ground, averaged over `sub` x `sub` points of each texel;
   * blocked(x, z) true where the texel is no part of the height field (a wall to it).
   */
  setBed(height, blocked = null, sub = 2) {
    const [nx, nz] = this.n;
    this._setBed(height, blocked || (() => false), sub, 0, 0, nx - 1, nz - 1);
  }

  /**
   * The bed anew where the ground has changed: the texels that the box of cells [x0, z0, x1, z1]
   * touches (their walls stay as they are). The water keeps its depth: it is lifted or let
   * down with the ground, and runs from there.
   */
  setBedIn(height, box, sub = 2) {
    const [nx, nz] = this.n, c = this.cell, o = this.origin;
    const i0 = Math.max(Math.floor((box[0] - o[0]) / c), 0), i1 = Math.min(Math.floor((box[2] - o[0]) / c), nx - 1);
    const j0 = nz > 1 ? Math.max(Math.floor((box[1] - o[1]) / c), 0) : 0, j1 = nz > 1 ? Math.min(Math.floor((box[3] - o[1]) / c), nz - 1) : 0;
    if (i0 > i1 || j0 > j1) return;
    this._setBed(height, null, sub, i0, j0, i1, j1);
  }

  /** blocked: null leaves the walls as they are. */
  _setBed(height, blocked, sub, i0, j0, i1, j1) {
    const [nx, nz] = this.n;
    const c = this.cell, w = i1 - i0 + 1, data = new Float32Array(w * (j1 - j0 + 1) * 4);
    const sz = nz > 1 ? sub : 1;
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        let sum = 0;
        for (let b = 0; b < sz; b++) {
          for (let a = 0; a < sub; a++) {
            sum += height(this.origin[0] + (i + (a + 0.5) / sub) * c, nz > 1 ? this.origin[1] + (j + (b + 0.5) / sz) * c : 0);
          }
        }
        const k = j * nx + i, q = (j - j0) * w + i - i0;
        this.bed[k] = sum / (sub * sz);
        if (blocked) this.blocked[k] = blocked(...this.center(i, j)) ? 1 : 0;
        data[4 * q] = this.bed[k];
        data[4 * q + 1] = this.blocked[k];
      }
    }
    this.glc.upload(this.Bd, data, i0, j0, w, j1 - j0 + 1);
    this._shoreDirty = true;
  }

  /**
   * The sea that the map lies in: `far` = { level, depth (of the open sea, cells), shallow,
   * reach, ramp, hold, tile, every }, or null for none. Far from everything that happens the
   * water of the map is then that sea's (this.sea, which the owner sets every step: World.sea),
   * a level and one long wave, and is not solved: pass "relax" of the shaders.
   *
   * Where that is, is kept in a small map of what needs the height field, one value per tile of
   * `tile` x `tile` texels: 1 within `reach` cells of
   *   - land, and water less than `shallow` times `depth` deep at rest (a shore, a shoal: there
   *     the wave is no longer the one of the open sea)
   *   - a wall, a source, the end of a link
   *   - whatever the owner has seen there within the last `hold` steps (seen(): water solved
   *     in 3D, a body that is awake)
   * falling to 0 over the next `ramp` cells. A tile that is needed is so at once; one that no
   * longer is goes back to the sea over a few seconds.
   */
  setFar(far) {
    if (this.need) this.glc.deleteTexture(this.need);
    this.need = null;
    this.far = far ? { shallow: 0.75, reach: 64, ramp: 128, hold: 300, tile: 8, every: 4, fall: 0.02, ...far } : null;
    if (!this.far) return;
    const T = this.far.tile, w = Math.ceil(this.n[0] / T), h = Math.ceil(this.n[1] / T);
    this.need = this.glc.texture(w, h, 'r32f');
    this.needA = new Float32Array(w * h).fill(1);    // what the shader has
    this._needD = new Float32Array(w * h);
    this._shore = new Uint8Array(w * h);
    this._seen = new Int32Array(w * h).fill(-1e9);
    this._shoreDirty = true;
    this.glc.upload(this.need, this.needA);
  }

  /** Something that needs the height field is in the box [x0, z0, x1, z1] (cells) now: see setFar. */
  seen(x0, z0, x1, z1) {
    if (!this.far) return;
    const T = this.far.tile * this.cell, w = this.need.w, h = this.need.h;
    const i0 = Math.max(Math.floor((x0 - this.origin[0]) / T), 0), i1 = Math.min(Math.floor((x1 - this.origin[0]) / T), w - 1);
    const j0 = h > 1 ? Math.max(Math.floor((z0 - this.origin[1]) / T), 0) : 0, j1 = h > 1 ? Math.min(Math.floor((z1 - this.origin[1]) / T), h - 1) : 0;
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) this._seen[j * w + i] = this.frame;
  }

  /** The tiles with a shore, a shoal, a wall, a source or the end of a link in them. */
  _shoreTiles() {
    const F = this.far, T = F.tile, w = this.need.w, h = this.need.h, [nx, nz] = this.n;
    const shore = this._shore.fill(0);
    const least = F.shallow * F.depth;
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (this.blocked[k] || F.level - this.bed[k] < least) shore[Math.floor(j / T) * w + Math.floor(i / T)] = 1;
      }
    }
    const c = this.cell * T;
    for (const b of [...this.sources.filter((q) => !q.swell).map((q) => q.box), ...this.links.flatMap((l) => [l.a, l.b])]) {
      const i0 = Math.max(Math.floor((b[0] - this.origin[0]) / c), 0), i1 = Math.min(Math.floor((b[2] - this.origin[0]) / c), w - 1);
      const j0 = Math.max(Math.floor((b[1] - this.origin[1]) / c), 0), j1 = Math.min(Math.floor((b[3] - this.origin[1]) / c), h - 1);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) shore[j * w + i] = 1;
    }
    this._shoreDirty = false;
  }

  /** The map of what needs the height field anew (setFar), every few steps. `fresh`: as it is from now on, at once. */
  _needUpdate(fresh = false) {
    const F = this.far, w = this.need.w, h = this.need.h, d = this._needD, a = this.needA;
    if (this._shoreDirty) this._shoreTiles();
    const shore = this._shore, seen = this._seen, since = this.frame - F.hold;
    for (let k = 0; k < d.length; k++) d[k] = shore[k] || seen[k] > since ? 0 : 1e9;
    // how far it is to the nearest tile that needs it, in tiles: along rows, columns and diagonals, there and back
    const D = Math.SQRT2;
    for (let j = 0; j < h; j++) {
      for (let i = 0; i < w; i++) {
        const k = j * w + i;
        let v = d[k];
        if (i > 0) v = Math.min(v, d[k - 1] + 1);
        if (j > 0) {
          v = Math.min(v, d[k - w] + 1);
          if (i > 0) v = Math.min(v, d[k - w - 1] + D);
          if (i < w - 1) v = Math.min(v, d[k - w + 1] + D);
        }
        d[k] = v;
      }
    }
    for (let j = h - 1; j >= 0; j--) {
      for (let i = w - 1; i >= 0; i--) {
        const k = j * w + i;
        let v = d[k];
        if (i < w - 1) v = Math.min(v, d[k + 1] + 1);
        if (j < h - 1) {
          v = Math.min(v, d[k + w] + 1);
          if (i < w - 1) v = Math.min(v, d[k + w + 1] + D);
          if (i > 0) v = Math.min(v, d[k + w - 1] + D);
        }
        d[k] = v;
      }
    }
    const c = F.tile * this.cell;
    let changed = false;
    for (let k = 0; k < d.length; k++) {
      const want = Math.min(Math.max(1 - (d[k] * c - F.reach) / F.ramp, 0), 1);
      const v = fresh || want >= a[k] ? want : Math.max(a[k] - F.fall, want);
      if (v !== a[k]) { a[k] = v; changed = true; }
    }
    if (changed || fresh) this.glc.upload(this.need, a);
  }

  /** What needs the height field at (x, z), cells: 1 it is the height field's water, 0 the sea's (setFar); 1 without such a sea. */
  needAt(x, z) {
    if (!this.far) return 1;
    const c = this.far.tile * this.cell, w = this.need.w, h = this.need.h;
    const i = Math.min(Math.max(Math.floor((x - this.origin[0]) / c), 0), w - 1), j = Math.min(Math.max(Math.floor((z - this.origin[1]) / c), 0), h - 1);
    return this.needA[j * w + i];
  }

  /** The state: depth(x, z, bed) and, if given, velocity(x, z) => [u, w] at the low faces of each texel. */
  setState(depth, velocity = null) {
    const [nx, nz] = this.n;
    const data = new Float32Array(nx * nz * 4);
    for (let j = 0; j < nz; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const [x, z] = this.center(i, j);
        data[4 * k] = this.blocked[k] ? 0 : Math.max(depth(x, z, this.bed[k]), 0);
        if (velocity) {
          data[4 * k + 1] = velocity(x - 0.5 * this.cell, z)[0];
          data[4 * k + 2] = velocity(x, z - 0.5 * this.cell)[1];
        }
      }
    }
    this.glc.upload(this.H[0], data);
    this.glc.clear(this.X[0]);
    this.frame = 0;
    if (this.far) { this._seen.fill(-1e9); this._shoreDirty = true; this._needUpdate(true); }
    this.speed = 0;
    this._run++;
    this.added = this.removed = this.linked = this.linkLost = 0;
    this.linkState = [];
    this.glc.clear(this.ledger);
    this._look(true);
  }

  /** Water up to `level` everywhere the bed is lower (within box [x0, z0, x1, z1], if given). */
  fill(level, box = null) {
    this.setState((x, z, b) => (box && (x < box[0] || x > box[2] || z < box[1] || z > box[3]) ? 0 : level - b));
  }

  _reduce(prog, src, first = {}) {
    const { glc, P, pyr } = this;
    let from = src;
    pyr.forEach((t, i) => { glc.draw(P[prog], t, { uSrc: from, ...(i === 0 ? first : { uFirst: 0, uG: 0, uDry: 0 }) }); from = t; });
    return from;
  }

  /** How fast the waves are: decides the number of sub-steps. */
  _look(block = false) {
    const p = this.params;
    if (this._busy && !block) return;
    const top = this._reduce('peak', this.H[0], { uFirst: 1, uG: this.gravity, uDry: p.dry });
    const take = (d) => { this.speed = d[0]; this.deepest = d[1]; this.fastest = d[2]; };
    if (block || p.sync) { take(this.glc.read(top)); return; }
    this._busy = true;
    const run = this._run;
    this.glc.readAsync(top).then((d) => { if (run === this._run) take(d); }, () => {}).finally(() => { this._busy = false; });
  }

  _sourceUniforms(live) {
    const box = new Float32Array(32), val = new Float32Array(32);
    const c = this.cell;
    live.slice(0, 8).forEach((s, i) => {
      box.set([(s.box[0] - this.origin[0]) / c, (s.box[1] - this.origin[1]) / c, (s.box[2] - this.origin[0]) / c, (s.box[3] - this.origin[1]) / c], 4 * i);
      val.set([s.rate || 0, s.level || 0, s.level === undefined ? 0 : s.pull || 0.05, 0], 4 * i);
    });
    return { uSrcN: Math.min(live.length, 8), uSrcBox: box, uSrcVal: val };
  }

  _linkUniforms() {
    const A = new Float32Array(16), B = new Float32Array(16), P = new Float32Array(16);
    const c = this.cell, o = this.origin;
    const rect = (b) => [Math.round((b[0] - o[0]) / c), Math.round((b[1] - o[1]) / c), Math.round((b[2] - o[0]) / c), Math.round((b[3] - o[1]) / c)];
    this.links.slice(0, 4).forEach((l, i) => {
      A.set(rect(l.a), 4 * i);
      B.set(rect(l.b), 4 * i);
      P.set(l.seep ? [l.seep, 1, l.sill ?? -1e30, 0] : [0.6 * l.area * Math.sqrt(2 * this.gravity), 0, l.sill ?? -1e30, 0], 4 * i);
    });
    return { uLinkN: Math.min(this.links.length, 4), uLinkA: A, uLinkB: B, uLinkP: P };
  }

  /** The flow of link i for the levels a and b of its ends (cells^3 / s, positive from a to b), as pass "link" has it. */
  linkFlow(i, a, b) {
    const l = this.links[i];
    const head = Math.max(Math.max(a, b) - Math.max(Math.min(a, b), l.sill ?? -1e30), 0);
    const q = l.seep ? l.seep * head : 0.6 * l.area * Math.sqrt(2 * this.gravity * head);
    return a > b ? q : -q;
  }

  /** The boxes (cells) in which sources and links move water: the 3D solver keeps out of them (SparseSim.hfLock). */
  lockBoxes() { return [...this.sources.map((s) => s.box), ...this.links.flatMap((l) => [l.a, l.b])]; }

  /** One time step: the sources and links, then as many sub-steps as the fastest wave needs. */
  step(dt) {
    const { glc, P, H, params: p } = this;
    const live = this.sources.filter((s) => !s.off);
    if (live.length) {
      glc.draw(P.source, [H[1], this.dS], { uH: H[0], uBed: this.Bd, uDt: dt, ...this._sourceUniforms(live) });
      H.reverse();
      this.meter();
    }
    if (this.links.length) {
      const U = { uH: H[0], uBed: this.Bd, ...this._linkUniforms() };
      glc.draw(P.linkStat, this.linkStat, { ...U, uDry: p.dry });
      glc.draw(P.link, [H[1], this.dS], { ...U, uStatA: this.linkStat[0], uStatB: this.linkStat[1], uDt: dt, uArea: this.area });
      H.reverse();
      this.meter();
      if (this.frame % p.look === 0) this._lookLinks();
    }
    // (the estimate is a few steps old: a quarter more, and the waves of the deepest water at the least)
    const c = Math.max(1.25 * this.speed, Math.sqrt(this.gravity * this.deepest));
    const sub = Math.min(Math.max(Math.ceil(dt * c / (p.cfl * this.cell)), 1), p.maxSub);
    const h = dt / sub;
    const umax = (this.n[1] > 1 ? 0.25 : 0.5) * this.cell / h;
    const U = { uBed: this.Bd, uDt: h, uDx: this.cell, uG: this.gravity, uDry: p.dry, uFric: p.friction, uUmax: umax };
    const D = { uBed: this.Bd, uDt: h, uDx: this.cell, uUmax: umax, uOne: 1 };
    const X = this.X;
    for (let k = 0; k < sub; k++) {
      glc.draw(P.depth, [H[1], X[1]], { uH: H[0], ...D, uX: X[0], uLeft: sub - k });
      X.reverse();
      glc.draw(P.vel, H[0], { uH: H[1], ...U });
    }
    if (this.far && this.sea) {
      if (this.frame % this.far.every === 0) this._needUpdate();
      const w = this.sea;
      glc.draw(P.relax, [H[1], this.dS], {
        uH: H[0], uBed: this.Bd, uNeed: this.need, uTile: this.far.tile, uG: this.gravity, uDry: p.dry,
        uSea: [w.level, w.amp, w.k, w.phase], uSeaDir: [w.dir[0], w.dir[1], w.origin[0] * w.dir[0] + w.origin[1] * w.dir[1], 0],
        uMap: [this.origin[0], this.origin[1], this.cell, 0],
      });
      H.reverse();
      this.meter();
    }
    this.subSteps = sub;
    this.frame++;
    if (this.frame % p.look === 0) {
      this._look();
      // (the ledger is a sum in single precision: left to grow, it rounds every step the same way)
      if (p.sync) this.readLedger(); else this.readLedgerAsync();
    }
  }

  /** The levels at the ends of the links and what flows, for whoever shows them (blocking with params.sync). */
  _lookLinks() {
    const take = (a, b) => {
      this.linkState = this.links.slice(0, 4).map((l, i) => {
        const A = a.subarray(4 * i, 4 * i + 4), B = b.subarray(4 * i, 4 * i + 4);
        const flows = (A[2] > B[2] ? A : B)[3] > 0.5 && (A[2] > B[2] ? B : A)[1] > 0.5;
        return { levelA: A[2], levelB: B[2], wetA: A[3] > 0.5, wetB: B[3] > 0.5, flow: flows ? this.linkFlow(i, A[2], B[2]) : 0 };
      });
    };
    if (this.params.sync) { take(this.glc.read(this.linkStat[0]), this.glc.read(this.linkStat[1])); return; }
    if (this._linkBusy) return;
    this._linkBusy = true;
    const run = this._run;
    Promise.all(this.linkStat.map((t) => this.glc.readAsync(t)))
      .then(([a, b]) => { if (run === this._run) take(a, b); }, () => {}).finally(() => { this._linkBusy = false; });
  }

  /** Add what a pass has written into dS (per texel: r = depth gained, g = lost) to the ledger. */
  meter() {
    this.glc.draw(this.P.sum, this.ledger, { uSrc: this._reduce('sum', this.dS) }, { blend: true });
    this._metered = true;
  }

  /** Move what the sources have added and removed so far into `added` / `removed` (blocking). */
  readLedger() {
    if (!this._metered) return;
    const d = this.glc.read(this.ledger);
    this.glc.clear(this.ledger);
    this._metered = false;
    this.added += d[0] * this.area;
    this.removed += d[1] * this.area;
    this.linked += d[2] * this.area;
    this.linkLost += d[3] * this.area - d[2] * this.area;
  }

  /**
   * The same without blocking: the totals follow a few frames later, in the order the reads were
   * made. Resolves with { added, removed } as they were at this call.
   */
  readLedgerAsync() {
    const run = this._run, before = this._ledgerRead || Promise.resolve();
    let read = null;
    if (this._metered) {
      read = this.glc.readAsync(this.ledger).catch(() => null);
      this.glc.clear(this.ledger);      // (the read above has its copy already)
      this._metered = false;
    }
    return (this._ledgerRead = Promise.all([read, before]).then(([d]) => {
      if (d && run === this._run) {
        this.added += d[0] * this.area; this.removed += d[1] * this.area;
        this.linked += d[2] * this.area; this.linkLost += (d[3] - d[2]) * this.area;
      }
      return { added: this.added, removed: this.removed };
    }));
  }

  /** readLedger once the non-blocking reads that are under way have arrived: the totals are then complete. */
  async settleLedger() {
    await this._ledgerRead;
    this.readLedger();
  }

  /** The water in the map (cells^3; 2D world: cells^2), with a blocking read-back. */
  volume() {
    const v = this.glc.read(this._reduce('sum', this.H[0]))[0];
    const x = this.glc.read(this._reduce('sum', this.X[0]));
    return (v + x[0] + x[1]) * this.area;
  }

  /** The state as arrays (blocking; tests): { h, u, w, m, x } per texel (x: the water of the texel that is not in h yet). */
  read() {
    const d = this.glc.read(this.H[0]);
    const k = d.length / 4;
    const out = { h: new Float32Array(k), u: new Float32Array(k), w: new Float32Array(k), m: new Float32Array(k), x: new Float32Array(k) };
    for (let i = 0; i < k; i++) { out.h[i] = d[4 * i]; out.u[i] = d[4 * i + 1]; out.w[i] = d[4 * i + 2]; out.m[i] = d[4 * i + 3]; }
    const x = this.glc.read(this.X[0]);
    for (let i = 0; i < k; i++) out.x[i] = x[4 * i] + x[4 * i + 1];
    return out;
  }

  dispose() {
    for (const t of [...this.H, this.Bd, this.dS, ...this.pyr, this.ledger, ...this.X, ...this.linkStat, ...(this.need ? [this.need] : [])]) this.glc.deleteTexture(t);
  }
}
