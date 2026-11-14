// Foam: whitewater on the surface, drawn only (src/shaders/foam.js).
//
// A layer on the map the scene has: the height field, or the map of waves beyond the solved
// water (Swell). One pass per step. Scenes with neither have none.
import { foamShaders } from '../shaders/foam.js';

export const FOAM_PARAMS = {
  life: 3.5,     // seconds in which foam fades to a third
  gain: 6,       // what water that breaks adds per second (1: all white)
};

let programs = null;

export class Foam {
  /** @param on  the HeightField or the Swell of sim */
  constructor(sim, on, params = {}) {
    const glc = sim.glc;
    this.sim = sim;
    this.glc = glc;
    this.on = on;
    this.hf = on === sim.hf;
    this.params = { ...FOAM_PARAMS, ...params };
    if (!programs || programs.glc !== glc) {
      const S = foamShaders();
      programs = { glc, hf: glc.program('foam.hf', S.hf), swell: glc.program('foam.swell', S.swell) };
    }
    const n = this.hf ? on.n : [on.params.n, on.params.n];
    this.n = n;
    this.cell = this.hf ? on.cell : on.params.cell;
    this.F = [glc.texture(n[0], n[1], 'rgba16f', { linear: true }), glc.texture(n[0], n[1], 'rgba16f', { linear: true })];
    this.reset();
  }

  reset() {
    for (const t of this.F) this.glc.clear(t);
    this.left = 0;            // seconds for which the foam of a map that is switched off is still stepped
  }

  /** One time step, after those of the solver and of the map. */
  step() {
    const { glc, sim, on, F, params: p } = this;
    const dt = sim.params.dt;
    if (!this.hf) {
      // (a map that is switched off has no water that breaks: what foam there is fades, and
      // after that there is nothing to do)
      if (on.active) this.left = 4 * p.life;
      else if (this.left <= 0) return;
      else if ((this.left -= dt) <= 0) { this.reset(); return; }
    }
    const U = { uF: F[0], uDt: dt, uKeep: Math.exp(-dt * Math.log(3) / p.life), uGain: p.gain, uDx: this.cell };
    if (this.hf) glc.draw(programs.hf, F[1], { ...U, uH: on.H[0], uBed: on.Bd, uG: on.gravity, uDry: on.params.dry });
    else glc.draw(programs.swell, F[1], { ...U, uZ: on.Z, uE: on.E });
    F.reverse();
  }

  /** For the renderer: the map and where it lies (world cells). */
  get uniforms() {
    const a = this.sim.anchor;
    const o = this.hf ? [this.on.origin[0] + a[0], this.on.origin[1] + a[2]] : this.on.origin;
    return { uFoam: this.F[0], uFoamMap: [o[0], o[1], 1 / (this.n[0] * this.cell), 1 / (this.n[1] * this.cell)] };
  }

  dispose() { for (const t of this.F) this.glc.deleteTexture(t); }
}
