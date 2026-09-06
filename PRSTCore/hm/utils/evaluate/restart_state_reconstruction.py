"""Numeric reconstruction called only by FAHM's external result reader.

This is not a compositional forward simulator or a flash/adjoint solver.
RestartEOS implements the numeric MRST EOS calls needed for imported x/y/Z.
"""

from types import SimpleNamespace

import numpy as np

from PRSTCore.deckformat.resultinput.restart_contract import RestartContractError


def reconstruct_compositional(state, eos):
    required = ('pressure', 'T', 'components', 'x', 'y', 'L')
    if any(key not in state for key in required):
        raise RestartContractError('Compositional restart requires pressure/T/components/x/y/L')
    z = np.asarray(state['components'], dtype=float)
    nc = np.size(state['pressure'])
    if z.ndim != 2 or z.shape[0] != nc or any(np.asarray(state[f]).shape != z.shape for f in ('x', 'y')):
        raise RestartContractError('Compositional cell/component shape mismatch')
    if np.size(state['T']) != nc or np.size(state['L']) != nc or not np.all(np.isfinite(z)):
        raise RestartContractError('Compositional T/L/component count or values are invalid')
    z = np.maximum(z, eos.minimumComposition)
    z = z / np.sum(z, axis=1, keepdims=True)
    x, y = z.copy(), z.copy()
    two_phase = ~np.asarray(eos.getSinglePhase(state), dtype=bool).ravel()
    x[two_phase], y[two_phase] = state['x'][two_phase], state['y'][two_phase]
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(x <= 0) or np.any(y < 0):
        raise RestartContractError('Compositional phase fractions cannot produce finite K=y/x')
    acf = eos.CompositionalMixture.acentricFactors
    sl, sv, al, av, bl, bv, bi = eos.getMixtureFugacityCoefficients(state['pressure'], state['T'], x, y, acf)
    state.update(x=x, y=y, K=y / x,
                 Z_L=eos.computeCompressibilityZ(state['pressure'], x, al, bl, sl, bi, True),
                 Z_V=eos.computeCompressibilityZ(state['pressure'], y, av, bv, sv, bi, False))


class RestartEOS:
    """MRST EquationOfStateModel numeric subset, with explicit mixture data.

    Critical pressure is Pa, temperature K; component order must be the deck
    order. No inferred critical properties or default black-oil conversion.
    """

    def __init__(self, critical_temperature, critical_pressure, acentric_factors,
                 binary_interaction, eos_type='PR', minimum_composition=1e-8,
                 select_gibbs_minimum=True, omega_a=None, omega_b=None):
        self.tc = np.asarray(critical_temperature, dtype=float).ravel()
        self.pc = np.asarray(critical_pressure, dtype=float).ravel()
        acf = np.asarray(acentric_factors, dtype=float).ravel()
        self.bic = np.asarray(binary_interaction, dtype=float)
        if self.tc.shape != self.pc.shape or acf.shape != self.tc.shape or self.bic.shape != (len(acf), len(acf)):
            raise RestartContractError('EOS mixture dimensions do not match')
        if np.any(self.tc <= 0) or np.any(self.pc <= 0):
            raise RestartContractError('EOS critical properties must be positive')
        if any(not np.all(np.isfinite(v)) for v in (self.tc,self.pc,acf,self.bic)) or not np.array_equal(self.bic,self.bic.T):
            raise RestartContractError('EOS properties must be finite and binary interaction symmetric')
        if not np.isfinite(minimum_composition) or minimum_composition <= 0:
            raise RestartContractError('EOS minimum composition must be finite and positive')
        self.CompositionalMixture = SimpleNamespace(acentricFactors=acf)
        self.minimumComposition = minimum_composition
        self.selectGibbsMinimum = select_gibbs_minimum
        self.kind = eos_type.upper()
        if self.kind in ('PR', 'PRCORR'):
            self.m1, self.m2 = 1 + np.sqrt(2), 1 - np.sqrt(2)
            oa, ob = .4572355, .0779691
        elif self.kind in ('SRK', 'RK'):
            self.m1, self.m2 = 0., 1.
            oa, ob = .427480, .086640
        else:
            raise RestartContractError('Unsupported MRST EOS type ' + self.kind)
        self.omegaA = np.asarray(oa if omega_a is None else omega_a)
        self.omegaB = np.asarray(ob if omega_b is None else omega_b)
        if any(v.size not in (1,len(acf)) or not np.all(np.isfinite(v)) or np.any(v <= 0) for v in (self.omegaA,self.omegaB)):
            raise RestartContractError('Invalid component-wise EOS omegaA/omegaB')

    def getSinglePhase(self, state):
        return np.asarray(state.get('flag', (state['L'] == 1) + 2 * (state['L'] == 0))) != 0

    def getMixtureFugacityCoefficients(self, P, T, x, y, acf):
        pr = np.asarray(P).reshape(-1, 1) / self.pc
        tr = np.asarray(T).reshape(-1, 1) / self.tc
        if np.any(pr <= 0) or np.any(tr <= 0):
            raise RestartContractError('EOS P/T must be positive')
        if self.kind in ('PR', 'PRCORR'):
            a = .37464 + 1.54226 * acf - .26992 * acf**2
            if self.kind == 'PRCORR':
                a = np.where(acf > .49, .379642 + 1.48503 * acf - .164423 * acf**2 + .016666 * acf**3, a)
            oa = self.omegaA * (1 + a * (1 - np.sqrt(tr)))**2
        elif self.kind == 'SRK':
            oa = self.omegaA * (1 + (.48 + 1.574 * acf - .176 * acf**2) * (1 - np.sqrt(tr)))**2
        else:
            oa = self.omegaA * tr**(-.5)
        ai = oa * pr / tr**2
        bi = self.omegaB * pr / tr
        aij = np.sqrt(ai[:, :, None] * ai[:, None, :]) * (1 - self.bic)

        def mix(z):
            s = np.zeros_like(z)
            a = np.zeros(z.shape[0])
            for i in range(z.shape[1]):
                a += np.sum(aij[:, i, :] * z[:, i, None] * z, axis=1)
                s += aij[:, i, :] * z[:, i, None]
            return s, a, np.sum(z * bi, axis=1)

        sl, al, bl = mix(x)
        sv, av, bv = mix(y)
        return sl, sv, al, av, bl, bv, bi

    def computeCompressibilityZ(self, p, xy, A, B, Si, Bi, isLiquid):
        m1, m2 = self.m1, self.m2
        e0 = -(A * B + m1 * m2 * B**2 * (B + 1))
        e1 = A - (m1 + m2 - m1 * m2) * B**2 - (m1 + m2) * B
        e2 = (m1 + m2 - 1) * B - 1
        output = []
        for i in range(len(A)):
            # Local MRST cubicPositive.m, not a replacement polynomial solver.
            Q = (e2[i]**2 - 3 * e1[i]) / 9
            R = (2 * e2[i]**3 - 9 * e2[i] * e1[i] + 27 * e0[i]) / 54
            M = R**2 - Q**3
            if M < 0:
                theta = np.arccos(R / np.sqrt(Q**3))
                roots = -2 * np.sqrt(Q) * np.cos((theta + np.array([0., 2 * np.pi, -2 * np.pi])) / 3) - e2[i] / 3
            else:
                S = -np.sign(R) * (abs(R) + np.sqrt(M))**(1 / 3)
                roots = np.array([S + (Q / S if S != 0 else 0.) - e2[i] / 3])
            valid = roots[np.isfinite(roots) & (roots > 0) & (roots >= B[i])]
            if not valid.size:
                raise RestartContractError('EOS has no physical compressibility root')
            lo, hi = valid.min(), valid.max()
            if self.selectGibbsMinimum:
                def gibbs(z):
                    phi = -np.log(z - B[i]) + np.log((z + m2 * B[i]) / (z + m1 * B[i])) * (A[i] / ((m1 - m2) * B[i])) * (2 * Si[i] / A[i] - Bi[i] / B[i]) + Bi[i] * ((z - 1) / B[i])
                    return np.sum(phi * xy[i])
                output.append(lo if gibbs(lo) < gibbs(hi) else hi)
            else:
                output.append(lo if isLiquid else hi)
        return np.asarray(output)


def reconstruct_aquifers(setup, states):
    """MRST aquifer recurrence, using actual selected/ministep dt.

    FIX: source retains old report dt after ministep expansion and fails to
    return the initialized aquifer state0. Both are explicit Stage 11 defects.
    """
    model = setup['model']
    aq = model.AquiferModel
    initial = aq.init_state_aquifer()
    previous = initial
    setup['state0']['aquiferSol'] = [dict(pressure=float(p), volume=float(v), flux=np.array([]))
                                     for p, v in zip(initial['pressure'], initial['volume'])]
    ix = aq.aquind
    conn = aq.aquifers[:, ix['conn']].astype(int)
    aquid = aq.aquifers[:, ix['aquid']].astype(int)
    for state, dt in zip(states, setup['schedule']['step']['val']):
        forces = {'W': state['wellSol'], 'bc': [], 'src': []}
        if hasattr(model, 'validateModel'):
            model = model.validateModel(forces)
        if hasattr(model, 'getProps'):
            phases = list(model.getPhaseNames())
            pW = np.asarray(model.getProps(state, 'PhasePressures')[phases.index('W')])
            rhoW = np.asarray(model.getProps(state, 'Density')[phases.index('W')])
        elif hasattr(model, '_phase_pressures') and getattr(model, '_blackoil_pvt', None) is not None:
            sat = np.asarray(state['s'])
            pW, _, _ = model._phase_pressures(state['pressure'], sat[:, 0], sat[:, -1] if model.gas else np.zeros(len(sat)))
            pvt = model._blackoil_pvt.eval(pW, rs_override=state['rs'], rv_override=state['rv'])
            rhoW = pvt['bw'] * model._mrst_surface_densities()[0]
        else:
            raise RestartContractError('Aquifer fallback requires model phase pressure/density properties')
        nc = np.size(state['pressure'])
        if np.size(pW) != nc or np.size(rhoW) != nc or np.any(conn < 0) or np.any(conn >= nc):
            raise RestartContractError('Aquifer phase-property/connection dimensions mismatch')
        q = aq.compute_aquifer_fluxes(p_aq=previous['pressure'], v_aq=previous['volume'],
            pW_conn=pW[conn], bW_conn=rhoW[conn], rhoWS=1.,
            gravity=float(np.asarray(model.gravity).ravel()[2]), dt=float(dt))
        previous = aq.update_after_convergence(previous, q, float(dt))
        state['aquiferSol'] = [dict(pressure=float(p), volume=float(v), flux=q[aquid == k + 1].copy())
                              for k, (p, v) in enumerate(zip(previous['pressure'], previous['volume']))]
