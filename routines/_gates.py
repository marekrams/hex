import numpy as np
import yastn
import yastn.tn.fpeps as peps


def gates_from_HH(HH, step):
    """ generate local gates for peps evolution """

    local = []
    for val, site, op, inds in HH['local']:
        II = op @ op
        G = np.cos(val * step) * II - 1j * np.sin(val * step) * op
        local.append(peps.Gate_local(G, site))

    nn = []
    for val, site0, op0, site1, op1, inds in HH['nn']:
        I0, I1 = op0 @ op0, op1 @ op1
        I0 = I0.add_leg(axis=2, s=1)
        op0 = op0.add_leg(axis=2, s=1)
        I1 = I1.add_leg(axis=2, s=-1)
        op1 = op1.add_leg(axis=2, s=-1)
        ampI, ampX = np.cos(val * step), -1j * np.sin(val * step)
        G0 = yastn.block({0: I0 * ampI, 1: op0 * ampX}, common_legs=(0, 1))
        G1 = yastn.block({0: I1, 1: op1}, common_legs=(0, 1))
        nn.append(peps.Gate_nn(G0, G1, bond=peps.Bond(site0, site1)))
        # II = peps.gates.fkron(I0, I1, sites=(0, 1))
        # XX = peps.gates.fkron(op0, op1, sites=(0, 1))
        # G01 = np.cos(val * step) * II - 1j * np.sin(val * step) * XX
        # nn.append(peps.gates.decompose_nn_gate(G01, peps.Bond(site0, site1)))

    return peps.Gates(nn=nn, local=local)


def measure_H_ctm(env_ctm, HH):
    eng = 0
    out = {}

    for val, site, op, inds in HH['local']:
        out[inds] = env_ctm.measure_1site(op, site=site).real
        eng += val * out[inds]

    for val, site0, op0, site1, op1, inds in HH['nn']:
        out[inds] = env_ctm.measure_nn(op0, op1, bond=(site0, site1)).real
        eng += val * out[inds]

    out = dict(sorted(out.items()))
    return eng, out


def measure_H_mps(env_mps, HH):
    eng = 0

    Os = {}
    for val, site, op, inds in HH['local']:
        if site not in Os:
            Os[site] = {}
        Os[site][inds] = op

    OPs = {}
    for val, s0, op0, s1, op1, inds in HH['nn']:
        if (s0, s1) not in OPs:
            OPs[s0, s1] = {}
        OPs[s0, s1][inds] = (op0, op1)

    ZZs2 = env_mps.measure_1site(Os)
    ZZs3 = env_mps.measure_nn(OPs)

    ZZs2 = {k[2]: v.real for k, v in ZZs2.items()}
    ZZs3 = {k[2]: v.real for k, v in ZZs3.items()}
    ZZs = {**ZZs2, **ZZs3}
    out = dict(sorted(ZZs.items()))

    eng = 0
    for val, site, op, inds in HH['local']:
        eng += val * ZZs[inds]
    for val, s0, op0, s1, op1, inds in HH['nn']:
        eng += val * ZZs[inds]

    return eng, out