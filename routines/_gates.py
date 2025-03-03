import numpy as np
import yastn.tn.fpeps as peps


def gates_from_HH(HH, step):
    """ generate local gates for peps evolution """

    local = []
    for val, site, op in HH['local']:
        II = op @ op
        G = np.cos(val * step) * II - 1j * np.sin(val * step) * op
        local.append(peps.Gate_local(G, site))

    nn = []
    for val, site0, op0, site1, op1 in HH['nn']:
        I0, I1 = op0 @ op0, op1 @ op1
        II = peps.gates.fkron(I0, I1, sites=(0, 1))
        XX = peps.gates.fkron(op0, op1, sites=(0, 1))
        G01 = np.cos(val * step) * II - 1j * np.sin(val * step) * XX
        nn.append(peps.gates.decompose_nn_gate(G01, peps.Bond(site0, site1)))

    return peps.Gates(nn=nn, local=local)


def measure_HH(env_ctm, HH):
    eng = 0

    for val, site, op in HH['local']:
        eng += val * env_ctm.measure_1site(op, site=site)

    for val, site0, op0, site1, op1 in HH['nn']:
        eng += val * env_ctm.measure_nn(op0, op1, bond=(site0, site1))

    return eng
