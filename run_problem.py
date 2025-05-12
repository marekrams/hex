import ast
import csv
import os
from pathlib import Path
import ray
import time
import time
import numpy as np
import yastn.tn.fpeps as peps
from routines import NSpin12, map_hex, gates_from_HH, gates_from_HH2, measure_H_ctm, measure_H_mps


def run_evolution(qpu, ang, les, p, D):
    #
    print(f"Starting {qpu=} {ang=} {les=} {p=} {D=}")
    keep_time = time.time()
    #

    Nspins = {"kyiv": 127, "torino": 133, "fez": 156}
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    HC = data["H"]
    HM = {(n,): 1 for n in range(Nspins[qpu])}
    ops = NSpin12(sym='dense')
    #
    hex = map_hex(N=Nspins[qpu])
    #
    HHC = hex.group_H(HC, ops.z, ops.I)
    HHM = hex.group_H(HM, ops.x, ops.I)
    #
    # initial product state
    #
    geometry = peps.SquareLattice(dims=hex.dims, boundary='obc')
    vectors = {site: ops.vec_x((1,) * Nloc) for site, Nloc in hex.s2N.items()}
    psi = peps.product_peps(geometry, vectors)
    #
    # run evolution
    #
    opts_svd_evol = {"D_total": D, "tol": 1e-12}
    env = peps.EnvBP(psi, which=which) if "BP" in which else peps.EnvNTU(psi, which=which)
    if "BP" in which:
        env.iterate_(max_sweeps=100, diff_tol=1e-8)
    #
    infoss = []
    t0 = time.time()
    for step, (gamma, beta) in enumerate(zip(data['gamma'], data['beta'])):

        gates = gates_from_HH2(HHC, gamma)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        infoss.append(infos)
        if "BP" in which:
            env.iterate_(max_sweeps=100, diff_tol=1e-8)
        gates = gates_from_HH2(HHM, beta)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        if "BP" in which:
            env.iterate_(max_sweeps=100, diff_tol=1e-8)
        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Step: {step}; Evolution time: {time.time() - t0:3.2f}; Truncation error: {Delta:0.6f}")
    #
    if D2 < D:
        print(" Truncating ")
        if which2 != which:
            env = peps.EnvBP(psi, which=which2) if "BP" in which2 else peps.EnvNTU(psi, which=which2)
            if "BP" in which2:
                env.iterate_(max_sweeps=50, diff_tol=1e-8)

        opts_svd_trun = {"D_total": D2}
        info2 = peps.truncate_(env, opts_svd=opts_svd_trun)
        if "BP" in which:
            env.iterate_(max_sweeps=50, diff_tol=1e-8)
        Delta2 = peps.accumulated_truncation_error([info2], statistics='mean')
        print(f"Truncation time: {time.time() - t0:3.2f}; Truncation error: {Delta2:0.6f}")
    else:
        Delta2 = 0.
    #
    if chi > 1:
        print(" Calculating energy from boundary MPS")
        opts_svd_mps = {"D_total":  chi}
        env_mps = peps.EnvBoundaryMPS(psi, opts_svd=opts_svd_mps, setup='tlbr')
        #
        eng, ZZ = measure_H_mps(env_mps, HHC)
        print(f"Energy for {p=}: {eng:0.4f}")
    else:
        print(" Calculating energy from BP")
        if (D2 < D and 'BP' not in which2) or 'BP' not in which:
            env = peps.EnvBP(psi, which=which2)
            env.iterate_(max_sweeps=50, diff_tol=1e-8)
        eng, ZZ = measure_H_ctm(env, HHC)
    #
    path = Path(f"./results/{data['Hamiltonian_name']}/{data['angles_name']}/{p=}/")
    path.mkdir(parents=True, exist_ok=True)
    #
    fieldnames = ["D", "which", "D2", "which2", "chi", "Delta", "Delta2", "time", "eng"]
    out = {"D" : D, "which" : which, "D2" : D2, "which2" : which2, "chi": chi,
           "Delta": Delta, "Delta2": Delta2, "time": time.time() - keep_time, "eng": eng}
    #
    fname = path / f"results_{r=}.csv"
    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)

    fname = path / f"results_{p=}_{r=}_{D=}_{which=}_{chi=}.npy"
    np.save(fname, ZZ)


# @ray.remote(num_cpus=4)
def run_test(p, r, D, which, D2, which2, chi):
    #
    print(f"Starting {p=} {r=} {D=} {which=} {D2=} {which2=} {chi=}")
    keep_time = time.time()
    #

    if p < 5:
        fname = f"./problems/transfer_Hamiltonian_ibm_kyiv_0_angles_16_0_p_5.pkl"
        data = np.load(fname, allow_pickle=True)
        data['gamma'] = [2.955984139930106, 2.8123696023056692] # data['gamma'][:p]
        data['beta'] = [0.48912372014861294, 0.2736708779840032] # data['beta'][:p]
    else:
        fname = f"./problems/transfer_Hamiltonian_ibm_kyiv_0_angles_16_0_p_{p}.pkl"
        data = np.load(fname, allow_pickle=True)
    HC = data["H"]
    HM = {(n,): 1 for n in range(127)}
    ops = NSpin12(sym='dense')
    #
    if r == 3:
        hex = MapHex127_r3()
    elif r == 4:
        # group spins
        hex = MapHex127_r4()
    else:
        raise ValueError()
    #
    HHC = hex.group_H(HC, ops.z, ops.I)
    HHM = hex.group_H(HM, ops.x, ops.I)
    #
    # initial product state
    #
    geometry = peps.SquareLattice(dims=hex.dims, boundary='obc')
    vectors = {site: ops.vec_x((1,) * Nloc) for site, Nloc in hex.s2N.items()}
    psi = peps.product_peps(geometry, vectors)
    #
    # run evolution
    #
    opts_svd_evol = {"D_total": D, "tol": 1e-12}
    env = peps.EnvBP(psi, which=which) if "BP" in which else peps.EnvNTU(psi, which=which)
    if "BP" in which:
        env.iterate_(max_sweeps=100, diff_tol=1e-8)
    #
    infoss = []
    t0 = time.time()
    for step, (gamma, beta) in enumerate(zip(data['gamma'], data['beta'])):

        gates = gates_from_HH2(HHC, gamma)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        infoss.append(infos)
        if "BP" in which:
            env.iterate_(max_sweeps=100, diff_tol=1e-8)
        gates = gates_from_HH2(HHM, beta)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        if "BP" in which:
            env.iterate_(max_sweeps=100, diff_tol=1e-8)
        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Step: {step}; Evolution time: {time.time() - t0:3.2f}; Truncation error: {Delta:0.6f}")
    #
    if D2 < D:
        print(" Truncating ")
        if which2 != which:
            env = peps.EnvBP(psi, which=which2) if "BP" in which2 else peps.EnvNTU(psi, which=which2)
            if "BP" in which2:
                env.iterate_(max_sweeps=50, diff_tol=1e-8)

        opts_svd_trun = {"D_total": D2}
        info2 = peps.truncate_(env, opts_svd=opts_svd_trun)
        if "BP" in which:
            env.iterate_(max_sweeps=50, diff_tol=1e-8)
        Delta2 = peps.accumulated_truncation_error([info2], statistics='mean')
        print(f"Truncation time: {time.time() - t0:3.2f}; Truncation error: {Delta2:0.6f}")
    else:
        Delta2 = 0.
    #
    if chi > 1:
        print(" Calculating energy from boundary MPS")
        opts_svd_mps = {"D_total":  chi}
        env_mps = peps.EnvBoundaryMPS(psi, opts_svd=opts_svd_mps, setup='tlbr')
        #
        eng, ZZ = measure_H_mps(env_mps, HHC)
        print(f"Energy for {p=}: {eng:0.4f}")
    else:
        print(" Calculating energy from BP")
        if (D2 < D and 'BP' not in which2) or 'BP' not in which:
            env = peps.EnvBP(psi, which=which2)
            env.iterate_(max_sweeps=50, diff_tol=1e-8)
        eng, ZZ = measure_H_ctm(env, HHC)
    #
    path = Path(f"./results/{data['Hamiltonian_name']}/{data['angles_name']}/{p=}/")
    path.mkdir(parents=True, exist_ok=True)
    #
    fieldnames = ["D", "which", "D2", "which2", "chi", "Delta", "Delta2", "time", "eng"]
    out = {"D" : D, "which" : which, "D2" : D2, "which2" : which2, "chi": chi,
           "Delta": Delta, "Delta2": Delta2, "time": time.time() - keep_time, "eng": eng}
    #
    fname = path / f"results_{r=}.csv"
    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)

    fname = path / f"results_{p=}_{r=}_{D=}_{which=}_{chi=}.npy"
    np.save(fname, ZZ)


if __name__ == '__main__':
    ray.init()
    refs = []
    for p in [2]:
        for r in [3]:
            for D in [4]:
                for which in ["BP"]:
                    for chi in [16, 32]:
                        D2 = D
                        which2 = which
                        job = run_test.remote(p, r, D, which, D2, which2, chi)
                        refs.append(job)
    ray.get(refs)
