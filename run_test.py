import csv
import os
from pathlib import Path
import ray
import time
from tqdm import tqdm  # progressbar
import time
import numpy as np
import yastn.tn.mps as mps
import yastn.tn.fpeps as peps
from routines import NSpin12, MapHex127_r4, MapHex127_r3, gates_from_HH, measure_H_ctm, measure_H_mps


@ray.remote
def run_test(p, r, D, which, D2, which2, chi):
    #
    print(f"Starting {p=} {r=} {D=} {which=} {D2=} {which2=} {chi=}")
    keep_time = time.time()
    #
    fname = "./problems/sherbrooke0v3.pkl"
    data = np.load(fname, allow_pickle=True)
    HCref = data["H"]
    #
    fname = f"./problems/transfsherbrooke0_p{p}.pkl"
    data = np.load(fname, allow_pickle=True)
    data.keys()
    #
    HC = {k: v for k, v in zip(HCref.keys(), data["H"])}
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
    env = peps.EnvLBP(psi, which=which) if "LBP" in which else peps.EnvNTU(psi, which=which)
    if "LBP" in which:
        env.lbp_(max_sweeps=50, diff_tol=1e-8)
    #
    infoss = []
    t0 = time.time()
    for step, (gamma, beta) in enumerate(zip(data['gamma'], data['beta'])):
        gates = gates_from_HH(HHC, gamma)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        infoss.append(infos)
        if "LBP" in which:
            env.lbp_(max_sweeps=50, diff_tol=1e-8)
        gates = gates_from_HH(HHM, beta)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        if "LBP" in which:
            env.lbp_(max_sweeps=50, diff_tol=1e-8)
        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Step: {step}; Evolution time: {time.time() - t0:3.2f}; Truncation error: {Delta:0.6f}")
    #
    if D2 < D:
        print(" Truncating ")
        if which2 != which:
            env = peps.EnvLBP(psi, which=which2) if "LBP" in which2 else peps.EnvNTU(psi, which=which2)
            if "LBP" in which2:
                env.lbp_(max_sweeps=50, diff_tol=1e-8)

        opts_svd_trun = {"D_total": D2}
        info2 = peps.truncate_(env, opts_svd=opts_svd_trun)
        if "LBP" in which:
            env.lbp_(max_sweeps=50, diff_tol=1e-8)
        Delta2 = peps.accumulated_truncation_error([info2], statistics='mean')
        print(f"Truncation time: {time.time() - t0:3.2f}; Truncation error: {Delta2:0.6f}")
    else:
        Delta2 = 0.
    #
    if chi > 1:
        print(" Calculating nergy from boundary MPS")
        opts_svd_mps = {"D_total":  chi}
        env_mps = peps.EnvBoundaryMPS(psi, opts_svd=opts_svd_mps, setup='tlbr')
        #
        eng, ZZ = measure_H_mps(env_mps, HHC)
        print(f"Energy for {p=}: {eng:0.4f}")
    else:
        print(" Calculating nergy from BP")
        if (D2 < D and 'LBP' not in which2) or 'LBP' not in which:
            env = peps.EnvLBP(psi, which=which2)
            env.lbp_(max_sweeps=50, diff_tol=1e-8)
        eng, ZZ = measure_H_ctm(env, HHC)
    #
    path = Path(f"./results/{p=}/")
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


if __name__ == '__main__':
    ray.init()
    refs = []
    for p in [5, 10]:
        for r in [3, 4]:
            for D in [16, 32]:
                for which in ["LBP"]:
                    D2 = D
                    which2 = which
                    chi = 1
                    # run_test(p, r, D, which, D2, which2, chi)
                    job = run_test.remote(p, r, D, which, D2, which2, chi)
                    refs.append(job)
    ray.get(refs)
    # run_test(p=5, r=3, D=8, which='LBP', D2=8, which2='LBP', chi=1)