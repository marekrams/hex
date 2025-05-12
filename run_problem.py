import ast
import csv
import os
from pathlib import Path
# import ray
import time
import time
import numpy as np
import yastn.tn.fpeps as peps
from routines import NSpin12, map_hex, gates_from_HH, gates_from_HH2, measure_H_ctm, measure_H_mps
from itertools import product


def run_evolution(qpu, ang, les, p, D):
    #
    print(f"Starting {qpu=} {ang=} {les=} {p=} {D=}")
    #
    Nspins = {"kyiv": 127, "torino": 133, "fez": 156}
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    HC = data["H"]
    HM = {(n,): 1 for n in range(Nspins[qpu])}
    #
    file = open(f"./problems/QAOA_angles/{ang}.txt", "r")
    angles = ast.literal_eval(file.read())
    file.close()
    angles = angles[les][p - 1]
    betas   = angles[:len(angles)//2]
    gammas  = angles[len(angles)//2:]
    #    
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
    env = peps.EnvBP(psi)
    env.iterate_(max_sweeps=100, diff_tol=1e-8)
    #
    infoss = []
    t0 = time.time()
    for step, (gamma, beta) in enumerate(zip(gammas, betas)):
        gates = gates_from_HH2(HHC, gamma)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        infoss.append(infos)
        env.iterate_(max_sweeps=100, diff_tol=1e-8)
        gates = gates_from_HH2(HHM, beta)
        infos = peps.evolution_step_(env, gates, opts_svd=opts_svd_evol, symmetrize=False)
        env.iterate_(max_sweeps=100, diff_tol=1e-8)
        Delta = peps.accumulated_truncation_error(infoss, statistics='mean')
        print(f"Step: {step}; Evolution time: {time.time() - t0:3.2f}; Truncation error: {Delta:0.6f}")
    #
    path = Path(f"./results/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    path.mkdir(parents=True, exist_ok=True)
    #
    fname = path / f"state_{D=}.npy"
    d = psi.save_to_dict()
    np.save(fname, d, allow_pickle=True)
    #
    # d = env.save_to_dict()
    # np.save(fname, d, allow_pickle=True)
    # fname = path / f"env_{D=}_BP.npy"
    #
    eng, ZZ = measure_H_ctm(env, HHC)
    fieldnames = ["D", "env", "chi", "eng"]
    out = {"D" : D, "env": "BP", "chi": 1, "eng": eng}
    fname = path / f"results.csv"
    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)
    fname = path / f"results_{D=}_BP_chi=1.npy"
    np.save(fname, ZZ)


def run_env(qpu, ang, les, p, D, which, chi):

    print(f"Starting {qpu=} {ang=} {les=} {p=} {D=} {which=} {chi=}")
    #
    Nspins = {"kyiv": 127, "torino": 133, "fez": 156}
    hex = map_hex(N=Nspins[qpu])
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    ops = NSpin12(sym='dense')
    HC = data["H"]
    HHC = hex.group_H(HC, ops.z, ops.I)
    #
    path = Path(f"./results/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    fname = path / f"state_{D=}.npy"
    d = np.load(fname, allow_pickle=True).item()
    psi = peps.load_from_dict(config=ops.config, d=d)
    
    if which == 'MPS':
        opts_svd_mps = {"D_total":  chi}
        env = peps.EnvBoundaryMPS(psi, opts_svd=opts_svd_mps, setup='tlbr')
        #
        eng, ZZ = measure_H_mps(env, HHC)
        print(f"Energy for {p=}: {eng:0.4f}")
    
    fname = path / f"env_{D=}_{which}_{chi=}.npy"
    d = env.save_to_dict()
    np.save(fname, d, allow_pickle=True)
    #
    fieldnames = ["D", "env", "chi", "eng"]
    out = {"D" : D, "env": "BP", "chi": 1, "eng": eng}
    fname = path / f"results.csv"
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        writer.writerow(out)
    fname = path / f"results_{D=}_BP_chi=1.npy"
    np.save(fname, ZZ)


def run_sample(qpu, ang, les, p, D, which, chi, nsamples):

    print(f"Starting {qpu=} {ang=} {les=} {p=} {D=} {which=} {chi=} {nsamples=}")
    #
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    ops = NSpin12(sym='dense')
    path = Path(f"./results/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    fname = path / f"env_{D=}_{which}_{chi=}.npy"
    d = np.load(fname, allow_pickle=True)
    env = peps.load_from_dict(config=ops.config, )
    #
    vecs = {site: [ops.vec_z(pp) for pp in product([[-1, 1]] * Nloc)] for site, Nloc in hex.s2N.items()}
    xx = env.sample(number=5, projectors=vecs)
    #
    print(xx)


if __name__ == '__main__':
    refs = []
    for p in [2]:
        for D in [4]:
            # run_evolution('kyiv', 16, 0, p, D)
            run_env('kyiv', 16, 0, p, D, "MPS", 2)
