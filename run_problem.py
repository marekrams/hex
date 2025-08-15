import ast
import csv
import os
from pathlib import Path
import ray
import time
import time
import numpy as np
import yastn.tn.fpeps as peps
from yastn import tensordot
from routines import NSpin12, map_hex, gates_from_HH, gates_from_HH2, measure_H_ctm, measure_H_mps
from itertools import product


Nspins = {"kyiv": 127, "torino": 133, "fez": 156}


@ray.remote(num_cpus=1)
def run_evolution(qpu, ang, les, p, D):
    #
    print(f"Evolution {qpu=} {ang=} {les=} {p=} {D=}")
    #
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    HC = data["H"]
    HM = {(n,): 1 for n in range(Nspins[qpu])}
    #
    fname2 = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/results_{D=}_BP_chi=1.npy")
    

    fname = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/state_{D=}.npy")
    if fname2.is_file():
        print(f"Evolution {qpu=} {ang=} {les=} {p=} {D=} was already done!")
        return True
    #
    file = open(f"./problems/QAOA_angles/{ang}.txt", "r")
    angles = ast.literal_eval(file.read())
    file.close()
    if len(angles[les]) < p:
        return False

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
        # print(f"Step: {step}; Evolution time: {time.time() - t0:3.2f}; Truncation error: {Delta:0.6f}")
    #
    path = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    path.mkdir(parents=True, exist_ok=True)
    #
    fname = path / f"state_{D=}.npy"
    dd = {"psi": psi.save_to_dict(), "infos": infos, "Delta": Delta}
    np.save(fname, dd, allow_pickle=True)
    #
    fname = path / f"env_{D=}_BP.npy"
    d = env.save_to_dict()
    np.save(fname, d, allow_pickle=True)
    #
    eng, ZZ = measure_H_ctm(env, HHC)
    fieldnames = ["D", "Delta", "env", "chi", "eng"]
    out = {"D" : D, "Delta": Delta, "env": "BP", "chi": 1, "eng": eng}
    fname = path / f"results.csv"
    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)
    fname = path / f"results_{D=}_BP_chi=1.npy"
    np.save(fname, ZZ)
    return True


@ray.remote(num_cpus=18)
def run_env(done, qpu, ang, les, p, D, which, chi):
    #
    print(f"Environment {qpu=} {ang=} {les=} {p=} {D=} {which=} {chi=}")
    #
    hex = map_hex(N=Nspins[qpu])
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    ops = NSpin12(sym='dense')
    HC = data["H"]
    HHC = hex.group_H(HC, ops.z, ops.I)
    #
    fname = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/env_{D=}_{which}_{chi=}.npy")
    if fname.is_file():
        print(f"Environment {qpu=} {ang=} {les=} {p=} {D=} {which=} {chi=} was already done")
        return True
    #
    path = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    fname = path / f"state_{D=}.npy"
    dd = np.load(fname, allow_pickle=True).item()
    psi = peps.load_from_dict(config=ops.config, d=dd['psi'])
    
    if which == 'MPS':
        opts_svd_mps = {"D_total":  chi}
        opts_var = {'max_sweeps': 10, 'method': '1site', 'normalize': False, "Schmidt_tol": 1e-6}

        env = peps.EnvBoundaryMPS(psi, opts_svd=opts_svd_mps, opts_var = opts_var, setup='tlbr')
        info = {}
        eng, ZZ = measure_H_mps(env, HHC)
        # print(f"Energy for {p=}: {eng:0.4f}")

    if which == 'CTM':
        opts_svd_ctm = {"D_total":  chi}
        env = peps.EnvCTM(psi, init='eye')
        info = env.ctmrg_(max_sweeps=100, opts_svd=opts_svd_ctm, corner_tol=1e-8, method='hex') #
        eng, ZZ = measure_H_ctm(env, HHC)
        # print(f"Energy for {p=}: {eng:0.4f}")
    
    fname = path / f"env_{D=}_{which}_{chi=}.npy"

    geometry = peps.SquareLattice(dims=hex.dims, boundary='obc')
    vectors = {site: ops.vec_x((1,) * Nloc) for site, Nloc in hex.s2N.items()}
    env.psi = peps.product_peps(geometry, vectors)

    d = env.save_to_dict()
    d['info'] = info
    np.save(fname, d, allow_pickle=True)
    #
    fieldnames = ["D", "Delta", "env", "chi", "eng"]
    out = {"D" : D,  "Delta": dd["Delta"], "env": which, "chi": chi, "eng": eng}
    fname = path / f"results.csv"
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        writer.writerow(out)
    fname = path / f"results_{D=}_{which}_{chi=}.npy"
    np.save(fname, ZZ)
    return True


@ray.remote(num_cpus=1)
def run_sample(done, qpu, ang, les, p, D, which, chi, number):
    #
    print(f"Sampling {qpu=} {ang=} {les=} {p=} {D=} {which=} {chi=} {number=}")
    hex = map_hex(N=Nspins[qpu])
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    ops = NSpin12(sym='dense')
    path = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    fname = path / f"env_{D=}_{which}_{chi=}.npy" if which != 'BP' else path / f"env_{D=}_{which}.npy"
    #    
    d = np.load(fname, allow_pickle=True).item()
    fname = path / f"state_{D=}.npy"
    dd = np.load(fname, allow_pickle=True).item()
    d['psi'] = dd['psi']
    env = peps.load_from_dict(config=ops.config, d=d)
 
    inds = {site: [pp for pp in product(*([[-1, 1]] * Nloc))] for site, Nloc in hex.s2N.items()}
    vecs = {site: [ops.vec_z(pp) for pp in ind] for site, ind in inds.items()}
    opss = {site: [tensordot(vv, vv.conj(), axes=((), ())) for vv in vvs] for site, vvs in vecs.items()}
    evss = {site: [env.measure_1site(op, site) for op in ops] for site, ops in opss.items()}
    #
    for site, evs in evss.items():
        arg = sorted(range(len(evs)), key=evs.__getitem__, reverse=True)
        evss[site] = [evss[site][i] for i in arg]
        opss[site] = [opss[site][i] for i in arg]
        vecs[site] = [vecs[site][i] for i in arg]
        inds[site] = [inds[site][i] for i in arg]

    t0 = time.time()
    samples, energies = [], []
    out, probabilities = env.sample(projectors=opss, return_probabilities=True, number=number)
    samples = np.zeros((number, Nspins[qpu]), dtype=np.int8)
    
    for s, vss in out.items():
        for j, vs in enumerate(vss):
            for i, v in zip(hex.s2i[s], inds[s][vs]):
                samples[j, i] = v

    energies = np.zeros(number)
    for k, v in data["H"].items():
        val = np.ones(number, dtype=np.int8)
        for i in k:
            val *= samples[:, i]
        energies += v * val
        
    d = {"samples": samples,
         "probabilities": probabilities,
         "energies": energies}
    fname = path / f"samples_{D=}_{which}_{chi=}.npy"
    np.save(fname, d, allow_pickle=True)

    print(f"Sampling {number} took {time.time() - t0:3.2f}")



@ray.remote(num_cpus=2)
def run_evolution_sample(qpu, ang, les, p, D, number):
    #
    print(f"Evolution {qpu=} {ang=} {les=} {p=} {D=}")
    #
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    HC = data["H"]
    HM = {(n,): 1 for n in range(Nspins[qpu])}
    #
    fname2 = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/results_{D=}_BP_chi=1.npy")
    

    fname = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/state_{D=}.npy")
    # if fname2.is_file():
    #     print(f"Evolution {qpu=} {ang=} {les=} {p=} {D=} was already done!")
    #     return True
    #
    file = open(f"./problems/QAOA_angles/{ang}.txt", "r")
    angles = ast.literal_eval(file.read())
    file.close()
    if len(angles[les]) < p:
        return False
    #
    path = Path(f"./results_fig15/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    path.mkdir(parents=True, exist_ok=True)
    fname = path / f"samples_{D=}_BP_chi=1.npy"
    if os.path.isfile(fname):
        print(f"Sampling for {qpu=} {ang=} {les=} {p=} {D=} was already done !!!! ")
        return True


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
        # print(f"Step: {step}; Evolution time: {time.time() - t0:3.2f}; Truncation error: {Delta:0.6f}")
    #
    #
    fname = path / f"state_{D=}.npy"
    dd = {"psi": psi.save_to_dict(), "infos": infos, "Delta": Delta}
    #np.save(fname, dd, allow_pickle=True)
    #
    fname = path / f"env_{D=}_BP.npy"
    d = env.save_to_dict()
    #np.save(fname, d, allow_pickle=True)
    #
    eng, ZZ = measure_H_ctm(env, HHC)
    fieldnames = ["D", "Delta", "env", "chi", "eng"]
    out = {"D" : D, "Delta": Delta, "env": "BP", "chi": 1, "eng": eng}
    fname = path / f"results.csv"
    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)
    fname = path / f"results_{D=}_BP_chi=1.npy"
    np.save(fname, ZZ)
    #
    inds = {site: [pp for pp in product(*([[-1, 1]] * Nloc))] for site, Nloc in hex.s2N.items()}
    vecs = {site: [ops.vec_z(pp) for pp in ind] for site, ind in inds.items()}
    opss = {site: [tensordot(vv, vv.conj(), axes=((), ())) for vv in vvs] for site, vvs in vecs.items()}
    evss = {site: [env.measure_1site(op, site) for op in ops] for site, ops in opss.items()}
    #
    for site, evs in evss.items():
        arg = sorted(range(len(evs)), key=evs.__getitem__, reverse=True)
        evss[site] = [evss[site][i] for i in arg]
        opss[site] = [opss[site][i] for i in arg]
        vecs[site] = [vecs[site][i] for i in arg]
        inds[site] = [inds[site][i] for i in arg]

    t0 = time.time()
    samples, energies = [], []
    out, probabilities = env.sample(projectors=opss, return_probabilities=True, number=number)
    samples = np.zeros((number, Nspins[qpu]), dtype=np.int8)
    
    for s, vss in out.items():
        for j, vs in enumerate(vss):
            for i, v in zip(hex.s2i[s], inds[s][vs]):
                samples[j, i] = v

    energies = np.zeros(number)
    for k, v in data["H"].items():
        val = np.ones(number, dtype=np.int8)
        for i in k:
            val *= samples[:, i]
        energies += v * val
        
    d = {"samples": samples,
         "probabilities": probabilities,
         "energies": energies}
    fname = path / f"samples_{D=}_BP_chi=1.npy"
    np.save(fname, d, allow_pickle=True)

    print(f"Sampling {number} took {time.time() - t0:3.2f}")




@ray.remote(num_cpus=2)
def run_evolution_sample_inhomo(qpu, ang, les, p, D, number):
    #
    print(f"Evolution {qpu=} {ang=} {les=} {p=} {D=}")
    #
    fname = f"./problems/transfer_Hamiltonian_ibm_{qpu}_0.pkl"
    data = np.load(fname, allow_pickle=True)
    HC = data["H"]
    #
    fname = Path(f"./results_inhomo/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/state_{D=}.npy")
    # if fname2.is_file():
    #     print(f"Evolution {qpu=} {ang=} {les=} {p=} {D=} was already done!")
    #     return True
    #
    file = open(f"./problems/QAOA_angles/{ang}.txt", "r")
    angles = ast.literal_eval(file.read())
    file.close()
    if len(angles[les]) < p:
        return False
    #
    path = Path(f"./results_inhomo/{data['Hamiltonian_name']}/{ang}_{les}/{p=}/")
    path.mkdir(parents=True, exist_ok=True)
    fname = path / f"samples_{D=}_BP_chi=1.npy"
    if os.path.isfile(fname):
        print(f"Sampling for {qpu=} {ang=} {les=} {p=} {D=} was already done !!!! ")
        return True


    angles = angles[les][p - 1]
    betas   = angles[:len(angles)//2]
    gammas  = angles[len(angles)//2:]
    #    
    ops = NSpin12(sym='dense')
    #
    hex = map_hex(N=Nspins[qpu])
    #
    HHC = hex.group_H(HC, ops.z, ops.I)
    #

    xys = [v[0] for v in hex.i2s.values()]
    cx = np.mean([v[0] for v in xys])
    cy = np.mean([v[1] for v in xys])

    md = np.max([np.sqrt((v[0] - cx) ** 2 + (v[1] - cy) ** 2) for v in xys])
    # md = np.max([v[0] for v in xys]) - np.min([v[0] for v in xys])
    HM = {}
    for k, v in hex.i2s.items():
        d = np.sqrt((v[0][0] - cx) ** 2 + (v[0][1] - cy) ** 2)
        # d = v[0][0] - np.min([v[0] for v in xys])
        HM[(k,)] = -(d / md) + 1.5
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
        # print(f"Step: {step}; Evolution time: {time.time() - t0:3.2f}; Truncation error: {Delta:0.6f}")
    #
    #
    fname = path / f"state_{D=}.npy"
    dd = {"psi": psi.save_to_dict(), "infos": infos, "Delta": Delta}
    #np.save(fname, dd, allow_pickle=True)
    #
    fname = path / f"env_{D=}_BP.npy"
    d = env.save_to_dict()
    #np.save(fname, d, allow_pickle=True)
    #
    eng, ZZ = measure_H_ctm(env, HHC)
    fieldnames = ["D", "Delta", "env", "chi", "eng"]
    out = {"D" : D, "Delta": Delta, "env": "BP", "chi": 1, "eng": eng}
    fname = path / f"results.csv"
    file_exists = os.path.isfile(fname)
    with open(fname, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out)
    fname = path / f"results_{D=}_BP_chi=1.npy"
    np.save(fname, ZZ)
    #
    inds = {site: [pp for pp in product(*([[-1, 1]] * Nloc))] for site, Nloc in hex.s2N.items()}
    vecs = {site: [ops.vec_z(pp) for pp in ind] for site, ind in inds.items()}
    opss = {site: [tensordot(vv, vv.conj(), axes=((), ())) for vv in vvs] for site, vvs in vecs.items()}
    evss = {site: [env.measure_1site(op, site) for op in ops] for site, ops in opss.items()}
    #
    for site, evs in evss.items():
        arg = sorted(range(len(evs)), key=evs.__getitem__, reverse=True)
        evss[site] = [evss[site][i] for i in arg]
        opss[site] = [opss[site][i] for i in arg]
        vecs[site] = [vecs[site][i] for i in arg]
        inds[site] = [inds[site][i] for i in arg]

    t0 = time.time()
    samples, energies = [], []
    out, probabilities = env.sample(projectors=opss, return_probabilities=True, number=number)
    samples = np.zeros((number, Nspins[qpu]), dtype=np.int8)
    
    for s, vss in out.items():
        for j, vs in enumerate(vss):
            for i, v in zip(hex.s2i[s], inds[s][vs]):
                samples[j, i] = v

    energies = np.zeros(number)
    for k, v in data["H"].items():
        val = np.ones(number, dtype=np.int8)
        for i in k:
            val *= samples[:, i]
        energies += v * val
        
    d = {"samples": samples,
         "probabilities": probabilities,
         "energies": energies}
    fname = path / f"samples_{D=}_BP_chi=1.npy"
    np.save(fname, d, allow_pickle=True)

    print(f"Sampling {number} took {time.time() - t0:3.2f}")





if __name__ == '__main__':
    ray.init()

    number = 10000
    refs = []
    evo_job = {}
    for ang in ['16']:
        for les in [0,1,2,3,4,5,6,7,8,9,'pos.','neg.']:
            for p in range(1, 60):
                for D in [32]:            
                    for qpu in ["torino"]:
                        # evo_job[qpu, ang, les, p, D] = run_evolution.remote(qpu, ang, les, p, D)
                        # refs.append(evo_job[qpu, ang, les, p, D])
                        # for chi in [2, 4, 8]:
                        #     job = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "CTM", chi)
                        #     evo_job[qpu, ang, les, p, D, "CTM", chi] = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "CTM", chi)
                        #     refs.append(job)
                        # # #     evo_job[qpu, ang, les, p, D, "MPS", chi] = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "MPS", chi)
                        #     refs.append(evo_job[qpu, ang, les, p, D, "MPS", chi])

                        job = run_evolution_sample.remote(qpu, ang, les, p, D, number)
                        # job = run_sample.remote(True, qpu, ang, les, p, D, "BP", 1, number)
                        refs.append(job)

                        # for chi in [2]:
                        #     # evo_job[qpu, ang, les, p, D, "CTM", chi]
                        #     job = run_sample.remote(True, qpu, ang, les, p, D, "CTM", chi, number)
                        #     refs.append(job)
                        #     # job  = run_sample.remote(evo_job[qpu, ang, les, p, D, "MPS", chi], qpu, ang, les, p, D, "MPS", chi, number)
                        #     # refs.append(job)
        
    ray.get(refs)
                        


# if __name__ == '__main__':
#     ray.init()

#     number = 1000
#     refs = []
#     evo_job = {}
#     for ang in ['16']:
#         for les in [0]:
#             for p in [20]:
#                 for D in [32]:            
#                     for qpu in ["fez"]:
#                         evo_job[qpu, ang, les, p, D] = run_evolution.remote(qpu, ang, les, p, D)
#                         refs.append(evo_job[qpu, ang, les, p, D])
#                         for chi in [4,]:
#                             job = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "CTM", chi)
#                             refs.append(job)
#                             # job = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "MPS", chi)
#                             # refs.append(job)
                            
#                         # for chi in [2]:
#                         #     # evo_job[qpu, ang, les, p, D, "CTM", chi]
#                         #     job = run_sample.remote(True, qpu, ang, les, p, D, "CTM", chi, number)
#                         #     refs.append(job)
#                         #     # job  = run_sample.remote(evo_job[qpu, ang, les, p, D, "MPS", chi], qpu, ang, les, p, D, "MPS", chi, number)
#                         #     # refs.append(job)
        
#     ray.get(refs)




# if __name__ == '__main__':
#     ray.init()

#     number = 1000
#     refs = []
#     evo_job = {}
#     for ang in ['16']:
#         for les in [5,]:
#             for p in [32]: # range(1, 20):
#                 for D in [16]:            
#                     for qpu in ["kyiv", "torino", "fez"]:
#                         # evo_job[qpu, ang, les, p, D] = run_evolution.remote(qpu, ang, les, p, D)
#                         # refs.append(evo_job[qpu, ang, les, p, D])
#                         # for chi in [2, 4, 8]:
#                         #     job = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "CTM", chi)
#                         #     evo_job[qpu, ang, les, p, D, "CTM", chi] = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "CTM", chi)
#                         #     refs.append(job)
#                         # # #     evo_job[qpu, ang, les, p, D, "MPS", chi] = run_env.remote(evo_job[qpu, ang, les, p, D], qpu, ang, les, p, D, "MPS", chi)
#                         #     refs.append(evo_job[qpu, ang, les, p, D, "MPS", chi])

#                         job = run_evolution_sample_inhomo.remote(qpu, ang, les, p, D, number)
#                         # job = run_sample.remote(True, qpu, ang, les, p, D, "BP", 1, number)
#                         refs.append(job)

#                         # for chi in [2]:
#                         #     # evo_job[qpu, ang, les, p, D, "CTM", chi]
#                         #     job = run_sample.remote(True, qpu, ang, les, p, D, "CTM", chi, number)
#                         #     refs.append(job)
#                         #     # job  = run_sample.remote(evo_job[qpu, ang, les, p, D, "MPS", chi], qpu, ang, les, p, D, "MPS", chi, number)
#                         #     # refs.append(job)
        
#     ray.get(refs)
                       