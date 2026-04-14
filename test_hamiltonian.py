import os
import numpy as np
import matplotlib.pyplot as plt

from hamiltonian import ScalarFieldHamiltonian, Unitary

t    = 2.
N_s  = 3
m    = 1.
a    = 1.
phi_max   = 1.0
delta_phi = 2/3
lambd_list = np.arange(0, 4.0, 0.5)


out_dir = f"evolved_states/t={t}"
os.makedirs(out_dir, exist_ok=True)


for lam in lambd_list:
    h = ScalarFieldHamiltonian(n_sites=N_s, phi_max=phi_max, delta_phi=delta_phi, m=m, lam=lam, a=a)
    v0 = h.config_to_state(h.id_to_config(0))

    print(f'lambda={lam}')

    U  = Unitary(t=t)

    H = h.hamiltonian
    U.compute(H)

    v_t = U.evolve(v0)

    fname = f"N_s={N_s}_m={m}_phimax={phi_max}_lam{lam:.2f}_t={t}.npy"
    np.save(os.path.join(out_dir, fname), v_t)
    print(' out = ', h.state_to_square(v_t))