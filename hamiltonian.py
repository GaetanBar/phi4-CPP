import numpy as np
import scipy 

class ScalarFieldHamiltonian:


    def __init__(self, n_sites: int, phi_max: float, delta_phi: float,
                 m: float, lam: float, a: float = 1.0):
        self.n_sites = n_sites
        self.phi_max = phi_max
        self.delta_phi = delta_phi
        self.m = m
        self.lam = lam
        self.a = a

        self.N_phi = round(2 * phi_max / delta_phi) + 1
        self.phi_vals = np.linspace(-phi_max, phi_max, self.N_phi)
        self.dim = self.N_phi ** n_sites
        
        self.basis = self._basis()  
        self._H = None  


    #Helpers

    def _embed(self, op: np.ndarray, site: int) -> np.ndarray:
        """Embed a single-site (N_phi x N_phi) operator at `site`."""
        left = np.eye(self.N_phi ** site)
        right = np.eye(self.N_phi ** (self.n_sites - site - 1))
        return np.kron(left, np.kron(op, right))

    def _embed2(self, op2: np.ndarray, site: int) -> np.ndarray:
        """Embed a two-site (N_phi^2 x N_phi^2) operator starting at `site`."""
        left = np.eye(self.N_phi ** site)
        right = np.eye(self.N_phi ** (self.n_sites - site - 2))
        return np.kron(left, np.kron(op2, right))


    #State manipulation methods

    def _basis(self) -> np.ndarray:
        """Generate the full basis of field configurations in lexicographic order."""

        grids = np.meshgrid(*[self.phi_vals] * self.n_sites, indexing='ij')
        return np.array(grids).reshape(self.n_sites, -1).T

    def id_to_config(self, idx: int)->np.ndarray:
        return self.basis[idx]
    
    def config_to_id(self, state: np.ndarray) -> int:
        indices = np.round((np.asarray(state) + self.phi_max) / self.delta_phi).astype(int)
        return int(np.ravel_multi_index(indices, (self.N_phi,) * self.n_sites))
    
    def config_to_state(self, config: np.ndarray) -> np.ndarray:
        psi = np.zeros(self.dim)
        psi[self.config_to_id(config)] = 1.0
        return psi
    
    def state_to_square(self, state: np.ndarray) -> np.ndarray:
        return (state * state.conj()).real

    #Matrix Construction 

    def _kinetic_local(self) -> np.ndarray:
        #ok
        """
        pi^2 / 2 in the field basis using a second-order finite-difference
        T[k,k]   =  1 / delta_phi^2
        T[k,k±1] = -1 / (2 * delta_phi^2)
        """
        n = self.N_phi
        d = self.delta_phi
        T = np.zeros((n, n))
        diag_val = 1.0 / d**2
        off_val = -1.0 / (2.0 * d**2)
        for k in range(n):
            T[k, k] = diag_val
            if k > 0:
                T[k, k - 1] = off_val
            if k < n - 1:
                T[k, k + 1] = off_val
        
        #BCs follow PHYSICAL REVIEW A 99, 052335 (2019)
        T[0, n-1] = off_val
        T[n-1, 0] = off_val
        return T

    def _potential_local(self) -> np.ndarray:
        #ok
        phi = self.phi_vals
        v = 0.5 * self.m**2 * phi**2 + (self.lam / 24.0) * phi**4
        return np.diag(v)


    def _build(self) -> np.ndarray:
        T_loc = self._kinetic_local()
        V_loc = self._potential_local()
        Phi = np.diag(self.phi_vals)
        Phi2 = np.diag(self.phi_vals**2)
        I = np.eye(self.N_phi)

        H = np.zeros((self.dim, self.dim))

        for j in range(self.n_sites):
            H += self._embed(T_loc, j)
            H += self._embed(V_loc, j)

        # Gradient interaction (phi_j - phi_{j+1})^2 / (2 a^2)
        if self.n_sites > 1:
            W_diag = np.kron(Phi2, I) + np.kron(I, Phi2) - 2.0 * np.kron(Phi, Phi)
            W_diag /= 2.0 * self.a**2
            for j in range(self.n_sites - 1):
                H += self._embed2(W_diag, j)

        return H


    @property
    def hamiltonian(self) -> np.ndarray:

        if self._H is None:
            self._H = self._build()
        return self._H

    #Data Management 

    def save_matrix(self, path: str) -> None:
        #Need to buid H first
        H = self.hamiltonian
        i, j = np.triu_indices(H.shape[0])
        np.savez_compressed(path, upper=H[i, j], dim=np.array(H.shape[0]))

    @staticmethod
    def load_matrix(path: str) -> np.ndarray:
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        dim = int(data["dim"])
        H = np.zeros((dim, dim))
        i, j = np.triu_indices(dim)
        H[i, j] = data["upper"]
        H[j, i] = data["upper"]  # mirror to lower triangle
        return H


class Unitary:
    #A class for exp(iHt), contains time evolution method. 
    def __init__(self, t):
        self.t = t
        self.U = None

    def compute(self, H):
        self.U = scipy.linalg.expm(-1j * H * self.t)    

    def evolve(self, psi0):
        if self.U is None:
            raise ValueError("Unitary not computed yet. Call compute(H) first.")
        return self.U @ psi0    