#Author: Francesco Fiorini

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import math  # For math.comb in Kraus operator generation

# Enable LaTeX rendering and set font preferences for better visualization
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# ### Core Functions

def apply_loss_channel(rho, eta, rho_env, mode_index, total_modes):
    """
    Apply a Gaussian loss channel to a specific photonic mode in a quantum system.

    Args:
        rho (Qobj): Input density matrix (photonic modes + qubit).
        eta (float): Transmissivity of the channel (0 <= eta <= 1).
        rho_env (Qobj): Environment state (e.g., vacuum or thermal state).
        mode_index (int): Index of the mode to apply the channel to (0-based).
        total_modes (int): Total number of photonic modes in the system.

    Returns:
        Qobj: Output density matrix after applying the loss channel.
    """
    dim_photonic = rho.dims[0][0]  # Dimension of each photonic mode
    dim_qubit = rho.dims[0][-1]    # Qubit dimension (2)
    aux_dim = rho_env.dims[0][0]   # Dimension of the auxiliary (environment) mode

    # Tensor the system with the environment state
    rho_total = qt.tensor(rho, rho_env)

    # Define operators
    a = qt.destroy(dim_photonic)  # Annihilation operator for the system mode
    b = qt.destroy(aux_dim)       # Annihilation operator for the environment mode
    I_photonic = qt.qeye(dim_photonic)
    I_qubit = qt.qeye(dim_qubit)
    I_aux = qt.qeye(aux_dim)

    # Construct operators in the full Hilbert space
    a_op = qt.tensor([I_photonic] * mode_index + [a] + [I_photonic] * (total_modes - mode_index - 1) + [I_qubit] + [I_aux])
    b_op = qt.tensor([I_photonic] * total_modes + [I_qubit] + [b])

    # Beamsplitter Hamiltonian: H = theta (a† b - a b†), where cos(theta) = sqrt(eta)
    theta = np.arccos(np.sqrt(eta))
    H = theta * (a_op.dag() * b_op - a_op * b_op.dag())

    # Unitary evolution: U = exp(H)
    U = H.expm()

    # Apply unitary and trace out the auxiliary mode
    rho_total = U * rho_total * U.dag()
    rho_out = rho_total.ptrace(list(range(total_modes + 1)))  # Keep system + qubit
    return rho_out

def generate_loss_kraus_ops(eta, dim_photonic):
    """
    Generate Kraus operators for a pure loss channel with transmissivity eta.

    Args:
        eta (float): Transmissivity (0 <= eta <= 1).
        dim_photonic (int): Dimension of the photonic Hilbert space.

    Returns:
        list: List of Kraus operators (Qobj).
    """
    kraus_ops = []
    for k in range(dim_photonic):
        K_k = qt.Qobj(np.zeros((dim_photonic, dim_photonic)))
        for n in range(k, dim_photonic):
            coeff = np.sqrt(math.comb(n, k)) * (eta ** ((n - k) / 2)) * ((1 - eta) ** (k / 2))
            K_k += coeff * qt.basis(dim_photonic, n - k) * qt.basis(dim_photonic, n).dag()
        kraus_ops.append(K_k)
    return kraus_ops

def apply_kraus_to_state_vector(psi, kraus_ops):
    """
    Apply a Kraus operator to a state vector probabilistically (Monte Carlo).

    Args:
        psi (Qobj): Input state vector.
        kraus_ops (list): List of Kraus operators (Qobj).

    Returns:
        Qobj: Output state vector after applying a randomly chosen Kraus operator.
    """
    probs = [qt.expect(K.dag() * K, psi) for K in kraus_ops]
    total_prob = sum(probs)
    if total_prob < 1e-10:  # State is effectively zero
        return psi
    probs = [p / total_prob for p in probs]
    i = np.random.choice(len(kraus_ops), p=probs)
    K = kraus_ops[i]
    psi_out = K * psi
    norm = psi_out.norm()
    return psi_out / norm if norm > 0 else psi_out

def simulate_model1_trajectory(F, eta_c, eta_t, n_bar, dim_photonic=4):
    """
    Simulate Model 1 (Time Bins) with probabilistic source generation and channel loss (Monte Carlo).

    Args:
        F (float): Success probability of the ideal entangled state.
        eta_c (float): Fiber channel transmissivity.
        eta_t (float): Transduction efficiency.
        n_bar (float): Mean photon number of the thermal environment.
        dim_photonic (int): Truncation dimension for photonic modes.

    Returns:
        tuple: (fidelity, ideal_state) for a single trajectory.
    """
    # Define basis states
    g = qt.basis(2, 0)  # |g>
    e = qt.basis(2, 1)  # |e>
    state10 = qt.tensor(qt.basis(dim_photonic, 1), qt.basis(dim_photonic, 0))  # |10>
    state01 = qt.tensor(qt.basis(dim_photonic, 0), qt.basis(dim_photonic, 1))  # |01>
    state00 = qt.tensor(qt.basis(dim_photonic, 0), qt.basis(dim_photonic, 0))  # |00>

    # Ideal state: (|10>|g> + |01>|e>) / sqrt(2)
    psi_ideal = (1 / np.sqrt(2)) * (qt.tensor(state10, g) + qt.tensor(state01, e))
    noise_state = qt.tensor(state00, g)  # Noise state: |00>|g>

    # Probabilistic source generation (Monte Carlo)
    if np.random.rand() < F:
        psi_source = psi_ideal
    else:
        psi_source = noise_state

    # Apply loss to mode 0 probabilistically
    kraus_ops_loss = generate_loss_kraus_ops(eta_c, dim_photonic)
    I_photonic = qt.qeye(dim_photonic)
    I_qubit = qt.qeye(2)
    kraus_ops_full_mode0 = [qt.tensor(K, I_photonic, I_qubit) for K in kraus_ops_loss]
    psi_after_loss_mode0 = apply_kraus_to_state_vector(psi_source, kraus_ops_full_mode0)

    # Apply loss to mode 1 probabilistically
    kraus_ops_full_mode1 = [qt.tensor(I_photonic, K, I_qubit) for K in kraus_ops_loss]
    psi_after_loss = apply_kraus_to_state_vector(psi_after_loss_mode0, kraus_ops_full_mode1)

    # Convert to density matrix for transduction
    rho_after_loss = psi_after_loss * psi_after_loss.dag()

    # Apply transduction (unchanged)
    rho_env_thermal = qt.thermal_dm(dim_photonic, n_bar)
    rho_after_transduction_mode0 = apply_loss_channel(rho_after_loss, eta_t, rho_env_thermal, 0, 2)
    rho_final = apply_loss_channel(rho_after_transduction_mode0, eta_t, rho_env_thermal, 1, 2)

    # Compute fidelity
    fidelity = (psi_ideal.dag() * rho_final * psi_ideal).real
    return fidelity, psi_ideal

def simulate_model2_trajectory(P, eta_c, eta_t, n_bar, dim_photonic=4):
    """
    Simulate Model 2 (Fock states) with probabilistic source generation and channel loss (Monte Carlo).

    Args:
        P (float): Success probability of the ideal entangled state.
        eta_c (float): Fiber channel transmissivity.
        eta_t (float): Transduction efficiency.
        n_bar (float): Mean photon number of the thermal environment.
        dim_photonic (int): Truncation dimension for photonic mode.

    Returns:
        tuple: (fidelity, ideal_state) for a single trajectory.
    """
    # Define basis states
    g = qt.basis(2, 0)  # |g>
    e = qt.basis(2, 1)  # |e>
    photonic0 = qt.basis(dim_photonic, 0)  # |0>
    photonic1 = qt.basis(dim_photonic, 1)  # |1>

    # Ideal state: (|0>|g> + |1>|e>) / sqrt(2)
    psi_ideal = (1 / np.sqrt(2)) * (qt.tensor(photonic0, g) + qt.tensor(photonic1, e))
    noise_state = qt.tensor(photonic0, g)  # Noise state: |0>|g>

    # Probabilistic source generation (Monte Carlo)
    if np.random.rand() < P:
        psi_source = psi_ideal
    else:
        psi_source = noise_state

    # Apply loss to the photonic mode probabilistically
    kraus_ops_loss = generate_loss_kraus_ops(eta_c, dim_photonic)
    I_qubit = qt.qeye(2)
    kraus_ops_full = [qt.tensor(K, I_qubit) for K in kraus_ops_loss]
    psi_after_loss = apply_kraus_to_state_vector(psi_source, kraus_ops_full)

    # Convert to density matrix for transduction
    rho_after_loss = psi_after_loss * psi_after_loss.dag()

    # Apply transduction (unchanged)
    rho_env_thermal = qt.thermal_dm(dim_photonic, n_bar)
    rho_final = apply_loss_channel(rho_after_loss, eta_t, rho_env_thermal, 0, 1)

    # Compute fidelity
    fidelity = (psi_ideal.dag() * rho_final * psi_ideal).real
    return fidelity, psi_ideal

def F1_analytical_model1(eta_t, F, eta_c, nbar):
    """Analytical fidelity F^1 for Model 1 (time bins)."""
    N_t = (1 - eta_t) * (nbar + 0.5)
    Lambda = eta_t / 2 + N_t + 0.5
    term1 = 1 / Lambda**2
    term2 = -(1 + F * eta_c * eta_t) / Lambda**3
    term3 = 3 * F * eta_c * eta_t / Lambda**4
    return 0.5 * (term1 + term2 + term3)

def F1_analytical_model2(eta_t, P, eta_c, nbar):
    """Analytical fidelity F^1 for Model 2 (single rail)."""
    N_t = (1 - eta_t) * (nbar + 0.5)
    a = eta_t / 2 + N_t + 0.5
    term1 = 1 / (2 * a)
    term2 = P * (2 * np.sqrt(eta_c * eta_t) - (1 + eta_c * eta_t)) / (4 * a**2)
    term3 = P * eta_c * eta_t / (2 * a**3)
    return term1 + term2 + term3
