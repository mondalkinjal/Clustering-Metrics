import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial


N = 30  # Lattice size
steps = 100000  # Number of simulation steps
num_processes= 200
temperatures = np.linspace(1.0, 5.0, 200)  # Range of temperatures

def initialize_lattice(N):
    """ Initialize the lattice with random spins. """
    lattice = np.random.choice([-1, 1], size=(N, N))
    return lattice

def metropolis_step(lattice, beta):
    """ Perform one Metropolis step. """
    N = lattice.shape[0]
    for _ in range(N*N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        s = lattice[i, j]
        delta_E = 2 * s * (lattice[(i+1)%N, j] + lattice[i, (j+1)%N] + lattice[(i-1)%N, j] + lattice[i, (j-1)%N])
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            lattice[i, j] = -s
        #return lattice

def ising_simulation(T):
    """ Simulate the 2D Ising model. """
    lattice = initialize_lattice(N)
    beta = 1.0 / T
    magnetizations = []
    energies = []
    lattices_array = []
    temp = str(T)
    part2 = "_macrostates"
    filename = temp+part2
    for step in range(steps):
        metropolis_step(lattice, beta)
        if (step>9999):
            if step % 10 == 0:  # Sample every 10 steps
                 lattices_array.append(lattice.copy())
#                magnetizations.append(calculate_magnetization(lattice))
#                energies.append(calculate_energy(lattice))
    combined_array = np.stack(lattices_array, axis=0)
#    combined_array = combined_array[::1000, :,:]
    return np.save(filename,combined_array)



# Simulation parameters
N = 30  # Lattice size
steps = 100000  # Number of simulation steps
# Range of temperatures

# Arrays to store properties

pool = mp.Pool(num_processes)
pool.starmap(partial(ising_simulation),zip(temperatures))


