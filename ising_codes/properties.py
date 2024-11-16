import numpy as np

class IsingProperties:
    def __init__(self, lattice_size):
        self.lattice_size=lattice_size
    def Energy(self, lattice_whole):
        energy_list = []
        
        for k in range(len(lattice_whole)):
            lattice=lattice_whole[k,...]
            energy=0
            for i in range(self.lattice_size):
                for j in range(self.lattice_size):
                    s = lattice[i, j]
                    nb = lattice[(i+1)%self.lattice_size, j] + lattice[i, (j+1)%self.lattice_size] + lattice[(i-1)%self.lattice_size, j] + lattice[i, (j-1)%self.lattice_size]
                    energy += -s * nb
            energy=energy/(self.lattice_size**2)
            energy_list.append(energy/2)  # Each pair counted twice
        return (energy_list)
    def Magnetization(self, lattice_whole):
        mag_list = []

        for k in range(len(lattice_whole)):
            
            lattice=lattice_whole[k,...]
            mag=np.sum(lattice)
            mag_list.append(abs(mag)/self.lattice_size**2)
        return mag_list
    def specific_heat(self, lattice_whole, temp_array):
        cv_array = np.zeros(len(temp_array))
        for i in range(len(temp_array)):
            temp = temp_array[i]
            temp_lattice = lattice_whole[i,...]
            energies = np.array(self.Energy(temp_lattice))
            energy_mean = np.mean(energies)
            energy_squared_mean = np.mean(energies**2)
            cv = (energy_squared_mean - energy_mean**2) / (temp**2)
            cv_array[i] = cv 
        return cv_array
    def sus(self, lattice_whole, temp_array):
        sus_array = np.zeros(len(temp_array))
        for i in range (len(temp_array)):
            temp = temp_array[i]
            temp_lattice = lattice_whole[i,...]
            mag = np.array(self.Magnetization(temp_lattice))
            mag_mean = np.mean(mag)
            mag_squared = np.mean(mag**2)
            sus = (mag_squared - mag_mean**2) / (temp)
            sus_array[i] = sus
        return sus_array


