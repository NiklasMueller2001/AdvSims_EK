import numpy as np
import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class XYZWriter:
    """Converts numpy arrays to xyz files that can be read by VMD.
    
    Parameters
    ----------
    file_path: str
        Relative path to the generated xyz file.

    data: np.ndarray
        Time series of particle positions.
        Must have shape (t, n, 3), where t specifies the number of time steps and n is the 
        number of particles.

    atom_type: str
        Symbol of element to write at the beginning of every line in generated xyz file.
    """

    def __init__(self, file_path: str, data: np.ndarray, atom_type: str) -> None:
        self.file_path = file_path
        self.data = data
        self.num_of_time_steps = data.shape[0]
        self.num_of_particles = data.shape[1]
        self.atoms_type = atom_type

    def write(self, v: bool = False) -> None:
        """Write all data to xyz file."""
        logging.info(f"Starting to write to file {self.file_path}.")
        with open(self.file_path, "w") as file:
            for i in tqdm.tqdm(range(self.num_of_time_steps), desc="Writing time steps"):
                msg_i = self.generate_msg(step=i)
                if v is True:
                    logging.info(msg_i)
                file.write(msg_i)
            
    def generate_msg(self, step: int) -> str:
        """Generates a string for one time step."""
        msg = f"""{self.num_of_particles}\nTime Step: {step}\n""" + "C " +  "C ".join(map(str, self.data[step, ...]))
        msg = msg.replace("[", "")
        msg = msg.replace("] ", "\n")
        msg = msg.replace("]", "\n")
        return msg


if __name__ == "__main__":
    polymer_pos = np.load("../data/polymer_equilibration.npy")
    ions_pos = np.load("../data/ion_equilibration.npy")
    polymer_writer = XYZWriter("../data/polymer_equilibration.xyz", polymer_pos, "C")
    ions_writer = XYZWriter("../data/ion_equilibration.xyz", ions_pos, "Cl")
    polymer_writer.write()
    ions_writer.write()
    