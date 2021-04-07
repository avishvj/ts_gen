
import numpy as np
import openbabel
import pybel

class Gaussian(object):
    """
    Class for reading data from Gaussian log files.

    The attribute `input_file` represents the path where the input file for the
    quantum job is located, the attribute `logfile` represents the path where
    the log file containing the results is located, the attribute `chkfile`
    represents the path where a checkpoint file for reading from a previous job
    is located, and the attribute `output` contains the output of the
    calculation in a list.
    """

    def __init__(self, input_file=None, logfile=None, chkfile=None):
        self.converge = True
        self.input_file = input_file
        self.logfile = logfile
        self.chkfile = chkfile
        if logfile is not None:
            self.read()
        else:
            self.output = None

    def read(self):
        """
        Reads the contents of the log file.
        """
        with open(self.logfile, 'r') as f:
            self.output = f.read().splitlines()

            if self.output[-1].split()[0] != 'Normal':
                self.converge = False

    @staticmethod
    def _formatArray(a, b=3):
        """
        Converts raw geometry or gradient array of strings, `a`, to a formatted
        :class:`numpy.ndarray` of size N x 'b'. Only the rightmost 'b' values of
        each row in `a` are retained.
        """
        vec = np.array([])
        for row in a:
            vec = np.append(vec, [float(e) for e in row.split()[-b:]])
        return vec.reshape(len(a), b)

    def getNumAtoms(self):
        """
        Extract and return number of atoms from Gaussian job.
        """
        read = False
        natoms = 0
        i = 0
        for line in self.output:
            if read:
                i += 1
                try:
                    natoms = int(line.split()[0])
                except ValueError:
                    if i > 5:
                        return natoms
                    continue
            elif 'Input orientation' in line or 'Z-Matrix orientation' in line:
                read = True
        raise QuantumError('Number of atoms could not be found in Gaussian output')

    def getEnergy(self):
        """
        Extract and return energy (in Hartree) from Gaussian job.
        """
        # Read last occurrence of energy
        for line in reversed(self.output):
            if 'SCF Done' in line:
                energy = float(line.split()[4])
                return energy
        raise QuantumError('Energy could not be found in Gaussian output')

    def getGradient(self):
        """
        Extract and return gradient (forces) from Gaussian job. Results are
        returned as an N x 3 array in units of Hartree/Angstrom.
        """
        natoms = self.getNumAtoms()

        # Read last occurrence of forces
        for line_num, line in enumerate(reversed(self.output)):
            if 'Forces (Hartrees/Bohr)' in line:
                force_mat_str = self.output[-(line_num - 2):-(line_num - 2 - natoms)]
                break
        else:
            raise QuantumError('Forces could not be found in Gaussian output')

        gradient = - self._formatArray(force_mat_str) / constants.bohr_to_ang  # Make negative to get gradient
        return gradient

    def getGeometry(self, atomType=True):
        """
        Extract and return final geometry from Gaussian job. Results are
        returned as an N x 3 array in units of Angstrom.
        If atomType is true, returns N x 4 array with first column as
        atomic number
        """
        natoms = self.getNumAtoms()

        # Read last occurrence of geometry
        for line_num, line in enumerate(reversed(self.output)):
            if 'Input orientation' in line or 'Z-Matrix orientation' in line:
                coord_mat_str = self.output[-(line_num - 4):-(line_num - 4 - natoms)]
                break
        else:
            raise QuantumError('Geometry could not be found in Gaussian output')

        if atomType:
            geometry = self._formatArray(coord_mat_str, b=5)
            geometry = np.delete(geometry, 1, 1)
        else:
            geometry = self._formatArray(coord_mat_str, b=3)

        return geometry

    def getIRCpath(self):
        """
        Extract and return IRC path from Gaussian job. Results are returned as
        a list of tuples of N x 3 coordinate arrays in units of Angstrom and
        corresponding energies in Hartrees. Path does not include TS geometry.
        """
        for line in self.output:
            if 'IRC-IRC' in line:
                break
        else:
            raise QuantumError('Gaussian output does not contain IRC calculation')

        natoms = self.getNumAtoms()

        # Read IRC path (does not include corrector steps of last point if there was an error termination)
        path = []
        for line_num, line in enumerate(self.output):
            if 'Input orientation' in line or 'Z-Matrix orientation' in line:
                coord_mat = self._formatArray(self.output[line_num + 5:line_num + 5 + natoms])
            elif 'SCF Done' in line:
                energy = float(line.split()[4])
            elif 'Forces (Hartrees/Bohr)' in line:
                force_mat_str = self.output[line_num + 3:line_num + 3 + natoms]
                gradient = - self._formatArray(force_mat_str) / constants.bohr_to_ang
            elif 'NET REACTION COORDINATE UP TO THIS POINT' in line:
                path.append((coord_mat, energy, gradient))

        if not path:
            raise QuantumError('IRC path is too short')
        return path

    def getNumImaginaryFrequencies(self):
        """
        Extract and return the number of imaginary frequencies from a Gaussian
        job.
        """
        for line in self.output:
            if 'imaginary frequencies' in line:
                nimag = int(line.split()[1])
                return nimag
        raise QuantumError('Frequencies could not be found in Gaussian output')

    def getCharge(self):
        """
        Extract and return charge from Gaussian log file
        """
        for line in self.output:
            if 'Charge' in line:
                charge = int(line.split()[2])
                return charge
        raise QuantumError('Charge could not be found in Gaussian output')

    def getMultiplicity(self):
        """
        Extract and return multiplicity from Gaussian log file
        """
        for line in self.output:
            if 'Multiplicity' in line:
                charge = int(line.split()[5])
                return charge
        raise QuantumError('Multiplicity could not be found in Gaussian output')

    def getFrequencies(self):
        """
        Extract and return list of frequencies
        """
        freqs = []
        for line in self.output:
            if line.split()[0] == "Frequencies":
                freqs.extend(line.split()[2:])
        return [float(freq) for freq in freqs]

    def getPybelMol(self):
        """
        Converts a Gaussian output log file topybel mol
        """
        return pybel.readfile('g09', self.logfile).next()

    def getSmiles(self, all_single_bonds=False, delete_stereochem=False):
        """
        Converts a Gaussian output log file to canonical SMILES with openbabel
        """
        mol = self.getPybelMol()

        if all_single_bonds:
            for bond in pybel.ob.OBMolBondIter(mol.OBMol):
                bond.SetBondOrder(1)
        mol.write()
        if delete_stereochem:
            mol.OBMol.DeleteData(openbabel.StereoData)
        return mol.write('smi').split()[0]

    def getDistanceMatrix(self):
        """
        Use extracted geometry from Gaussian log file to generate NxN distance matrix
        """
        X = self.getGeometry(atomType=False)
        Dsq = np.square(np.expand_dims(X, 1)-np.expand_dims(X, 0))
        return np.sqrt(np.sum(Dsq, 2))


class constants():

    def __init__(self):
        pass

    def bohr_to_ang(self):
        return 0.529177211

    def hartree_to_kcal_per_mol(self):
        return 627.5095

    def kcal_to_J(self):
        return 4184


class QuantumError(Exception):
    """dummy exception class for gaussian class"""
    pass
