# Copyright 2025 Sandro Wieser
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import numpy as np
from scipy.spatial.transform import Rotation as R

import mlp_converters.fileIO.ml_abn as ml_abn
import mlp_converters.util as util
from mlp_converters.aprops import elems


class geo:
    def __init__(
        self,
        filen,
        filetype="detect",
        elems=None,
        coors=None,
        cell=None,
        masses=None,
        atom_types=None,
        qe_numq=1,
    ):
        """
        masses can be added here - for the most part they are not needed - but they
        can be useful if there are atoms of the same element with different atom types in
        elems
        a similar purpose exists for the explicit definition of atom types
        If the filetype is POSCARS_all then filen has to be a file handle instead
        qe_numq serves to read quantum espresso dynamical matrix files

        """
        self.masses = masses
        self.atom_types = atom_types
        self.adata = {}
        if filetype == "direct":
            self.cell = np.asarray(cell)
            self.elems = np.asarray(elems)
            self.coors = np.asarray(coors)
        else:
            self.read_file(filen, filetype=filetype, qe_numq=qe_numq)
        return

    def read_file(self, filen, filetype, qe_numq=1):

        self.cell = None
        self.molsys_obj = None

        if filetype == "detect":
            if filen != "POSCAR":
                filetype = filen.split(".")[-1]
            else:
                filetype = "POSCAR"

        self.filetype = filetype

        if filetype == "xyz":
            data = util.readXYZ(filen)
            self.cell = np.asarray(data["cell"])
            self.elems = np.asarray(data["elems"])
            self.coors = np.asarray(data["coors"])
        elif filetype == "POSCAR":
            data = util.readPOSCAR(filen)
            self.cell = np.asarray(data["cell"])
            self.elems = np.asarray(data["elems"])
            self.coors = np.asarray(data["coors"])
        elif filetype == "POSCARS_all":  # read from filehandle
            data = util.readPOSCAR("POSCAR", from_fp=filen)
            self.cell = np.asarray(data["cell"])
            self.elems = np.asarray(data["elems"])
            self.coors = np.asarray(data["coors"])
        elif (filetype == "data") | (filetype == "lmp"):
            data = util.read_lammps_data(filen)
            self.lmp_species = np.asarray(data["species"])
            self.type_masses = np.asarray(data["masses"])
            self.massids = np.asarray(data["massids"])
            try:
                self.get_elements_from_comments(self.massids, data["macomments"])
            except:
                self.elems = list(map(str, data["species"]))
            self.coors = np.asarray(data["coors"])
            if "xy" in data.keys():
                self.cell = [
                    [data["xx"], 0.0, 0.0],
                    [data["xy"], data["yy"], 0.0],
                    [data["xz"], data["yz"], data["zz"]],
                ]
            else:
                self.cell = [
                    [data["xx"], 0.0, 0.0],
                    [0.0, data["yy"], 0.0],
                    [0.0, 0.0, data["zz"]],
                ]
            self.cell = np.asarray(self.cell)
        elif filetype == "mfpx":
            import molsys

            m = molsys.mol.from_file(filen)
            self.molsys_obj = m
            self.cell = np.asarray(m.get_cell())
            self.coors = np.asarray(m.xyz)
            # capitalize elements
            elnames = []
            for e in m.elems:
                elnames.append(e.capitalize())
            self.elems = elnames
        elif filetype == "dyn":  # Quantum espresso format of dynamical matrix files
            import cellconstructor as CC
            import cellconstructor.Phonons

            self.dyn = CC.Phonons.Phonons(filen, qe_numq)
            self.cell = self.dyn.structure.unit_cell
            self.coors = self.dyn.structure.coords
            self.elems = self.dyn.structure.atoms

        else:
            print("ERROR: file type not known")

        if self.cell is not None:
            self.a = np.linalg.norm(self.cell[0])
            self.b = np.linalg.norm(self.cell[1])
            self.c = np.linalg.norm(self.cell[2])

        return

    def write_file(self, filen, filetype=None, bond_num=None, uelemsort=False):
        """
        writes the structure as a file
        filen ... name of the output file
        filetype ... filetype to write as a string - it will be attempted to guess the filetype if none is given
        bond_num ... specifically for lammps data files - if bond_num is provided the file will include this as the maximum number of bond types
        uelemsort ... in the case of a lammps file when trying to assign atom types they will be sorted according to the alphabetical ordering of the element name instead of according to which atom types comes first in the atom list.
        """
        if filetype == None:
            if filen != "POSCAR":
                filetype = filen.split(".")[1]
            else:
                filetype = "POSCAR"

        if filetype == "xyz":
            util.write_xyz(filen, self.cell, self.elems, self.coors)
        elif filetype == "POSCAR":
            util.write_POSCAR(filen, self.cell, self.elems, self.coors)
        elif filetype == "mfpx":
            lower_els = []
            for e in self.elems:
                lower_els.append(e.lower())
            if self.molsys_obj is not None:
                self.molsys_obj.set_cell(self.cell)
                self.molsys_obj.xyz = self.coors
                self.molsys_obj.elems = lower_els

            else:
                import molsys

                self.molsys_obj = molsys.mol.from_array(self.coors)
                self.molsys_obj.set_cell(self.cell)
                self.molsys_obj.elems = lower_els
            self.molsys_obj.write(filen)
            if "ff" in self.molsys_obj.loaded_addons:
                self.molsys_obj.ff.write(filen.split(".mfpx")[0])
        elif (filetype == "lmp_atomic") | (filetype == "lmp"):
            util.write_lammps_atomic(
                filen,
                self.cell,
                self.elems,
                self.coors,
                masses=self.masses,
                atom_types=self.atom_types,
                bond_num=bond_num,
                uelemsort=uelemsort,
            )
        elif filetype == "lmp_charge":
            util.write_lammps_atomic(
                filen,
                self.cell,
                self.elems,
                self.coors,
                masses=self.masses,
                atom_types=self.atom_types,
                charges=self.charges,
                bond_num=bond_num,
                uelemsort=uelemsort,
            )
        elif filetype == "cfg":  # MLIP style cfg
            mobj = ml_abn.ml_abn(filetype=None)
            mobj.from_geos([self])
            mobj.write_MLIP_cfg(filen)
        else:
            print("ERROR: file type not known")
        return

    # extracts elements from molsys style comments
    def get_elements_from_comments(self, massids, commdata):

        self.elems = np.asarray(self.lmp_species, dtype=np.str)
        for i, comm in enumerate(commdata):

            try:
                # molsys style comment
                elstr = comm.split(")")[0].split("(")[1].split("_")[0]
                self.elems[self.lmp_species == massids[i]] = elstr[
                    : (len(elstr) - 1)
                ].capitalize()
            except:
                # just element as comment
                elstr = comm.split()[0]
                self.elems[self.lmp_species == massids[i]] = elstr
        return

    # rotates atoms around a certain axis by a specified angle in degrees
    def rotate_atoms(self, indices, axis, angle, offset=[0.0, 0.0, 0.0]):

        offset = np.array(offset)
        rotation_radians = np.radians(angle)

        axis = np.array(axis)

        rotation_vector = rotation_radians * axis

        rotation = R.from_rotvec(rotation_vector)

        for atom in indices:
            vec = self.coors[atom]
            self.coors[atom] = rotation.apply(vec - offset) + offset

    def assign_masses(self):
        """
        method to add masses based on the element list
        """
        mass_data = elems.mass
        self.masses = []
        for elem in self.elems:
            if elem.lower() in mass_data.keys():
                self.masses.append(mass_data[elem.lower()])
            else:
                print(
                    "WARNING: mass for element %s not found, using 1 amu instead" % elem
                )
                self.masses.append(1)
        self.masses = np.array(self.masses)

    # displace all atoms along a set of vectors by a specific amplitude
    # allows modulation along other q vectors (parameters qv and phase)
    # if massweight is True, then the displacement is divided by the square root of the mass
    # phase shift in radians
    def displace_atoms_along_vectors(
        self,
        vectors,
        amplitude,
        qv=[0, 0, 0],
        phase=0,
        masswgt=False,
        cell_index=[0, 0, 0],
    ):
        if masswgt:
            if self.masses is None:
                self.assign_masses()
            massfact = np.array([np.sqrt(self.masses)] * 3).T
        else:
            massfact = 1
        xyz = (
            self.coors
            + np.real(
                vectors
                * np.exp(1j * np.array(qv) * np.array(cell_index))
                * np.exp(1j * phase)
            )
            * amplitude
            / massfact
        )
        self.coors = xyz

    # adds additional per atom vectors
    def add_additional_data(self, name, data):
        self.adata[name] = np.asarray(data)

    # rotates atoms from a cell matrix to a different one
    def rotate_cell(self, newcell):
        self.coors = util.rotate_vec(self.coors, self.cell, newcell)
        for key in self.adata:
            if (
                key == "stress"
            ):  # stress and all tensor properties have to be rotated different
                self.adata[key] = util.rotate_tens(self.adata[key], self.cell, newcell)
            else:  # this is mostly here for forces - could cause some issues for other types of non-vector properties
                self.adata[key] = util.rotate_vec(self.adata[key], self.cell, newcell)
        self.cell = newcell

    # converts fractional coordinates to Cartesian coordinates
    def frac_to_cart(self):
        self.coors = np.dot(self.coors, self.cell)

    def cart_to_frac(self):
        cell_inv = np.linalg.inv(self.cell)
        self.coors = np.dot(self.coors, cell_inv)

    # computes and returns the reciprocal lattice vectors
    def get_reciprocal_lattice_vectors(self):
        volume = self.get_volume()
        prefact = 2 * np.pi / volume
        rcell = np.zeros((3, 3))
        rcell[0] = prefact * np.cross(self.cell[1], self.cell[2])
        rcell[1] = prefact * np.cross(self.cell[2], self.cell[0])
        rcell[2] = prefact * np.cross(self.cell[0], self.cell[1])
        return rcell

    # conversion of cell vectors and internal coordinates to lammps format
    def convert_cell_to_lammps(self):
        newcell = np.zeros(np.shape(self.cell))
        # math is directly taken from the lammps documentation
        normA = np.linalg.norm(self.cell[0])
        normB = np.linalg.norm(self.cell[1])
        normC = np.linalg.norm(self.cell[2])
        newcell[0, 0] = normA
        newcell[1, 0] = np.dot(self.cell[1], self.cell[0] / normA)
        newcell[1, 1] = np.sqrt(normB**2 - newcell[1, 0] ** 2)
        newcell[2, 0] = np.dot(self.cell[2], self.cell[0] / normA)
        newcell[2, 1] = (
            np.dot(self.cell[1], self.cell[2]) - newcell[1, 0] * newcell[2, 0]
        ) / newcell[1, 1]
        # to prevent negative square root
        sqarg = normC**2 - newcell[2, 0] ** 2 - newcell[2, 1] ** 2
        if abs(sqarg) > 1e-10:
            newcell[2, 2] = np.sqrt(sqarg)
        else:
            newcell[2, 2] = 0.0
        self.rotate_cell(newcell)

    # computes the distance between two coordinates of neighboring cells
    def _dist_cellpm(self, coor1, coor2, direc):
        coorm = coor2 - self.cell[direc]
        distm = np.linalg.norm(coor1 - coorm)
        coorp = coor2 + self.cell[direc]
        distp = np.linalg.norm(coor1 - coorp)
        return distp, distm, coorp, coorm

    # get the shortest distance between two atoms considering periodic boundary conditions
    # a botchy solution - I did not want to assume anything about the layout of coordinates
    # there are certainly better solutions that are faster, this is essentially trial and error

    def get_distance(self, aid1, aid2):
        dists = [np.linalg.norm(self.coors[aid1] - self.coors[aid2])]
        if (dists[0] > self.a / 2) | (dists[0] > self.b / 2) | (dists[0] > self.c / 2):
            for direc in range(3):
                dist_1, dist_2, coorp, coorm = self._dist_cellpm(
                    self.coors[aid1], self.coors[aid2], direc
                )
                dists.append(dist_1)
                dists.append(dist_2)
                for direc2 in range(2 - direc):
                    dist1, dist2, coorp2, coorm2 = self._dist_cellpm(
                        self.coors[aid1], coorm, direc2
                    )
                    dists.append(dist1)
                    dists.append(dist2)
                    for direc3 in range(1 - direc2):
                        dist1, dist2, coorp3, coorm3 = self._dist_cellpm(
                            self.coors[aid1], coorm2, direc3
                        )
                        dists.append(dist1)
                        dists.append(dist2)
                        dist1, dist2, coorp3, coorm3 = self._dist_cellpm(
                            self.coors[aid1], coorp2, direc3
                        )
                        dists.append(dist1)
                        dists.append(dist2)
                    dist1, dist2, coorp2, coorm2 = self._dist_cellpm(
                        self.coors[aid1], coorp, direc2
                    )
                    dists.append(dist1)
                    dists.append(dist2)
                    for direc3 in range(1 - direc2):
                        dist1, dist2, coorp3, coorm3 = self._dist_cellpm(
                            self.coors[aid1], coorm2, direc3
                        )
                        dists.append(dist1)
                        dists.append(dist2)
                        dist1, dist2, coorp3, coorm3 = self._dist_cellpm(
                            self.coors[aid1], coorp2, direc3
                        )
                        dists.append(dist1)
                        dists.append(dist2)
            # print (min(dists),dists)
        return min(dists)

    def get_volume(self):
        return np.dot(np.cross(self.cell[0], self.cell[1]), self.cell[2])

    def set_cell(self, cell):
        self.cell = cell
        if self.molsys_obj is not None:
            self.molsys_obj.set_cell(self.cell)

    def set_charges(self, charges):
        self.charges = charges

    def get_cell_lengths_and_angles(self):
        lengths = np.zeros(3)
        lengths[0] = np.linalg.norm(self.cell[0])
        lengths[1] = np.linalg.norm(self.cell[1])
        lengths[2] = np.linalg.norm(self.cell[2])
        angles = np.zeros(3)
        angles[0] = np.arccos(
            np.dot(self.cell[1], self.cell[2]) / lengths[1] / lengths[2]
        )
        angles[1] = np.arccos(
            np.dot(self.cell[0], self.cell[2]) / lengths[0] / lengths[2]
        )
        angles[2] = np.arccos(
            np.dot(self.cell[0], self.cell[1]) / lengths[0] / lengths[1]
        )

        return lengths, angles

    def get_stress_from_forces(self, forces=None, periodic=True):
        """
        This method computes the stress tensor from the forces when they are either stored
        in adata or given as an argument in forces
        Apparently one has to compute the stress very differently for periodic systems
        And this is very complicated - which is why I currently have no intention of doing it.
        It can also not be done based on forces and positions alone. One would need to compute
        local energies for groups of atoms - meaning this is only straightforward to do while
        the interactions are being calculated
        """
        if forces is None:
            assert (
                "forces" in self.adata.keys()
            ), "ERROR: no forces provided in get_stress_from_forces"
            forces = self.adata["forces"]
        if self.cell is None:
            periodic = False
        if not periodic:
            man_stress = np.zeros((3, 3))
            for cid, coor in enumerate(self.coors):
                man_stress += np.outer(forces[cid], coor)
            return -man_stress
        else:
            assert (
                False
            ), "ERROR: computing stress with the geometry object is not implemented for periodic systems yet"

    # adds an atom to the structure while considering a potential molsys object
    def add_atom(self, vec, elem, atype=None, fragtype="-1", fragnumber=-1):
        vec = np.array(vec)
        self.coors = np.append(self.coors, vec)
        self.coors = np.reshape(self.coors, (int(len(self.coors) / 3), 3))
        self.elems = np.append(self.elems, elem)
        if self.molsys_obj is not None:
            melem = elem.lower()
            if atype is None:
                matype = melem
            else:
                matype = atype
            self.molsys_obj.add_atom(
                melem, matype, vec, fragtype=fragtype, fragnumber=fragnumber
            )

    # creates a molsys object out of the current geometry if it does not exist yet
    def load_molsys(self):
        if self.molsys_obj is None:
            import molsys

            lower_els = []
            for e in self.elems:
                lower_els.append(e.lower())
            self.molsys_obj = molsys.mol.from_array(self.coors)
            self.molsys_obj.set_cell(self.cell)
            self.molsys_obj.elems = lower_els

    # adds multiple atoms
    def add_atoms(self, vecs, elems):
        for aid, vec in enumerate(vecs):
            self.add_atom(vec, elems[aid])

    # adds connectivity between two atom lists, relies on molsys
    def add_conn(self, aid1, aid2):
        assert self.molsys_obj is not None
        self.molsys_obj.add_bonds(aid1, aid2)

    # removes connectivity between two atoms if it exists, relies on molsys
    def remove_conn(self, aid1, aid2):
        assert self.molsys_obj is not None
        self.molsys_obj.delete_bond(aid1, aid2)

    # sets up an FF using molsys and the connectivities
    # initially, it will be set up as a fit
    def setup_ff(self, refsysname="default", cross_terms=["strbnd", "bb13"]):
        assert self.molsys_obj is not None
        if "ff" not in self.molsys_obj.loaded_addons:
            self.molsys_obj.addon("ff")
        else:
            print("WARNING, ff module is already loaded")
        self.molsys_obj.ff.assign_params(
            refsysname + "-FF", refsysname=refsysname, cross_terms=cross_terms
        )
        self.molsys_obj.ff.par.variables.cleanup()
        self.molsys_obj.ff.par.variables()

    # converts the variables in the stored FF to actual parameters, the argument can limit the types
    def apply_parameters(self, types=["a", "b", "d", "o"]):
        assert self.molsys_obj is not None
        assert "ff" in self.molsys_obj.loaded_addons
        self.molsys_obj.ff.remove_pars(identifier=types)

    # method to blanket change a set of parameters in the FF stored inside the molsys object
    # possible ric types are "bnd", "ang", "dih", "oop", "cha", "vdw", "vdwpr", "chapr"
    # specify None in the value list if you wish to keep the value
    # the value list must be equally long as the specific parameter
    # if cross terms are to be considered their string has to specified extra
    def set_all_params(self, rictype, values, cross_term=None):
        assert self.molsys_obj is not None
        assert "ff" in self.molsys_obj.loaded_addons
        pars = self.molsys_obj.ff.par[rictype]
        parinds = self.molsys_obj.ff.parind[rictype]
        for pid, par in enumerate(pars.keys()):
            for vid, v in enumerate(values):
                if v is not None:
                    pars[par][1][vid] = v

    def to_ase(self):
        from ase.atoms import Atoms
        atoms = Atoms(symbols=self.elems,positions=self.coors,cell=self.cell)
        return atoms
