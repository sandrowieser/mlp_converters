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

"""
 provides a class to read the structures contained in an ML_ABN file
"""


import numpy as np
from ase.atoms import Atoms

echg = 1.60217663 * 1e-19


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class ml_abn:
    def __init__(
        self, root_direc=".", fname="ML_ABN", filetype="ML_ABN", atom_types=None
    ):
        self.coors = None
        self.cells = None
        self.forces = None
        self.energies = None
        self.elems = None
        self.stress = None
        self.ctifor = None
        if filetype == "ML_ABN":
            self.read_ML_ABN(root_direc + "/" + fname)
        elif filetype == "cfg":
            self.read_MLIP_cfg(root_direc + "/" + fname, atom_types=atom_types)
        return

    def read_ML_ABN(self, fname):

        with open(fname, "r") as fp:
            self.coors = []
            self.cells = []
            self.forces = []
            self.energies = []
            self.elems = []
            self.stress = []
            self.ctifor = []
            readcoor = False
            readcell = False
            readforce = False
            readenergy = False
            readtype = False
            readstress = False
            readctifor = False
            for line in fp:
                elements = line.split()
                if len(elements) == 1:
                    if readctifor:
                        if is_number(elements[0]):
                            self.ctifor.append(float(elements[0]))
                    if readenergy:
                        if is_number(elements[0]):
                            self.energies.append(float(elements[0]))
                    if elements[0] == "CTIFOR":
                        readctifor = True
                        readtype = False
                if len(elements) == 2:
                    if readtype:
                        numel = int(elements[1])
                        for i in range(numel):
                            self.elems[-1].append(elements[0])
                    if (elements[0] == "Stress") & (elements[1] == "(kbar)"):
                        readforce = False
                        readstress = True
                if len(elements) == 3:
                    if readforce:
                        if is_number(elements[0]):
                            self.forces[-1].append(
                                [
                                    float(elements[0]),
                                    float(elements[1]),
                                    float(elements[2]),
                                ]
                            )
                    elif readcoor:
                        if is_number(elements[0]):
                            self.coors[-1].append(
                                [
                                    float(elements[0]),
                                    float(elements[1]),
                                    float(elements[2]),
                                ]
                            )
                    elif readcell:
                        if is_number(elements[0]):
                            self.cells[-1].append(
                                [
                                    float(elements[0]),
                                    float(elements[1]),
                                    float(elements[2]),
                                ]
                            )
                    elif readstress:
                        if is_number(elements[0]):
                            for ind in range(3):
                                self.stress[-1].append(float(elements[ind]))
                if len(elements) > 2:
                    if (elements[0] == "Configuration") & (elements[1] == "num."):
                        self.coors.append([])
                        self.cells.append([])
                        self.forces.append([])
                        self.elems.append([])
                        self.stress.append([])
                        readstress = False
                    if (elements[0] == "Atom") & (elements[1] == "types"):
                        readtype = True
                        readforce = False
                    if (
                        (elements[0] == "Primitive")
                        & (elements[1] == "lattice")
                        & (elements[2] == "vectors")
                    ):
                        readcell = True
                        readctifor = False
                        readtype = False
                    if (elements[0] == "Atomic") & (elements[1] == "positions"):
                        readcoor = True
                        readcell = False
                    if (elements[0] == "Total") & (elements[1] == "energy"):
                        readenergy = True
                        readcoor = False
                    if (elements[0] == "Forces") & (elements[1] == "(eV"):
                        readforce = True
                        readenergy = False

            self._check_empty_arrays()

        return

    def write_ML_ABN(self, fname):
        sout = ""
        starline = "**************************************************\n"
        dashline = "--------------------------------------------------\n"
        eqline = "==================================================\n"
        #        sout = "1.0 Version"
        #        sout += starline
        #        sout += "     The number of configurations\n"
        #        sout += dashline
        #        sout += "        " + str(len(self.coors)) + "\n"
        #        sout += starline
        #        sout += "     The maximum number of atom type\n"
        #        sout += dashline
        #        sout += "       " + str(len(n_unique_elems)) + "\n"
        #        sout += starline
        #        sout += "     The maximum number of atoms per system\n"
        #        sout += dashline
        #        sout += "            " + str(np.max(list(map(len,self.coors[:,:,0])))) + "\n"
        #        sout += starline
        #        sout += "     The maximum number of atoms per atom type\n"
        #        sout += dashline
        #        sout +=
        #        sout += starline
        #        sout += "     Reference atomic energy (eV)\n"
        #        sout += dashline
        #        sout +=
        #        sout +=
        #        sout +=
        #        sout +=

        # HACK - for now only write out the body of the content including coordinates and so on
        for sid, coor in enumerate(self.coors):

            elemlist = []
            typelist = {}
            # not using np.unique here because how it is sorted is important
            for eid, elem in enumerate(self.elems[sid]):
                if elem not in elemlist:
                    elemlist.append(elem)
                    typelist[elem] = [eid]
                else:
                    typelist[elem].append(eid)
            n_unique_elems = len(elemlist)
            sout += starline
            sout += "     Configuration num.      " + str(sid + 1) + "\n"
            sout += eqline
            sout += "     System name\n"
            sout += dashline
            sout += "     # File generated with swfileIO\n"
            sout += eqline
            sout += "     The number of atom types\n"
            sout += dashline
            sout += "       " + str(n_unique_elems) + "\n"
            sout += eqline
            sout += "     The number of atoms\n"
            sout += dashline
            sout += "        " + str(len(coor)) + "\n"
            sout += starline
            sout += "     Atom types and atom numbers\n"
            sout += dashline
            for eid, elem in enumerate(elemlist):
                sout += "%6s %8d\n" % (elem, len(typelist[elem]))
            sout += eqline
            if self.ctifor is not None:
                sout += "     CTIFOR\n"
                sout += dashline
                sout += "%24.16f\n" % self.ctifor[sid]
                sout += eqline
            sout += "     Primitive lattice vectors (ang.)\n"
            sout += dashline
            for i in range(3):
                sout += "%24.16f %24.16f %24.16f\n" % (
                    self.cells[sid][i][0],
                    self.cells[sid][i][1],
                    self.cells[sid][i][2],
                )
            sout += eqline
            sout += "     Atomic positions (ang.)\n"
            sout += dashline
            for cart_coor in coor:
                sout += "%24.16f %24.16f %24.16f\n" % (
                    cart_coor[0],
                    cart_coor[1],
                    cart_coor[2],
                )
            sout += eqline
            sout += "     Total energy (eV)\n"
            sout += dashline
            sout += "%24.16f\n" % self.energies[sid]
            sout += eqline
            sout += "     Forces (eV ang.^-1)\n"
            sout += dashline
            for i in range(len(coor)):
                sout += "%24.16f %24.16f %24.16f\n" % (
                    self.forces[sid][i][0],
                    self.forces[sid][i][1],
                    self.forces[sid][i][2],
                )
            sout += eqline
            sout += "     Stress (kbar)\n"
            sout += dashline
            sout += "     XX YY ZZ\n"
            sout += dashline
            sout += "%24.16f %24.16f %24.16f\n" % (
                self.stress[sid][0],
                self.stress[sid][1],
                self.stress[sid][2],
            )
            sout += dashline
            sout += "     XY YZ ZX\n"
            sout += dashline
            sout += "%24.16f %24.16f %24.16f\n" % (
                self.stress[sid][3],
                self.stress[sid][4],
                self.stress[sid][5],
            )
        with open(fname, "w") as ofp:
            ofp.write(sout)
            ofp.close()
        return

    def _check_empty_arrays(self):
        if len(self.coors) == 0:
            self.coors = None
        else:
            self.coors = np.array(self.coors)
        if len(self.cells) == 0:
            self.cells = None
        else:
            self.cells = np.array(self.cells)
        if len(self.forces) == 0:
            self.forces = None
        else:
            self.forces = np.array(self.forces)
        if len(self.energies) == 0:
            self.energies = None
        else:
            self.energies = np.array(self.energies)
        if len(self.stress) == 0:
            self.stress = None
        else:
            self.stress = np.array(self.stress)

    def write_MLIP_cfg(self, fname, use_nums=False):
        """
        method to write a MLIP style cfg file containing the data
        fname ... name of the output file
        use_nums ... if True, the elems will be interpreted as numbers which will be written in the output directly
        """
        with open(fname, "w") as ofp:
            for sid in range(len(self.coors)):
                ofp.write("BEGIN_CFG\n")
                ofp.write(" Size\n")
                ofp.write(" %5d\n" % (len(self.coors[sid])))
                ofp.write(" Supercell\n")
                for ind in range(3):
                    ofp.write(
                        " %12.8f %12.8f %12.8f\n"
                        % (
                            self.cells[sid][ind, 0],
                            self.cells[sid][ind, 1],
                            self.cells[sid][ind, 2],
                        )
                    )
                if self.forces is not None:
                    ofp.write(
                        " AtomData:  %5s %5s %12s %12s %12s %12s %12s %12s\n"
                        % (
                            "id",
                            "type",
                            "cartes_x",
                            "cartes_y",
                            "cartes_z",
                            "fx",
                            "fy",
                            "fz",
                        )
                    )
                else:
                    ofp.write(
                        " AtomData:  %5s %5s %12s %12s %12s\n"
                        % ("id", "type", "cartes_x", "cartes_y", "cartes_z")
                    )
                # species identified as numbers
                uel, uinds = np.unique(self.elems[sid], return_index=True)
                logical = np.argsort(
                    uinds
                )  # to sort in the same way as the in the structure WARNING: this will not work if the atom types in different POSCARs are ordered differently
                uel = uel[logical]
                if sid == 1:
                    print(uel)
                for aid, coor in enumerate(self.coors[sid]):
                    if use_nums:
                        wel = int(self.elems[sid][aid])
                    else:
                        wel = np.where(uel == self.elems[sid][aid])[0][0]
                    if self.forces is not None:
                        force = self.forces[sid][aid]
                        ofp.write(
                            "            %5d %5d %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n"
                            % (
                                aid + 1,
                                wel,
                                coor[0],
                                coor[1],
                                coor[2],
                                force[0],
                                force[1],
                                force[2],
                            )
                        )
                    else:
                        ofp.write(
                            "            %5d %5d %12.8f %12.8f %12.8f\n"
                            % (aid + 1, wel, coor[0], coor[1], coor[2])
                        )
                if self.energies is not None:
                    ofp.write(" Energy\n")
                    ofp.write(" %14.8f\n" % (self.energies[sid]))
                if self.stress is not None:
                    ofp.write(
                        " PlusStress: %12s %12s %12s %12s %12s %12s\n"
                        % ("xx", "yy", "zz", "yz", "xz", "xy")
                    )
                    cell = self.cells[sid]
                    volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
                    # unit conversion - in VASP stress is given in kbar and in cfg it is supposed to be the virial stress in eV
                    stress = (
                        self.stress[sid] * volume / echg * 1e-22
                    )  # careful, stresses are ordered differently in cfg
                    ofp.write(
                        "             %12.8f %12.8f %12.8f %12.8f %12.8f %12.8f\n"
                        % (
                            stress[0],
                            stress[1],
                            stress[2],
                            stress[4],
                            stress[5],
                            stress[3],
                        )
                    )
                ofp.write(" Feature   EFS_by       VASP\n")
                ofp.write("END_CFG\n\n")

        return

    def read_MLIP_cfg(self, fname, atom_types=None):
        """
        method to read an MLIP config file
         fname ... name of the file
         atom_types ... list with names of atom types which are given as numbers in the cfg file
        """
        self.coors = []
        self.cells = []
        self.forces = []
        self.energies = []
        self.elems = []
        self.stress = []
        readcell = False
        readcoor = False
        readenergy = False
        readstress = False
        volume = 0
        with open(fname, "r") as fp:
            for line in fp:
                elements = line.split()
                if len(elements) > 0:
                    if elements[0] == "BEGIN_CFG":
                        self.coors.append([])
                        self.cells.append([])
                        self.forces.append([])
                        self.elems.append([])
                    elif elements[0] == "Supercell":
                        readcell = True
                    elif elements[0] == "Energy":
                        readcoor = False
                        readenergy = True
                    elif elements[0] == "Feature":
                        readcoor = False
                    elif readenergy:
                        if is_number(elements[0]):
                            self.energies.append(float(elements[0]))
                            readenergy = False
                if len(elements) > 2:
                    if readcell:
                        if is_number(elements[0]):
                            self.cells[-1].append(
                                [
                                    float(elements[0]),
                                    float(elements[1]),
                                    float(elements[2]),
                                ]
                            )
                if (len(elements) == 6) or (len(elements) == 7):
                    if elements[0] == "PlusStress:":
                        readstress = True
                    elif readstress:
                        # careful with the ordering, in .cfg it is ordered xx,yy,zz,yz,xz,xy and in VASP xx,yy,zz,xy,yz,xz
                        stress = np.array(
                            [
                                float(elements[0]),
                                float(elements[1]),
                                float(elements[2]),
                                float(elements[5]),
                                float(elements[3]),
                                float(elements[4]),
                            ]
                        )
                        stress = stress / volume * echg / 1e-22
                        self.stress.append(stress)
                        readstress = False
                if len(elements) > 4:
                    if elements[0] == "AtomData:":
                        readcell = False
                        cell = self.cells[-1]
                        volume = np.dot(cell[0], np.cross(cell[1], cell[2]))
                        readcoor = True
                    elif readcoor:
                        self.coors[-1].append(
                            [float(elements[2]), float(elements[3]), float(elements[4])]
                        )
                        if len(elements) > 7:
                            self.forces[-1].append(
                                [
                                    float(elements[5]),
                                    float(elements[6]),
                                    float(elements[7]),
                                ]
                            )
                        elif len(self.coors[-1]) == 1:
                            self.forces.pop()
                        if atom_types is None:
                            self.elems[-1].append(elements[1])
                        else:
                            try:
                                self.elems[-1].append(atom_types[int(elements[1])])
                            except:
                                raise ValueError("Not enough element information provided to read .cfg file")
            self._check_empty_arrays()

        return


    def get_stress_matrix(self, cid):
        smat = np.zeros((3, 3))
        # this object uses the VASP ML_ABN formatting for the stress vector
        smat[0, 0] = self.stress[cid][0]
        smat[1, 1] = self.stress[cid][1]
        smat[2, 2] = self.stress[cid][2]
        smat[0, 1] = self.stress[cid][3]
        smat[1, 0] = self.stress[cid][3]
        smat[1, 2] = self.stress[cid][4]
        smat[2, 1] = self.stress[cid][4]
        smat[0, 2] = self.stress[cid][5]
        smat[2, 0] = self.stress[cid][5]
        return smat
    
    def get_geos(
        self,
        forcename="forces",
        stressname="stress",
        stress_format="vector",
        stress_unit="kbar",
    ):
        # convert the structures into geometry objects for other processing tools
        # use the arguments forcename and stressname for the additional entries in the adata dictionary of the geometry objects
        # stress format ... matrix or vector
        # stress unit ... specify which units the stress uses - default is kbar as it is the unit in VASP
        from mlp_converters.geometry import geo

        geos = []
        if stress_unit == "bar":
            stress_factor = 1e3
        elif stress_unit == "kbar":
            stress_factor = 1
        else:
            raise NotImplementedError("unknown stress unit " + stress_unit)
        for cid, cells in enumerate(self.cells):
            geom = geo(
                "POSCAR",
                filetype="direct",
                elems=self.elems[cid],
                coors=self.coors[cid],
                cell=self.cells[cid],
            )
            if self.forces is not None:
                geom.add_additional_data(forcename, self.forces[cid])
            if self.stress is not None:
                if stress_format == "vector":
                    geom.add_additional_data(
                        stressname, self.stress[cid] * stress_factor
                    )
                elif stress_format == "matrix":
                    smat = self.get_stress_matrix(cid)
                    smat *= stress_factor
                    geom.add_additional_data(stressname, smat)
            geos.append(geom)
        return geos
    
    def get_alist(self,
        energyname="energy",
        forcename="forces",
        stressname="stress",
        stress_unit="kbar"):
        """
        get a list of atoms objects, which should be way easier to handle.
        """

        from ase.calculators.singlepoint import SinglePointCalculator

        if stress_unit == "bar":
            stress_factor = 1e3
        elif stress_unit == "kbar":
            stress_factor = 1
        else:
            raise NotImplementedError("unknown stress unit " + stress_unit)
        
        alist = []
        for cid, cell in enumerate(self.cells):
            atoms = Atoms(symbols=self.elems[cid], positions=self.coors[cid], cell=cell,)
            rdict = {}
            rdict[energyname] = self.energies[cid]
            rdict[forcename] = self.forces[cid]
            smat = self.get_stress_matrix(cid) * stress_factor
            rdict[stressname] = smat
            # atoms.set_array(forcename, self.forces[cid])
            # atoms.set_array(stressname, smat)
            calc = SinglePointCalculator(atoms, **rdict)
            atoms.calc = calc
            
            alist.append(atoms)

        return alist


    def from_geos(
        self, geos, forcename="forces", stressname="stress", stress_format="vector"
    ):
        # build the ml_abn internal structure from a list of geometry objects
        self.coors = []
        self.cells = []
        self.elems = []
        for gid, geo in enumerate(geos):
            self.coors.append(geo.coors)
            if geo.atom_types is None:
                self.elems.append(geo.elems)
            else:
                self.elems.append(geo.atom_types)
            if forcename in geo.adata.keys():
                if self.forces is None:
                    self.forces = []
                self.forces.append(geo.adata[forcename])
            if stressname in geo.adata.keys():
                if self.stress is None:
                    self.stress = []
                if stress_format == "vector":
                    self.stress.append(geo.adata[stressname])
                elif (
                    stress_format == "matrix"
                ):  # will not be symmetrized, assuming the stress matrix is symmetric already
                    self.stress.append(
                        [
                            geo.adata[stressname][0, 0],
                            geo.adata[stressname][1, 1],
                            geo.adata[stressname][2, 2],
                            geo.adata[stressname][0, 1],
                            geo.adata[stressname][1, 2],
                            geo.adata[stressname][0, 2],
                        ]
                    )
            self.cells.append(geo.cell)
        self.coors = np.array(self.coors)
        self.cells = np.array(self.cells)
        self.elems = np.array(self.elems)
        if self.forces is not None:
            self.forces = np.array(self.forces)
        if self.stress is not None:
            self.stress = np.array(self.stress)
        return
    
    def from_atoms(self, alist : list[Atoms]):
        self.coors = []
        self.cells = []
        self.elems = []
        self.energies = []
        self.forces = []
        self.stress = []
        for aid, atoms in enumerate(alist):
            self.coors.append(atoms.positions)
            self.cells.append(atoms.cell.array)
            self.elems.append(atoms.symbols)
            if self.forces is not None:
                try:
                    self.forces.append(atoms.get_forces())
                except:
                    print("WARNING: forces not found")
                    self.forces = None
            if self.energies is not None:
                try:
                    self.energies.append(atoms.get_potential_energy())
                except:
                    print("WARNING: energy not found")
                    self.energies = None
            if self.stress is not None:
                try:
                    stress = atoms.get_stress()
                    self.stress.append(stress)
                except:
                    print("WARNING: stress not found")
                    self.stress = None
        return
