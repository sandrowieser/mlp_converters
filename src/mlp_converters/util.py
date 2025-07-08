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
import pickle
import os


# reads the force constants from a phonopy FORCE_CONSTANTS file
# shape: full or tensor
def read_FCs(fname, shape="full"):
    f = open(fname, "r")
    firstline = f.readline().split()
    numats = int(firstline[-1])
    primats = int(firstline[0])
    if shape == "full":
        hessian = np.zeros([3 * numats, 3 * numats])
    else:
        hessian = np.zeros([primats, numats, 3, 3])
        scsize = int(numats / primats)

    i = 0
    j = 0
    ii = 0
    jj = 0
    for line in f:
        elements = line.split()
        # in case of one element - there is no space between the matrix indices
        if (len(elements) == 2) | (len(elements) == 1):
            if len(elements) == 2:
                i = int(elements[0]) - 1
                j = int(elements[1]) - 1
                curri = i
                # in this case the space is missing
            else:
                i = curri
                fstr = str(curri + 1)
                # convoluted, but works
                j = int(elements[0][(elements[0].find(fstr) + len(fstr)) :]) - 1
                # print j,curri, elements,elements[0].find(str(curri+1))
            jj = 0
            # print(str(3*j+jj) +" " +str(3*i+ii))
        if len(elements) == 3:
            for e in elements:
                if shape == "full":
                    hessian[3 * i + jj][3 * j + ii] = float(e)
                    ii += 1
                else:
                    hessian[int(i / scsize)][j][jj][ii] = float(e)
                    ii += 1
            jj += 1
            ii = 0
    return hessian


# reads an xyz file
def readXYZ(filename):
    fp = open(filename, "r")
    numats = int(fp.readline())
    latline = fp.readline()
    if latline.split("=")[0] == "Lattice":
        cellparams = np.asarray(list(map(float, latline.split('"')[1].split())))
        cellparams = cellparams.reshape([3, 3], order="F")
    else:
        cellparams = None
    elems = []
    coors = []
    for line in fp:
        elements = line.split()
        elems.append(elements[0])
        coors.append(list(map(float, elements[1:4])))
    structdict = {}
    structdict["coors"] = np.asarray(coors)
    structdict["cell"] = cellparams
    structdict["elems"] = capitalize(elems)
    return structdict


def get_cell_params(cell, writeout=False):
    a = np.np.sqrt(np.dot(cell[0], cell[0]))
    b = np.np.sqrt(np.dot(cell[1], cell[1]))
    c = np.np.sqrt(np.dot(cell[2], cell[2]))
    alpha = np.arccos(np.dot(cell[1], cell[2]) / b / c)
    beta = np.arccos(np.dot(cell[0], cell[2]) / a / c)
    gamma = np.arccos(np.dot(cell[0], cell[1]) / a / b)
    if writeout:
        print(a)
        print(b)
        print(c)
        print(alpha / 2 / np.pi * 360)
        print(beta / 2 / np.pi * 360)
        print(gamma / 2 / np.pi * 360)
    return a, b, c, alpha, beta, gamma


def fractional_to_cartesian(coors, cell):
    coors = coors[0] * cell[0] + coors[1] * cell[1] + coors[2] * cell[2]
    return coors


def readPOSCAR(filename, from_fp=None):
    structdict = {}
    if from_fp is not None:
        fp = from_fp
    else:
        fp = open(filename, "r")
    comment = fp.readline()
    latvecscale = float(fp.readline())
    cell = np.zeros((3, 3))
    for i in range(3):
        vec = fp.readline()
        cell[i] = list(map(float, vec.split()))
    structdict["cell"] = cell
    elline = fp.readline().split()
    ellist = None
    try:
        float(elline[0])
        print("no elinfo found")
    except:
        ellist = elline
        elline = fp.readline().split()
    natoms = 0
    structdict["elems"] = []
    elline = list(map(int, elline))
    for i, element in enumerate(elline):
        natoms += element
        if ellist != None:
            structdict["elems"] += [ellist[i]] * element
    structdict["coors"] = np.zeros([natoms, 3])
    mode = fp.readline().split()[0]
    for i in range(natoms):
        vec = fp.readline()
        structdict["coors"][i, :] = list(map(float, vec.split()))
        if (mode == "Direct") | (mode == "Fractional"):
            structdict["coors"][i] = fractional_to_cartesian(
                structdict["coors"][i], cell
            )
    return structdict


# capitalizes first letter of each string in an array
def capitalize(array):
    carray = []
    for e in array:
        carray.append(e.capitalize())
    return np.asarray(carray)


# read a mesh yaml file to only get  frequencies in a yaml file
def read_yaml_freqs(fname):
    fp = open(fname, "r")
    for line in fp:
        elements = line.split()
        if len(elements) >= 1:
            if elements[0] == "nqpoint:":
                nqpoint = int(elements[1])
                qpoints = np.zeros([nqpoint, 3])
                weights = np.zeros(nqpoint)
                qcount = -1
            if elements[0] == "frequency:":
                freqcounter += 1
                freqs[qcount, freqcounter] = float(elements[1])
            elif elements[0] == "natom:":
                natom = int(elements[1])
                freqs = np.zeros([nqpoint, natom * 3])
        if len(elements) >= 2:
            if elements[1] == "q-position:":
                qcount += 1
                qpoints[qcount, 0] = float(elements[3].split(",")[0])
                qpoints[qcount, 1] = float(elements[4].split(",")[0])
                qpoints[qcount, 1] = float(elements[5].split(",")[0])
                freqcounter = -1
    return qpoints, np.array(freqs)


# read a mesh yaml file containing eigenvectors, phonopy version 1.13
def read_evectors(fname):
    fp = open(fname, "r")

    latmode = 0
    evecmode = -2
    cell = np.zeros([3, 3])
    mesh = np.zeros(3)

    for line in fp:
        elements = line.split()

        if len(elements) > 3:
            if latmode > 0:
                latmode += 1
                cell[latmode - 2, 0] = float(elements[2].split(",")[0])
                cell[latmode - 2, 1] = float(elements[3].split(",")[0])
                cell[latmode - 2, 2] = float(elements[4].split(",")[0])
                if latmode == 4:
                    latmode = 0
            elif (evecmode >= -1) & (elements[1] == "["):
                evecmode += 1
                evecs[
                    qcount, freqcounter, int(np.floor(evecmode / 3)), evecmode % 3
                ] = (
                    float(elements[2].split(",")[0])
                    + float(elements[3].split(",")[0]) * 1j
                )
                if evecmode >= (natom * 3 - 1):
                    evecmode = -2
        if len(elements) >= 1:
            if elements[0] == "lattice:":
                latmode = 1
            elif elements[0] == "nqpoint:":
                nqpoint = int(elements[1])
                qpoints = np.zeros([nqpoint, 3])
                weights = np.zeros(nqpoint)
                qcount = -1
            elif elements[0] == "mesh:":
                mesh[0] = float(elements[2].split(",")[0])
                mesh[1] = float(elements[3].split(",")[0])
                mesh[2] = float(elements[4].split(",")[0])
            elif elements[0] == "weight:":
                weights[qcount] = int(elements[1])
            elif elements[0] == "natom:":
                natom = int(elements[1])
                freqs = np.zeros([nqpoint, natom * 3])
                evecs = np.zeros([nqpoint, natom * 3, natom, 3], dtype=complex)
                gvels = np.zeros([nqpoint, natom * 3, 3])
            elif elements[0] == "frequency:":
                freqcounter += 1
                freqs[qcount, freqcounter] = float(elements[1])
            elif elements[0] == "eigenvector:":
                evecmode = -1
            elif elements[0] == "group_velocity:":
                gvels[qcount, freqcounter, 0] = elements[2].split(",")[0]
                gvels[qcount, freqcounter, 1] = elements[3].split(",")[0]
                gvels[qcount, freqcounter, 2] = elements[4].split(",")[0]
        if len(elements) >= 2:
            if elements[1] == "q-position:":
                qcount += 1
                qpoints[qcount, 0] = float(elements[3].split(",")[0])
                qpoints[qcount, 1] = float(elements[4].split(",")[0])
                qpoints[qcount, 2] = float(elements[5].split(",")[0])
                freqcounter = -1

    return freqs, evecs, qpoints, cell, mesh, weights, gvels


# parses a molsys ff style comment
def parse_comment(fullcomm):
    # print fullcomm
    if (
        (fullcomm.find("-") == -1)
        | (fullcomm.find(">") == -1)
        | (fullcomm.find("|") == -1)
        | (fullcomm.find("(") == -1)
        | (fullcomm.find(")") == -1)
    ):
        return None, None, None
    termtype = fullcomm.split("->")[0].split("#")[1].split(" ")[1]
    frag = fullcomm.split("|")[1]
    atypes = (
        fullcomm.split("->")[1].split("|")[0].split("(")[1].split(")")[0].split(",")
    )
    return termtype, frag, atypes


# reverses a molsys comment
def reverse_comment(fullcomm):
    termtype, frag, atypes = parse_comment(fullcomm)
    rstr = "# " + termtype + "->("
    atypes.reverse()
    rstr += ",".join(atypes) + ")"
    rstr += "|" + frag
    return rstr


def str_is_number(s):
    """Returns True if string is a number."""
    return s.replace(".", "", 1).replace("-", "", 1).isdigit()


def is_numeric(s):
    """returns True if the string is numeric. (saver than previous function)"""
    try:
        float(s)
    except:
        return False
    return True


def read_fpar(fname):

    int_strs = ["bnd", "ang", "dih", "oop", "cha", "vdw"]
    fpar_dict = {}

    fpar_dict["presettings"] = []
    currtypename = ""
    infile = open(fname, "r")
    for line in infile:
        bcomm = line.split("#")
        if len(bcomm) > 1:
            comment = line.split("#")[1]
        else:
            comment = None
        elements = bcomm[0].split()
        if len(elements) == 2:
            typestr = elements[0].split("_")
            if len(typestr) > 1:
                if typestr[1] == "type":
                    currtypename = typestr[0]
                    if currtypename == "vdwpr":
                        fpar_dict["vdwpr_type"] = int(elements[1])
                    if currtypename == "chapr":
                        fpar_dict["chapr_type"] = int(elements[1])
            if elements[0] == "variables":
                currtypename = elements[0]
                fpar_dict["numvars"] = int(elements[1])
            if elements[0] in int_strs:
                currtypename = elements[0]
            if elements[0] == "refsysname":
                fpar_dict["refsysname"] = elements[1]
            if (currtypename not in fpar_dict.keys()) & (currtypename != ""):
                fpar_dict[currtypename] = []
            if currtypename == "":
                fpar_dict["presettings"].append([elements[0], elements[1]])
        elif (
            (len(elements) > 2) & (currtypename != "vdwpr") & (currtypename != "chapr")
        ):
            apel = []
            if currtypename != "variables":
                apel.append(int(elements[0]))
            else:
                apel.append(elements[0])
            for i, el in enumerate(elements[1:]):
                if str_is_number(el):
                    apel.append(float(el))
                else:
                    apel.append(el)

            if comment != None:
                apel.append(comment)

            fpar_dict[currtypename].append(apel)

            # else:

    infile.close()

    return fpar_dict


def write_fpar(dic, fname):
    fp = open(fname, "w")
    for entry in dic["presettings"]:
        fp.write("%s %s\n" % (entry[0], entry[1]))
    fp.write("\n\n")
    types = ["bnd", "ang", "dih", "oop", "cha", "vdw"]
    for t in types:
        fp.write("\n%s_type %d\n" % (t, len(dic[t])))
        for entry in dic[t]:
            st = "%d\t %12s " % (entry[0], entry[1])
            for num in entry[2:-1]:
                if str_is_number(str(num)):
                    st += "%16.8f " % num
                else:
                    st += "%16s " % num
            st += " # %s" % entry[-1]
            fp.write(st)
    fp.write("\nvdwpr_type %d\n\n" % dic["vdwpr_type"])
    fp.write("\nchapr_type %d\n\n" % dic["chapr_type"])
    fp.write("refsysname %s\n\n" % dic["refsysname"])
    fp.write("\n%s %d\n" % ("variables", len(dic["variables"])))
    for variable in dic["variables"]:
        st = "%16s " % (variable[0])
        for num in variable[1:]:
            if str_is_number(str(num)):
                st += "%16.8f " % num
            else:
                st += "%16s " % num
        fp.write(st + "\n")


def dumprms(ref, opt, num, cmt="", silent=False):
    perc = np.mean(abs(opt[3:][ref[3:] < num] / ref[3:][ref[3:] < num] - 1.0))
    rms = np.sqrt(np.mean((ref[ref < num] - opt[ref < num]) ** 2))
    if not silent:
        print("rms " + cmt + " " + str(num) + ": " + str(rms))
        print("average deviation " + cmt + " " + str(num) + ": " + str(perc))
    return perc, rms


# rotates a vector based on two cells A-->B
def rotate_vec(vec, A, B):
    invcell = np.linalg.inv(A)
    frac = np.dot(vec, invcell)
    return np.dot(frac, B)


def rotate_tens(tens, A, B):
    invA = np.linalg.inv(A)
    Q = np.matmul(invA, B)
    # print(Q)
    invQ = np.linalg.inv(Q)
    newtens = np.matmul(np.matmul(invQ, tens), Q)
    return newtens








# calculates a normal distribution at a certain value with a certain sigma over a range of data points
def gaussian(freqs, mode, sigma):
    return (
        1.0
        / np.sqrt(2.0 * np.pi * sigma**2)
        * np.exp(-((freqs - mode) ** 2.0) / (2.0 * sigma**2))
    )


def wgt_gaussian(freqs, mode, sigma, wgt):
    return (
        1.0
        / np.sqrt(2.0 * np.pi * sigma**2)
        * np.exp(-((freqs - mode) ** 2.0) / (2.0 * sigma**2))
    ) * wgt


# save and load objects in pickles
def save_obj(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


# calculates the root mean square deviation between two given datasets
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


# calculates a spectrum of rmses over a certain range
def rmse_spec(predictions, targets, minval=0, maxval=50, numvals=100):
    rmsetoeval = np.linspace(minval, max(predictions) + maxval, numvals)
    rmses = []
    for val in rmsetoeval:
        rmses.append(
            rmse(
                np.asarray(predictions)[predictions < val],
                np.asarray(targets)[predictions < val],
            )
        )
    return rmses


# basically only changes the format of the data
def read_mfpx(fname):
    import molsys

    mol = molsys.mol.from_file(fname)
    outdic = {}
    outdic["coors"] = mol.xyz
    outdic["cell"] = mol.cell
    outdic["elems"] = mol.elems

    return outdic


# file conversion from mfpx to POSCAR that includes everything
def mfpx_to_POSCAR(fname, outname="POSCAR"):
    os.system("x2x2 " + fname + " temp.xyz")
    if os.path.exists("POSCAR"):
        os.system("mv POSCAR temp.poscar")
    os.system("atomsk temp.xyz POSCAR")
    if outname != "POSCAR":
        os.system("mv POSCAR " + outname)
        if os.path.exists("temp.poscar"):
            os.system("mv temp.poscar POSCAR")
    os.system("rm temp.*")
    return


# reads a lammps data file
def read_lammps_data(fname):
    outdic = {}
    fp = open(fname, "r")
    outdic["commenttext"] = fp.readline()
    outdic["masses"] = []
    outdic["coors"] = []
    outdic["charges"] = []
    outdic["species"] = []
    outdic["bndtypes"] = []
    outdic["bonds"] = []
    outdic["angtypes"] = []
    outdic["angles"] = []
    outdic["dihtypes"] = []
    outdic["dihedrals"] = []
    outdic["imptypes"] = []
    outdic["impropers"] = []
    outdic["atcomments"] = []
    outdic["macomments"] = []
    outdic["bndcomments"] = []
    outdic["angcomments"] = []
    outdic["dihcomments"] = []
    outdic["impcomments"] = []
    outdic["massids"] = []
    # indicates position in file
    # 0 ... number of interactions, beginning of file
    mode = 0
    for line in fp:
        elements = line.split()
        nels = len(elements)
        # detect mode
        if nels >= 1:
            if elements[0] == "Masses":
                mode = 1
            elif elements[0] == "Atoms":
                mode = 2
            elif elements[0] == "Bonds":
                mode = 3
            elif elements[0] == "Angles":
                mode = 4
            elif elements[0] == "Dihedrals":
                mode = 5
            elif elements[0] == "Impropers":
                mode = 6
            else:
                # mode indicating beginning of file before masses
                if mode == 0:
                    # read number of atoms etc.
                    if nels == 2:
                        amount = int(elements[0])
                        outdic["n" + elements[1]] = amount
                    elif nels == 3:
                        if elements[2] == "types":
                            amount = int(elements[0])
                            outdic["n" + elements[1] + " " + elements[2]] = amount

                    # cell information
                    elif nels == 4:
                        direction = elements[2][0]
                        outdic[direction + direction] = float(elements[1]) - float(
                            elements[0]
                        )
                    # off diagonals
                    elif nels == 6:
                        for i in range(3):
                            outdic[elements[i + 3]] = float(elements[i])
                # read masses
                elif mode == 1:
                    if nels > 0:
                        if len(outdic["masses"]) < outdic["natom types"]:
                            outdic["massids"].append(int(elements[0]))
                            outdic["masses"].append(float(elements[1]))
                            if "#" in line:
                                outdic["macomments"].append(
                                    line.split("#")[1].split("\n")[0]
                                )
                # read atoms
                elif mode == 2:
                    if nels > 0:
                        # atom style full
                        if len(line.split("#")[0].split()) == 7:
                            outdic["coors"].append(
                                [
                                    float(elements[4]),
                                    float(elements[5]),
                                    float(elements[6]),
                                ]
                            )
                            outdic["charges"].append(float(elements[3]))
                            outdic["species"].append(int(elements[2]))
                        # atom style atomic
                        elif len(line.split("#")[0].split()) == 6:
                            outdic["coors"].append(
                                [
                                    float(elements[3]),
                                    float(elements[4]),
                                    float(elements[5]),
                                ]
                            )
                            outdic["charges"].append(float(elements[2]))
                            outdic["species"].append(int(elements[1]))
                        elif len(line.split("#")[0].split()) == 5:
                            outdic["coors"].append(
                                [
                                    float(elements[2]),
                                    float(elements[3]),
                                    float(elements[4]),
                                ]
                            )
                            outdic["charges"].append(None)
                            outdic["species"].append(int(elements[1]))
                        if "#" in line:
                            outdic["atcomments"].append(
                                line.split("#")[1].split("\n")[0]
                            )
                elif mode == 3:
                    if nels > 0:
                        outdic["bndtypes"].append(int(elements[1]))
                        outdic["bonds"].append([int(elements[2]), int(elements[3])])
                        if "#" in line:
                            outdic["bndcomments"].append(
                                line.split("#")[1].split("\n")[0]
                            )
                elif mode == 4:
                    if nels > 0:
                        outdic["angtypes"].append(int(elements[1]))
                        outdic["angles"].append(
                            [int(elements[2]), int(elements[3]), int(elements[4])]
                        )
                        if "#" in line:
                            outdic["angcomments"].append(
                                line.split("#")[1].split("\n")[0]
                            )
                elif mode == 5:
                    if nels > 0:
                        outdic["dihtypes"].append(int(elements[1]))
                        outdic["dihedrals"].append(
                            [
                                int(elements[2]),
                                int(elements[3]),
                                int(elements[4]),
                                int(elements[5]),
                            ]
                        )
                        if "#" in line:
                            outdic["dihcomments"].append(
                                line.split("#")[1].split("\n")[0]
                            )
                elif mode == 6:
                    if nels > 0:
                        outdic["imptypes"].append(int(elements[1]))
                        outdic["impropers"].append(
                            [
                                int(elements[2]),
                                int(elements[3]),
                                int(elements[4]),
                                int(elements[5]),
                            ]
                        )
                        if "#" in line:
                            outdic["impcomments"].append(
                                line.split("#")[1].split("\n")[0]
                            )

    return outdic


def write_lammps_data(fname, dic):
    fp = open(fname, "w")
    fp.write("%s\n" % (dic["commenttext"]))
    interacts = [
        "atoms",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "atom types",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
    ]
    for interact in interacts:
        if ("n" + interact) in dic.keys():
            fp.write("%12d %s\n" % (dic["n" + interact], interact))
    fp.write("\n")
    coorls = ["x", "y", "z"]
    for l in coorls:
        fp.write("%12.6f %12.6f %s %s\n" % (0, dic[l + l], l + "lo ", l + "hi"))
    if "xy" in dic.keys():
        fp.write(
            "%12.6f %12.6f %12.6f %s %s %s\n"
            % (dic["xy"], dic["xz"], dic["yz"], "xy", "xz", "yz")
        )
    fp.write("\nMasses\n\n")
    for i, mass in enumerate(dic["masses"]):
        fp.write(
            "%12d %12.6f # %s\n"
            % (dic["massids"][i], dic["masses"][i], dic["macomments"][i])
        )
    fp.write("\nAtoms\n\n")
    for i, at in enumerate(dic["coors"]):
        fp.write(
            "%12d %6d %6d %12.6f %12.6f %12.6f %12.6f # %s\n"
            % (
                i + 1,
                1,
                dic["species"][i],
                dic["charges"][i],
                at[0],
                at[1],
                at[2],
                dic["atcomments"][i],
            )
        )
    fp.write("\nBonds\n\n")
    for i, bnd in enumerate(dic["bonds"]):
        fp.write(
            "%12d %6d %6d %6d # %s\n"
            % (i + 1, dic["bndtypes"][i], bnd[0], bnd[1], dic["bndcomments"][i])
        )
    fp.write("\nAngles\n\n")
    for i, ang in enumerate(dic["angles"]):
        fp.write(
            "%12d %6d %6d %6d %6d # %s\n"
            % (i + 1, dic["angtypes"][i], ang[0], ang[1], ang[2], dic["angcomments"][i])
        )
    fp.write("\nDihedrals\n\n")
    for i, dih in enumerate(dic["dihedrals"]):
        fp.write(
            "%12d %6d %6d %6d %6d %6d # %s\n"
            % (
                i + 1,
                dic["dihtypes"][i],
                dih[0],
                dih[1],
                dih[2],
                dih[3],
                dic["dihcomments"][i],
            )
        )
    fp.write("\nImpropers\n\n")
    for i, imp in enumerate(dic["impropers"]):
        fp.write(
            "%12d %6d %6d %6d %6d %6d # %s\n"
            % (
                i + 1,
                dic["imptypes"][i],
                imp[0],
                imp[1],
                imp[2],
                imp[3],
                dic["impcomments"][i],
            )
        )


def write_lammps_atomic(
    fname,
    cell,
    elems,
    coors,
    masses=None,
    atom_types=None,
    charges=None,
    bond_num=None,
    uelemsort=False,
):
    # only write out coordinates without any extra information about bonds dihedrals etc.
    # masses can be added by hand for each element type - otherwise they will automatically
    # be assigned by a table
    # separate atom types can be defined as well
    # if charges are specified - the file type will be "charge" instead
    # if bond_num is given - space for this many bond types will be reserved
    with open(fname, "w") as fp:
        fp.write(" # generated by swscripts\n\n")
        fp.write("%12d %s\n" % (len(coors), "atoms"))

        if atom_types is None:
            uelems, uinds = np.unique(elems, return_index=True)
        else:
            uelems, uinds = np.unique(atom_types, return_index=True)
        #        print(uelems,uinds)
        if uelemsort:
            logical = np.argsort(uelems)
        else:
            logical = np.argsort(uinds)
        uelems = uelems[logical]
        #        print(uelems,uinds)
        fp.write("%12d %s\n" % (len(uelems), "atom types"))
        if bond_num is not None:
            fp.write("%12d %s\n\n" % (bond_num, "bond types"))
        fp.write("\n")
        assert (
            (cell[0, 1] == 0) & (cell[0, 2] == 0) & (cell[1, 2] == 0)
        ), "ERROR during writing lmp file: cell is not an upper triangular matrix!"

        fp.write("%16.10f %16.10f %s %s\n" % (0, cell[0, 0], "xlo", "xhi"))
        fp.write("%16.10f %16.10f %s %s\n" % (0, cell[1, 1], "ylo", "yhi"))
        fp.write("%16.10f %16.10f %s %s\n" % (0, cell[2, 2], "zlo", "zhi"))
        fp.write(
            "%16.10f %16.10f %16.10f %s %s %s\n"
            % (cell[1, 0], cell[2, 0], cell[2, 1], "xy", "xz", "yz")
        )

        fp.write("Masses\n\n")

        # assign masses based on element name if none are included in the input
        if masses is None:
            import mlp_converters.aprops.elems as elems

            mass_data = elems.mass
            masses = []
            for uid, uel in enumerate(uelems):
                if atom_types is not None:
                    corresponding_elems = elems[np.array(atom_types) == uel]
                    found_els = np.unique(corresponding_elems)
                    assert (
                        len(found_els) == 1
                    ), "ERROR during assignment of masses: all atoms of a specific atom type need to be of the same element!"
                    uel = found_els[0]
                if uel.lower() in mass_data.keys():
                    masses.append(mass_data[uel.lower()])
                else:
                    print("WARNING during writing lmp file: mass for %s missing" % uel)
                    masses.append(0)

        for uid, uel in enumerate(uelems):
            fp.write("%12d %14.8f # %s\n" % (uid + 1, masses[uid], uel))
        fp.write("\nAtoms # atomic\n\n")
        for cid, coor in enumerate(coors):
            if atom_types is not None:
                typeid = np.where(atom_types[cid] == uelems)[0] + 1
            else:
                typeid = np.where(elems[cid] == uelems)[0] + 1
            if charges is not None:
                if bond_num is not None:
                    # in this case it is a hybrid atom style with a different syntax. Here, it is assumed that charge is first, then bond
                    fp.write(
                        "%12d %5d %16.10f %16.10f %16.10f %16.10f %5d\n"
                        % (cid + 1, typeid, coor[0], coor[1], coor[2], charges[cid], 1)
                    )
                else:
                    fp.write(
                        "%12d %5d %16.10f %16.10f %16.10f %16.10f\n"
                        % (cid + 1, typeid, charges[cid], coor[0], coor[1], coor[2])
                    )
            elif bond_num is not None:
                # style bond - need to specify molecule id
                fp.write(
                    "%12d %5d %5d %16.10f %16.10f %16.10f\n"
                    % (cid + 1, 1, typeid, coor[0], coor[1], coor[2])
                )
            else:
                fp.write(
                    "%12d %5d %16.10f %16.10f %16.10f\n"
                    % (cid + 1, typeid, coor[0], coor[1], coor[2])
                )


def write_xyz(fname, cell, elems, xyz):
    fp = open(fname, "w")
    numats = len(elems)
    fp.write("%d\n" % numats)
    if cell is not None:
        fp.write(
            'Lattice="%14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f" \n'
            % tuple(cell.ravel(order="F"))
        )
    else:
        fp.write("nonperiodic \n")
    for aid, atom in enumerate(xyz):
        fp.write("%3s %14.8f %14.8f %14.8f\n" % (elems[aid], atom[0], atom[1], atom[2]))
    fp.close()


# write a geometry in the VASP POSCAR file format
def write_POSCAR(fname, cell, elems, xyz, comment=None, velocities=None):
    fp = open(fname, "w")
    if comment is None:
        fp.write("# file generated with sw scripts\n")
    else:
        fp.write("# " + comment + "\n")
    # scaling - always 1 in this case
    fp.write("%14.8f\n" % (1.0))
    # cell vectors
    fp.write("%14.8f %14.8f %14.8f\n" % (cell[0, 0], cell[0, 1], cell[0, 2]))
    fp.write("%14.8f %14.8f %14.8f\n" % (cell[1, 0], cell[1, 1], cell[1, 2]))
    fp.write("%14.8f %14.8f %14.8f\n" % (cell[2, 0], cell[2, 1], cell[2, 2]))
    # atom line
    cat = ""
    elist = []
    enums = []
    for eid, el in enumerate(elems):
        if el == cat:
            enums[-1] += 1
        else:
            cat = el
            elist.append(el)
            enums.append(1)
            fp.write("%5s" % (el))
    fp.write("\n")
    for num in enums:
        fp.write("%8d" % (num))
    fp.write("\n")
    fp.write("Cartesian\n")
    for aid, atom in enumerate(xyz):
        fp.write("%14.8f %14.8f %14.8f\n" % (atom[0], atom[1], atom[2]))

    if velocities is not None:
        fp.write("\n")
        for vel in velocities:
            fp.write("%14.8f %14.8f %14.8f\n" % (vel[0], vel[1], vel[2]))

    fp.close()


# maps string directions to indices
def dirmap(strdir):
    direc = None
    if strdir == "x":
        direc = 0
    elif strdir == "y":
        direc = 1
    elif strdir == "z":
        direc = 2
    return direc


# displace coordinates based on a per-atom set of vectors by a specified amplitude
def displace_atoms_along_vectors(coors, vectors, amplitude):
    xyz = np.shape(coors)
    xyz = coors + np.real(vectors) * amplitude
    # for vid,vector in enumerate(vectors):
    #  print coors[vid],np.real(vector),amplitude
    #  xyz[vid] = coors[vid]+np.real(vector)*amplitude

    return xyz


# create a directory if it does not exist yet
def mkdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def align_subplots(numvals, figsize=None):
    import matplotlib.pyplot as plt

    psize = int(numvals**0.5)
    diff = numvals - psize**2
    if diff == 0:
        psizex = psize
        psizey = psize
    elif diff <= psize:
        psizex = psize + 1
        psizey = psize
    else:
        psizex = psize + 1
        psizey = psize + 1
    if figsize is None:
        figsize = (4 * psizex, 4 * psizey)
    f, axs = plt.subplots(psizex, psizey, figsize=figsize)
    return f, axs


# function to compute the overlap integral between two functions
def overlap_integral(func1, func2, xdat=None):
    if xdat is None:
        return (
            np.trapz(func1 * func2) / (np.trapz(func1**2) * np.trapz(func2**2)) ** 0.5
        )
    else:
        return (
            np.trapz(func1 * func2, x=xdat)
            / (np.trapz(func1**2, x=xdat) * np.trapz(func2**2, x=xdat)) ** 0.5
        )


def running_avg(data, w):
    """
    function to compute the running average from a dataset data with the width w
    for equidistant datapoints
    """
    numvals = len(data) - w + 1
    return np.convolve(data, np.ones(w), "valid") / w, numvals
