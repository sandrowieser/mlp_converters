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


import sys
import ase.io


def main():
    assert len(sys.argv) >= 3, "ERROR: need to specify input and output file name"
    inname = sys.argv[1]
    outname = sys.argv[2]

    if "ML_AB" in inname or ".cfg" in inname:
        from mlp_converters.fileIO.ml_abn import ml_abn
        if "ML_AB" in inname:
            mlp_traj = ml_abn(fname=inname, filetype="ML_ABN")
        else:
            if len(sys.argv) > 3 :
                atom_types = sys.argv[3:]
            else:
                atom_types = None
            mlp_traj = ml_abn(fname=inname, filetype="cfg", atom_types=atom_types)
        
        if "ML_AB" in outname:
            mlp_traj.write_ML_ABN(fname=outname)
        elif ".cfg" in outname:
            mlp_traj.write_MLIP_cfg(fname=outname)
        else:
            alist = mlp_traj.get_alist()
            ase.io.write(outname, alist)

    
    elif "dyn" in inname or "mfpx" in inname:
        # special file formats supported by the legacy geometry object
        from mlp_converters.geometry import geo
        geom = geo(inname)
        geom.write_file(outname)

    else:
        # in all other cases, we assume it can be read with ASE
        from mlp_converters.fileIO.ml_abn import ml_abn
        alist = ase.io.read(inname, index=":")
        if type(alist) is not list:
            alist = [alist]
        mlp_traj = ml_abn(filetype="direct")
        mlp_traj.from_atoms(alist)
        if "ML_AB" in outname:
            mlp_traj.write_ML_ABN(fname=outname)
        elif ".cfg" in outname:
            mlp_traj.write_MLIP_cfg(fname=outname)
        else:
            raise NotImplementedError("output filetype not known")


        

if __name__ == "__main__":
    main()