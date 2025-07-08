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

"""
 this file provides a class to read the atomic energies from a VASP MD run with machine learned force fields
"""

class ml_eatom():
    def __init__(self, root_direc=".", rmax=None):
        
        self.read_ML_EATOM(root_direc+"/ML_EATOM", rmax=rmax)
        return
    
    def read_ML_EATOM(self, fname, rmax):
        """
        as this file can be very large it might be sensible to do this with an actual 
        file handler
        """
        fp = open(fname)
        aenergies = []
        timestep = []
        for line in fp:
            elements = line.split()
            if len(elements) == 2:
                if elements[0] == "NSTEP=":
                    timestep.append(int(elements[1]))
                    if rmax is not None:
                        if len(timestep) == rmax:
                            break
                    aenergies.append([])
            elif len(elements) == 6:
                aenergies[-1].append(float(elements[5]))
        self.timestep = np.array(timestep)
        self.aenergies = np.array(aenergies)
        self.structenergies = np.sum(self.aenergies,axis=1)
        return self.timestep, self.aenergies
        
