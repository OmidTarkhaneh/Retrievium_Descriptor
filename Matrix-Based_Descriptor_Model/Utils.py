# -*- coding: utf-8 -*-
"""Retrievium_Descriptor_Model utils


"""

# -*- coding: utf-8 -*-

import torch.nn.functional as F
import torch
# import torchani
import os
import math
import torch.utils.tensorboard
import numpy as np
import pandas as pd


# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch


import math
from typing import Tuple, Optional, NamedTuple





##################################################################################
import torch
from collections import OrderedDict
from torch import Tensor
from typing import Tuple, NamedTuple, Optional
from sklearn.preprocessing import QuantileTransformer

# import utils


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor


import pandas as pd

class ProposedModel(torch.nn.ModuleDict):
    """
    """

    @staticmethod
    def ensureOrderedDict(modules):
        if isinstance(modules, OrderedDict):
            return modules
        od = OrderedDict()
        for i, m in enumerate(modules):
            od[str(i)] = m
        return od

    def __init__(self, modules):
        super().__init__(self.ensureOrderedDict(modules))

    def forward(self, species_aev: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]

        atomic_energies = self._atomic_energies((species, aev))
        # print(atomic_energies)
        # shape of atomic energies is (C, A)
        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    @torch.jit.export
    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species, aev = species_aev

        # Normalizing AEV

        aev_temp=aev
        aev=aev.view(aev.size(0), -1)
        aev = F.normalize(aev, p=2, dim=1)
        aev=aev.view_as(aev_temp)


        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        df=pd.DataFrame()

        output = aev.new_zeros(species_.shape)

        for i, m in enumerate(self.values()):
            lst=[1,6,7,8,16,17]
            mask = (species_ == lst[i])
            midx = mask.nonzero().flatten()
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)

                output.masked_scatter_(mask, m(input_).flatten())
        output = output.view_as(species)
        return output



class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc)
        return input_





##################################################################################
"""# Utils"""
##################################################################################


import torch


import math
from collections import defaultdict
from typing import Tuple, NamedTuple, Optional
# from units import sqrt_mhessian2invcm, sqrt_mhessian2milliev, mhessian2fconst
# from nn import SpeciesEnergies


def stack_with_padding(properties, padding):
    output = defaultdict(list)
    for p in properties:
        for k, v in p.items():
            output[k].append(torch.as_tensor(v))
    for k, v in output.items():
        if v[0].dim() == 0:
            output[k] = torch.stack(v)
        else:
            output[k] = torch.nn.utils.rnn.pad_sequence(v, True, padding[k])
    return output


def broadcast_first_dim(properties):
    num_molecule = 1
    for k, v in properties.items():
        shape = list(v.shape)
        n = shape[0]
        if num_molecule != 1:
            assert n == 1 or n == num_molecule, "unable to broadcast"
        else:
            num_molecule = n
    for k, v in properties.items():
        shape = list(v.shape)
        shape[0] = num_molecule
        properties[k] = v.expand(shape)
    return properties






class EnergyShifter(torch.nn.Module):
    """Helper class for adding and subtracting self atomic energies

    This is a subclass of :class:`torch.nn.Module`, so it can be used directly
    in a pipeline as ``[input->AEVComputer->ANIModel->EnergyShifter->output]``.

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): Sequence of floating
            numbers for the self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
        fit_intercept (bool): Whether to calculate the intercept during the LSTSQ
            fit. The intercept will also be taken into account to shift energies.
    """

    def __init__(self, self_energies, fit_intercept=False):
        super().__init__()

        self.fit_intercept = fit_intercept
        if self_energies is not None:
            self_energies = torch.tensor(self_energies, dtype=torch.double)

        self.register_buffer('self_energies', self_energies)

    # def __getitem__(self, key):
    #   return self.__dict__(key)

    def sae(self, species):
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """

        species[species == 0] = -1
        species[species == 1] = 0
        species[species == 6] = 1
        species[species == 7] = 2
        species[species == 8] = 3
        # species[species == 9] = 4
        species[species == 16] = 4
        species[species == 17] = 5
        # species[species == 35] = 7

        species=species.type(torch.long)

        intercept = 0.0
        if self.fit_intercept:
            intercept = self.self_energies[-1]

        self_energies = self.self_energies[species].to(species.device)
        # Fix the problem with species in CUDA and self_energies in CPU
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        return self_energies.sum(dim=1) + intercept

    def forward(self, species_energies: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        species, energies = species_energies
        sae = self.sae(species)
        return SpeciesEnergies(species, energies + sae)





PERIODIC_TABLE = ['Dummy'] + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()



# __all__ = ['pad_atomic_properties']

"""# Data Loader"""

##################################################################################
