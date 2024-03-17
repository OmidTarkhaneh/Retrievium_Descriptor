"""# Data Loader File"""
##################################################################################


# Written by Roman Zubatyuk and Justin S. Smith
import h5py
import os
import numpy as np
import pandas as pd
from Utils import *

class anidataloader:

    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - ' + store_file)
        self.store = h5py.File(store_file, 'r')

    def h5py_dataset_iterator(self, g, prefix=''):
        """Group recursive iterator

        Iterate through all groups in all branches and return datasets in dicts)
        """
        data = {}
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            # keys = [i for i in item.keys()]
            # if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset

            #     for k in key:
            if not isinstance(item, h5py.Group):
                dataset = np.array(item[()])

                if isinstance(dataset, np.ndarray):
                    if dataset.size != 0:
                        if isinstance(dataset[0], np.bytes_):
                            dataset = [a.decode('ascii')
                                        for a in dataset]
                data.update({key: dataset})
        yield data
        # else:  # test for group (go down)
        #     yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)"""
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    def get_group_list(self):
        """Returns a list of all groups in the file"""
        return [g for g in self.store.values()]

    def iter_group(self, g):
        """Allows interation through the data in a given group"""
        for data in self.h5py_dataset_iterator(g):
            yield data

    def get_data(self, path, prefix=''):
        """Returns the requested dataset"""
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k][()])

                if isinstance(dataset, np.ndarray):
                    if dataset.size != 0:
                        if isinstance(dataset[0], np.bytes_):
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    def group_size(self):
        """Returns the number of groups"""
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    def cleanup(self):
        """Close the HDF5 file"""
        self.store.close()



from os.path import join, isfile, isdir
import os
# from ._pyanitools import anidataloader
# from .. import utils
import importlib
import functools
import math
import random
from collections import Counter
import numpy
import gc

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

verbose = True

PROPERTIES = ('energies',)

PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0
}


def collate_fn(samples, padding=None):
    if padding is None:
        padding = PADDING

    return stack_with_padding(samples, padding)


class IterableAdapter:
    """https://stackoverflow.com/a/39564774"""
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())


class IterableAdapterWithLength(IterableAdapter):

    def __init__(self, iterable_factory, length):
        super().__init__(iterable_factory)
        self.length = length

    def __len__(self):
        return self.length


class Transformations:
    """Convert one reenterable iterable to another reenterable iterable"""

    @staticmethod
    def species_to_indices(reenterable_iterable, species_order=('H', 'C', 'N', 'O','S','Cl')):

        # species_order='periodic_table'
        if species_order == 'periodic_table':
            species_order = PERIODIC_TABLE
        idx = {k: i for i, k in enumerate(species_order)}

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                d['species'] = numpy.array([idx[s] for s in d['species']], dtype='i8')
                yield d
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def subtract_self_energies(reenterable_iterable, self_energies=None, species_order=None):
        intercept = 0.0
        shape_inference = False
        if isinstance(self_energies, EnergyShifter):
            shape_inference = True
            shifter = self_energies
            self_energies = {}
            counts = {}
            Y = []
            for n, d in enumerate(reenterable_iterable):
                species = d['species']
                count = Counter()
                for s in species:
                    if s!=0:
                        count[s] += 1
                for s, c in count.items():
                    if s not in counts:
                        counts[s] = [0] * n
                    counts[s].append(c)
                for s in counts:
                    if len(counts[s]) != n + 1:
                        counts[s].append(0)
                Y.append(d['energies'])

            # sort based on the order in periodic table by default
            if species_order is None:
                species_order = PERIODIC_TABLE

            # species = sorted(list(counts.keys()), key=lambda x: species_order.index(x))

            species=[1.0,6.0,7.0,8.0,16.0,17.0]



            X = [counts[s] for s in species]
            if shifter.fit_intercept:
                X.append([1] * n)
            X = numpy.array(X).transpose()
            Y = numpy.array(Y)
            if Y.shape[0] == 0:
                raise RuntimeError("subtract_self_energies could not find any energies in the provided dataset.\n"
                                   "Please make sure the path provided to data.load() points to a dataset has energies and is not empty or corrupted.")
            sae, _, _, _ = numpy.linalg.lstsq(X, Y, rcond=None)
            sae_ = sae
            if shifter.fit_intercept:
                intercept = sae[-1]
                sae_ = sae[:-1]
            for s, e in zip(species, sae_):
                self_energies[s] = e
            shifter.__init__(sae, shifter.fit_intercept)
        gc.collect()

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                e = intercept
                for s in d['species']:
                    if s!=0:
                        e += self_energies[s]
                d['energies'] -= e
                yield d
        if shape_inference:
            return IterableAdapterWithLength(reenterable_iterable_factory, n)
        return IterableAdapter(reenterable_iterable_factory)


    @staticmethod
    def remove_outliers(reenterable_iterable, threshold1=15.0, threshold2=8.0):
        assert 'subtract_self_energies', "Transformation remove_outliers can only run after subtract_self_energies"

        # pass 1: remove everything that has per-atom energy > threshold1
        def scaled_energy(x):

            cc=x['species'].astype('long')
            num_atoms = len(cc[cc!=0])

            return abs(x['energies']) / math.sqrt(num_atoms)
        filtered = IterableAdapter(lambda: (x for x in reenterable_iterable if scaled_energy(x) < threshold1))

        # pass 2: compute those that are outside the mean by threshold2 * std
        n = 0
        mean = 0
        std = 0
        for m in filtered:
            n += 1
            mean += m['energies']
            std += m['energies'] ** 2
        mean /= n
        std = math.sqrt(std / n - mean ** 2)

        return IterableAdapter(lambda: filter(lambda x: abs(x['energies'] - mean) < threshold2 * std, filtered))

    @staticmethod
    def shuffle(reenterable_iterable):
        if isinstance(reenterable_iterable, list):
            list_ = reenterable_iterable
        else:
            list_ = list(reenterable_iterable)
            del reenterable_iterable
            gc.collect()
        random.shuffle(list_)
        return list_

    @staticmethod
    def cache(reenterable_iterable):
        if isinstance(reenterable_iterable, list):
            return reenterable_iterable
        ret = list(reenterable_iterable)
        del reenterable_iterable
        gc.collect()
        return ret

    @staticmethod
    def collate(reenterable_iterable, batch_size, padding=None):
        def reenterable_iterable_factory(padding=None):
            batch = []
            i = 0
            for d in reenterable_iterable:
                batch.append(d)
                i += 1
                if i == batch_size:
                    i = 0
                    yield collate_fn(batch, padding)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch, padding)

        reenterable_iterable_factory = functools.partial(reenterable_iterable_factory,
                                                         padding)
        try:
            length = (len(reenterable_iterable) + batch_size - 1) // batch_size
            return IterableAdapterWithLength(reenterable_iterable_factory, length)
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def pin_memory(reenterable_iterable):
        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                yield {k: d[k].pin_memory() for k in d}
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)


class TransformableIterable:
    def __init__(self, wrapped_iterable, transformations=()):
        self.wrapped_iterable = wrapped_iterable
        self.transformations = transformations

    def __iter__(self):
        return iter(self.wrapped_iterable)

    def __getattr__(self, name):
        transformation = getattr(Transformations, name)

        @functools.wraps(transformation)
        def f(*args, **kwargs):
            return TransformableIterable(
                transformation(self.wrapped_iterable, *args, **kwargs),
                self.transformations + (name,))

        return f

    def split(self, *nums):
        length = len(self)
        iters = []
        self_iter = iter(self)
        for n in nums:
            list_ = []
            if n is not None:
                for _ in range(int(n * length)):
                    list_.append(next(self_iter))
            else:
                for i in self_iter:
                    list_.append(i)
            iters.append(TransformableIterable(list_, self.transformations + ('split',)))
        del self_iter
        gc.collect()
        return iters

    def __len__(self):
        return len(self.wrapped_iterable)


def load(path, additional_properties=()):
    properties = PROPERTIES + additional_properties

    def h5_files(path):
        """yield file name of all h5 files in a path"""
        if isdir(path):
            for f in os.listdir(path):
                f = join(path, f)
                yield from h5_files(f)
        elif isfile(path) and (path.endswith('.h5') or path.endswith('.hdf5')):
            yield path

    def molecules():
        for f in h5_files(path):
            anidata = anidataloader(f)
            anidata_size = anidata.group_size()
            use_pbar = PKBAR_INSTALLED and verbose
            if use_pbar:
                pbar = pkbar.Pbar('=> loading {}, total molecules: {}'.format(f, anidata_size), anidata_size)
            for i, m in enumerate(anidata):
                yield m
                if use_pbar:
                    pbar.update(i)

    def conformations():
        for m in molecules():
            species = m['species']
            coordinates = m['data']
            for i in range(coordinates.shape[0]):
                ret = {'species': species[i], 'coordinates': coordinates[i]}
                for k in properties:
                    if k in m:
                        ret[k] = m[k][i]
                yield ret

    return TransformableIterable(IterableAdapter(lambda: conformations()))


__all__ = ['load', 'collate_fn']