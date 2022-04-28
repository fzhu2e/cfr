from multiprocessing.sharedctypes import Value
import os
import copy
from shutil import ReadError
import yaml
import pandas as pd
from .climate import ClimateField
from .proxy import ProxyDatabase

from .utils import (
    p_header,
    p_hint,
    p_success,
    p_fail,
    p_warning,
)

import pprint
pp = pprint.PrettyPrinter()

class ReconJob:
    ''' Reconstruction Job
    '''
    def __init__(self, configs=None, verbose=False):
        ''' Initialize a reconstruction job

        Args:
            configs (dict, optional): a dictionary of configurations. Defaults to None.
        '''
        self.configs = {} if configs is None else configs
        if verbose:
            p_header(f'>>> job.configs:')
            pp.pprint(self.configs)

    def copy(self):
        return copy.deepcopy(self)

    def load_proxydb(self, path=None, verbose=False, **kwargs):
        if path is not None:
            self.configs.update({'proxydb_path': path})
            if verbose: p_header(f'>>> job.configs["proxydb_path"] = {path}')
        else:
            path = self.configs['proxydb_path']

        _, ext =  os.path.splitext(path)
        if ext.lower() == '.pkl':
            df = pd.read_pickle(path)
        else:
            raise ReadError(f'The extention of the file [{ext}] is not supported. Support list: [.pkl, ] .')

        self.proxydb = ProxyDatabase().from_df(df, **kwargs)
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records loaded')
            p_success(f'>>> job.proxydb created')

    def filter_proxydb(self, *args, verbose=False, **kwargs):
        self.proxydb = self.proxydb.filter(*args, **kwargs)
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records remaining')
            p_success(f'>>> job.proxydb updated')

    def annualize_proxydb(self, verbose=False, **kwargs):
        self.proxydb = self.proxydb.annualize(**kwargs)
        if verbose:
            p_success(f'>>> {self.proxydb.nrec} records remaining')
            p_success(f'>>> job.proxydb updated')

    def load_gridded(self, tag, path_dict=None, rename_dict=None, center_period=None, verbose=False):
        if path_dict is not None:
            self.configs.update({'{tag}_path': path_dict})
            if verbose: p_header(f'>>> job.configs["{tag}_path"] = {path_dict}')
        else:
            path_dict = self.configs[f'{tag}_path']

        self.obs = {}
        for vn, path in path_dict.items():
            if rename_dict is None:
                vn_in_file = vn
            else:
                vn_in_file = rename_dict[vn]

            self.__dict__[tag][vn] = ClimateField().load_nc(path, vn=vn_in_file).center(center_period)

        if verbose:
            p_success(f'>>> instrumental observation variables {list(self.__dict__[tag].keys())} loaded')
            p_success(f'>>> job.{tag} created')

    def annualize_ds(self, tag, verbose=False, **kwargs):
        for vn, fd in self.__dict__[tag].items():
            if verbose: p_header(f'Processing {vn} ...')
            self.__dict__[tag][vn] = fd.annualize(**kwargs)

        if verbose:
            p_success(f'>>> job.{tag} updated')
