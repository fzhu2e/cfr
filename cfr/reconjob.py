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
            p_header(f'CFR >>> job.configs:')
            pp.pprint(self.configs)

    def copy(self):
        return copy.deepcopy(self)

    def load_proxydb(self, path=None, verbose=False, **kwargs):
        if path is not None:
            self.configs.update({'proxydb_path': path})
            if verbose: p_header(f'job.configs["proxydb_path"] = {path}')
        else:
            path = self.configs['proxydb_path']

        _, ext =  os.path.splitext(path)
        if ext.lower() == '.pkl':
            df = pd.read_pickle(path)
        else:
            raise ReadError(f'The extention of the file [{ext}] is not supported. Support list: [.pkl, ] .')

        self.proxydb = ProxyDatabase().from_df(df, **kwargs)
        if verbose:
            p_success(f'{self.proxydb.nrec} records loaded')
            p_success(f'job.proxydb created')

    def filter_proxydb(self, *args, verbose=False, **kwargs):
        self.proxydb = self.proxydb.filter(*args, **kwargs)
        if verbose:
            p_success(f'{self.proxydb.nrec} records left')
            p_success(f'job.proxydb updated')

    def load_prior(self, path_dict=None, verbose=False):
        if path_dict is not None:
            self.configs.update({'prior_path': path_dict})
            if verbose: p_header(f'job.configs["prior_path"] = {path_dict}')
        else:
            path_dict = self.configs['prior_path']

        self.prior = {}
        for vn, path in path_dict.items():
            self.prior[vn] = ClimateField().load_nc(path)

        if verbose:
            p_success(f'prior variables {list(self.prior.keys())} loaded')
            p_success(f'job.prior created')