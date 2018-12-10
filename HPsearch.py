#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Sequence
from collections.abc import Mapping
from collections import OrderedDict
import numpy as np
import json
import os
import time

class hp_search_base(object):
    
    def __init__(self, experiment_name='experiment', log_folder = None, search_method='random', seed=66, fixed_params_class_attr_name = 'fixed_params', **kwargs):
        """
        Args:
            experiment_name (str): uses this to name the results file
            log_folder (str): the folder of the experiment
            search_method (str): One of 'random', 'grid_search'
            seed (int): random seed used for 'random' rearch method
            fixed_params_class_attr_name (str): Name of the class attribute for fixed params
            
        Returns:
            
        """
        if not search_method in ['random', 'grid_search']:
            raise ValueError("Search method {} is not supported".format(search_method) )
        
        if log_folder is None:
            raise ValueError("log_folder is a compulsory parameter")
        
        self.search_method = search_method
        self.seed = seed # required only for random search method
        self.experiment_name = experiment_name
        self.results_file_postfix = '_results.txt' # The experiment result file name is: self.experiment_name + self.results_file_postfix
        self.log_folder = os.path.abspath(log_folder)
        self.fixed_par_name = fixed_params_class_attr_name
        
        fixed_params = self.__class__.__dict__.get(self.fixed_par_name, None)
        if fixed_params is not None:
            if not isinstance(fixed_params, Mapping):
                raise ValueError("Fixed params are not of Mapping type: {}".format(fixed_params) )
            self.fixed_params = fixed_params
            print("Fixed params: {}".format(fixed_params) )
        else:
            self.fixed_params = {}
            print("Fixed params: None" )
        
        for pn, pv in kwargs.items():
            print("Extra argument:   {}: {}".format(pn,pv) )
            setattr(self,pn,pv)
            
        print("Experiment: {}     will log into: {}".format(self.experiment_name, self.log_folder) )
        
    @classmethod
    def _get_hp_values(cls,):
        """
        This method finds values of hps in the class definition and their order (if it is needed for 'grid_search' method).
        """
        
        hp_names = []
        hp_values = []
        hp_order = []
        for key in cls.__dict__.keys():
            if key.startswith('hp_'):
                val = cls.__dict__[key]
                real_key = key[3:]
                #import pdb;pdb.set_trace()
                if isinstance(val, Mapping):
                    hp_values.append( val['val'] )
                    hp_order.append( ( real_key,val['ord']) )
                if isinstance(val,Sequence):
                    hp_values.append( val )
                hp_names.append( real_key )
                
        return hp_names, hp_values, hp_order
    
    @staticmethod
    def params_values_list_random(seed, hp_names, hp_values, hp_order):
        """
        Return the list of hyperparameter values for random rearch method.
        """
        import random
        from sklearn.model_selection import ParameterGrid
        
        #import pdb;pdb.set_trace()
        params_dict = { vv[0]: vv[1] for vv in zip(hp_names, hp_values) }
        
        params_list = list(ParameterGrid(params_dict))
        
        random.seed(seed)
        random.shuffle(params_list)
        
        return params_list
    
    @staticmethod
    def params_values_list_grid(hp_names, hp_values, hp_order):
        
        from itertools import product
        
        if len(hp_order) < len(hp_names):
            raise ValueError("""Order is not defined for all hyperparameters. 
                Hyperparameters: {}
                Hyperparameters order: {}""".format(hp_names, hp_order) )
            
        sorted_inds = np.argsort([ i[1] for i in hp_order])

        hp_names = [hp_names[i] for i in sorted_inds]
        hp_values = [hp_values[i] for i in sorted_inds]
        #hp_order = [hp_order[i] for i in sorted_inds]
        
        prod = list( product(*hp_values) )
        out = [ { hp_names[i]: val for (i,val) in enumerate(vv) } for vv in prod ]
        
        return out
    
    def get_iter_params(self,):
        """
        Outputs the list with all values of hyperparameters.
        """
        
        hp_names, hp_values, hp_order = self._get_hp_values()
        if self.search_method == 'random':
            params_list = self.params_values_list_random(self.seed, hp_names, hp_values, hp_order)
            
        elif self.search_method == 'grid_search':
            params_list = self.params_values_list_grid(hp_names, hp_values, hp_order)
            
        return params_list
    
    def before_run(self, **kwargs):
        pass
        
    def run(self, parallel=False, max_iter=None, **before_run_params):
        """
        Args:   
            parallel (bool): Not implemented yet.
            max_iter (int): How many evaluations are done now.
        Returns:
            None, outputs to the file.
        """
        
        params_list = self.get_iter_params()
        
        # Run 'before_run' ->
        self.before_run(**before_run_params)
        # Run 'before_run' <-
        
        # Folder and log file ->
        if not os.path.isdir(self.log_folder): # folder does not exist
            os.mkdir(self.log_folder)
            
        results_file_path = os.path.join(self.log_folder, self.experiment_name + self.results_file_postfix )
        
        start_ind = self._read_last_evaluation_ind(results_file_path)
        # Folder and log file <-
        
        
        print("Start ind: {}".format(start_ind) )
        run_ind = start_ind
        if parallel == False:
            for params_vals in params_list[start_ind:]:
                print("Run: {},  param values: {}".format(run_ind, params_vals), end =" " )
                tt = time.time()
                params_to_train = {**params_vals, **self.fixed_params}
                
                res = self.train(**params_to_train)
                
                print("time: {}".format(time.time() - tt) )
                self._save_evaluation(results_file_path, run_ind, params_vals, self.fixed_params, res)
                run_ind += 1
                if max_iter is not None:
                    if (run_ind - start_ind) >= max_iter:
                        print("Maximum number of iterations reached: {}".format(max_iter) )
                        return 
        else:
            pass
        print("Experiment finished")
        
    @staticmethod
    def _save_evaluation(results_file_path, run_ind, params_vals, fixed_params, res):
        
        #import pdb; pdb.set_trace()
        if isinstance(res, Mapping):
            output = {'ind': run_ind, 'params': params_vals, 'fixed_params': fixed_params,'results': res }
        elif isinstance(res, Sequence):
            output = {'ind': run_ind, 'params': params_vals, 'fixed_params': fixed_params, 'results': { i:r for i,r in enumerate(res) } }
        else: # assume res is 1 value
            output = {'ind': run_ind, 'params': params_vals, 'fixed_params': fixed_params, 'results': { 0:res } }
            
        results_json = json.dumps( output )
        
        with open(results_file_path,"a") as ff:
            ff.write(results_json + '\n')
            ff.flush()
            
    @staticmethod
    def _read_last_evaluation_ind(results_file_path):
        
        
        if os.path.isfile(results_file_path):
            with open(results_file_path, 'r') as ff:
                exp_data_txt = ff.readlines()
            
            exp_data = []
            for ll in exp_data_txt:
                exp_data.append( json.loads(ll) )
            
            max_ind = np.max( [dd['ind'] for dd in exp_data] ) + 1
            return  int(max_ind)
        else:
            return 0
    
    @staticmethod
    def look_at_hyperparameters_search(file_path, result_names=('mse',), higher_better=(False,), output_top_n=(10,)):
        """

        """

        file_path = os.path.abspath(file_path)

        #filenames_all = glob.glob(folder, recursive=False)
        #import pdb; pdb.set_trace()
        experiments = []
        #for fn in filenames_all:
        with open(file_path, 'r') as ff:
            output = ff.readlines()
        
        for exp_data_txt in output:
            exp_data = json.loads(exp_data_txt)
            experiments.append(exp_data)

        if not isinstance(result_names, Sequence):
            result_names = (result_names,)
            
        if not isinstance(higher_better, Sequence):
            higher_better = (higher_better,)
        
        if not isinstance(output_top_n, Sequence):
            output_top_n = (output_top_n,)
        
        #import pdb; pdb.set_trace()
        
        #{'ind': run_ind, 'params': params_vals, 'results': res }
        
        sorted_dict = OrderedDict()
        for (ii,rn) in enumerate(result_names):
            arg_sorted = np.argsort( [ exp['results'][rn] for exp in experiments ] )
            if higher_better[ii]:
                arg_sorted = np.flip( arg_sorted )
            arg_sorted = arg_sorted[0: output_top_n[ii] ]
            sorted_dict[rn] = arg_sorted
            

        # Output:
        for rn, sinds in sorted_dict.items():
            print(rn)
            print("========================================================")
            for ii in sinds:
                print(experiments[ii]['results'][rn], "ind: ", ii, ",   params: ", experiments[ii]['params'], 
                      ",  other results:  ", experiments[ii]['results'])
            
    def clean_experiment_folder(self,p_clean=True):
        """
        Cleans the experiment folder.
        
        Args:
            p_clean (str): if true - clear the folder if false - do nothing
        """
        
        path = self.log_folder
        if p_clean:
            for the_file in os.listdir(path):
                file_path = os.path.join(path, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path): 
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
    