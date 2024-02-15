from pathlib import Path
import argparse
import pprint
import sys
import yaml

import json
import shutil

from typing import Callable

def read_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")
            
        return data
    
class Runner():
    """
    Runs one run of an experiment. It has a technical role: importing modules, saving data etc.
    """
    
    def __init__(self, run_func: Callable, runner_config_section='runner', **run_params):
        """
        
        run_func (Callable): function which actually run experiment.
        runner_config_section (str): key in `run_params` which contains parameters of runner.
        run_params (dict{dict}): hierarchical params of experiment. One section `runner` is related to
            this runner class. The rest to experiment itself and must be passed there.
        """
        
        self.runner_params = run_params.get(runner_config_section)
        self.run_func = run_func
        
        self.current_file = Path(__file__).absolute()
        self.current_folder = self.current_file.parent
        self.current_name = self.current_file.name
        
        self.config_filename = 'config.yaml'
        self.results_filename = 'results.json'
        # ----------------------- import code ----------------------->
        
        # ----------------------- import code -----------------------<
        
        # --------------------------------------------- Save Paths --------------------------------------------->-

        try:
            self.run_folder = Path(self.runner_params['folder']).absolute()
        except Exception as ee:
            self.run_folder = self.current_folder
        print(f'Run folder:  {self.run_folder}')

        #import pdb; pdb.set_trace()
        # --------------------------------------------- Save Paths ---------------------------------------------<-
    
    def run(self, **run_params):
        
        result_dict = self.run_func(**run_params) #self.run_func(**self.runner_params)
        
        # ------------------------------------------ Output Results ------------------------------------------>
        if str(self.current_folder) != str(self.run_folder): # need to create new run folder.
            self.run_folder.mkdir(parents=True, exist_ok=True)

            # save config to run folder:
            with open(self.run_folder / self.config_filename, 'w') as yaml_file:
                yaml.dump(run_params, yaml_file, default_flow_style=False, indent=2)

            # copy run file to the folder:
            shutil.copy(self.current_file, self.run_folder / self.current_name)

        # save results to run folder:
        with open(self.run_folder / self.results_filename, 'w') as file:
            json.dump(result_dict, file, indent=2)
        # ------------------------------------------ Output Results ------------------------------------------<
        
        
        
## ----------------------------------------Example of using Runner ------------------------------------------>
#if __name__ == '__main__':
#    
#    parser = argparse.ArgumentParser(description='Command line params')
#    
#    parser.add_argument('--config', type=str, default=None, help='Config file')
#    args = parser.parse_args()
#    
#    hyper_params = read_yaml(args.config)
#    
#    runner_config_section='runner'
#    run_func_param = hyper_params[runner_config_section]['run_func']
#    
#    if 'ms' in run_func_param:
#        run_func = ms_run
#    elif 'dnb' in run_func_param:
#        run_func = dnb_run
#    
#    runner = Runner( run_func=run_func, runner_config_section='runner', **hyper_params )
#    runner.run(**hyper_params)
## ----------------------------------------Example of using Runner ------------------------------------------<