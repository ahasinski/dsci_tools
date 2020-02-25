import numpy as np
import pandas as pd
import os
import json
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt import UtilityFunction
from sklearn.metrics import roc_auc_score
import xgboost as xgb

class BayesSearchCV(object):
    '''
    Performs a search for optimal hyperparameter
    values using a Gaussian Process.
    
    Usage Overview
    --------------
    __init__:
    - initializes empty dicts for:
      - bounds, cat_levs, param_types
    - stores paths for:
      - data (list of .npz files)
      - log path (.json)
      - report path (.csv file)
    - defines:
      - estimator (est)
      - scoring metric (metric_func)
    set_bounds:
    - defines lower & upper bounds for 
      all float and int params
      - cat params not required
    - sets param types--will code param
      as 'int' if both bounds are ints
    set_cat_levs:
    - creates int-string mapping for 
      cat params
      - float and int params not required
    - sets param types to 'cat'
    update_bounds:
    - 
    maximize:
    - if 1st iteration:
      - initializes optimizer
        - if log file exists, will load
          & incorporate into optimizer
      - initializes report
        - if report exists, will load
          this file as report object
    - performs maximize method for 
      specified number of iterations
    get_report:
    - outputs report in specified format
    write_report:
    - writes report to spec
    '''
    def __init__(self, 
                 est, 
                 train_test_paths,
                 metric_func,
                 log_path,
                 report_path=None,
                 verbose=2, 
                 random_state=None, 
                 **fixed_params):
        self.verbose     = verbose
        self.random_state= random_state
        self.bounds      = {}
        self.cat_levs    = {}
        self.param_types = {}
        self.optimizer   = None
        self.report      = None
        self.est         = est
        self.fixed_params= fixed_params
        self.metric_func = metric_func
        if log_path[-5:]!= '.json':
            log_path    += '.json'
        self.log_path    = log_path
        self.report_path = report_path
        self.set_train_test(train_test_paths)
        
    def _init_optimizer(self):
        '''
        Initializes optimizer.
        '''
        self.optimizer = BayesianOptimization(
            f=self._validator_helper,
            pbounds=self.bounds,
            verbose=self.verbose,
            random_state=self.random_state)
        if os.path.exists(self.log_path):
            # New optimizer is loaded with previously seen points
            load_logs(self.optimizer, logs=[self.log_path])   
        self.logger = JSONLogger(path=self.log_path)
        self.optimizer.subscribe(Events.OPTMIZATION_STEP, 
                                 self.logger)

    def _init_report(self):
        '''
        Initializes report object.
        '''
        new_report = False
        if self.report_path is not None:
            if self.report_path[-4:]!= '.csv':
                self.report_path    += '.csv'
            if os.path.exists(self.report_path):
                self.report = pd.read_csv(self.report_path)\
                                .to_dict(orient='list')
            else:
                new_report = True
        else:
            new_report = True
        if new_report:
            self.report = {p:[] for p in self.bounds.keys()}
            self.report.update({p:[] for p in self.fixed_params.keys()})
            self.report['trn_score'] = []
            self.report['tst_score'] = []

    def get_report(self, form='df'):
        '''
        Outputs report object in specified form:
          - 'df':  pd.Dataframe (default)
          - 'arr': np.array
          - otherwise: dict
        '''
        if form=='df':
            return pd.DataFrame(self.report)
        elif form=='arr':
            return pd.DataFrame(self.report).values
        else:
            return self.report
        
    def write_report(self, report_path=None):
        if report_path is None:
            if self.report_path is None:
                print('Must provide report_path.')
            else:
                report_path = self.report_path
        if report_path[-4:]!= '.csv':
            report_path    += '.csv'
        df = pd.DataFrame(self.report)
        df.to_csv(report_path, index=False)
            
    def load_param_meta(self, meta_path=None):
        if meta_path is None:
            if self.meta_path is None:
                print('Must provide meta_path.')
            else:
                meta_path = self.meta_path    
        with open(meta_path, 'r') as mp:
            meta_data = json.load(mp)
            self.bounds       = meta_data['bounds']
            self.cat_levs     = meta_data['cat_levs']
            self.fixed_params = meta_data['fixed_params']
    
    def write_param_meta(self, meta_path=None):
        if meta_path is None:
            if self.meta_path is None:
                print('Must provide meta_path.')
            else:
                meta_path = self.meta_path
        if meta_path[-5:]!= '.json':
            meta_path    += '.json'        
        with open(meta_path, 'w') as mp:
            meta_data = {'bounds':       self.bounds,
                         'cat_levs':     self.cat_levs,
                         'fixed_params': self.fixed_params}
            json.dump(meta_data, mp)            
            
    def set_train_test(self, train_test_paths):
        '''
        Stores train_test_paths.
        If only 1 path is provided, it loads the data now.
        Otherwise each fold will be loaded in turn on each
        iteration of maximize to conserve memory.
        '''
        if (type(train_test_paths) is str):
            train_test_paths = [train_test_paths]
        if len(train_test_paths)==1:
            self.X_trn,\
            self.y_trn,\
            self.X_tst,\
            self.y_tst = self.get_train_test(train_test_paths[0])
            self.data_loaded = True
        else:
            self.data_loaded = False
        self.train_test_paths = train_test_paths

    def get_train_test(self, train_test_path):
        '''
        Loads train/test X/y arrays from npz file.
        Assumes arrays in npz adhere to name convention:
          - X_trn, X_tst, y_trn, y_tst
        '''
        with np.load(train_test_path) as data_dct: 
            X_trn = data_dct['X_trn'].astype(np.float32)
            X_tst = data_dct['X_tst'].astype(np.float32)
            y_trn = data_dct['y_trn']
            y_tst = data_dct['y_tst']
            #fold_idx = data_dct['fold_idx']
        return (X_trn, y_trn, X_tst, y_tst)
            
    def set_param_types(self, **params):
        '''
        Fills dict that stores each param's type:
          - 'float', 'int', or 'cat'
        '''
        for k,v in params.items():
            self.param_types[k] = v
            
    def set_cat_levs(self, **params):
        '''
        Provide each categorical variable, each with 
        a corresponding list of string values. 
        '''
        for k,v in params.items():
            cat_levs[k] = v
            self.set_bounds(**{k:[0,len(v)]})
            self.set_param_types(**{k:'cat'})
            self.cat_levs[k] = v
        
    def set_bounds(self, **params):
        '''
        Must provide lower/upper bounds for each param.
        If param is to be treated as integer, BOTH bounds
        should be int type to set param type as int. Also
        upper bound should be 1+ desired upper bound.
        If param is categorical, pass levels to 
        `set_cat_levs` method instead.
        '''
        for k,v in params.items():
            self.bounds[k] = v
            if (type(v[0]) is int) and\
               (type(v[1]) is int):
                self.set_param_types(**{k:'int'})
            else:
                self.set_param_types(**{k:'float'})
    
    def update_bounds(self, **params):
        '''
        Update params
        '''
        self.optimizer.set_bounds(new_bounds=params)
    
    def _validator(self, **params):
        '''
        Gets validation score for passed set of params,
        given pre-sets:
          - estimator type
          - train/test data
          - scoring metric
        '''
        trn_scores = []
        tst_scores = []
        metric_func = self.metric_func
        # initialize estimator
        params.update(self.fixed_params)
        est = self.est(**params)
        # loop over folds
        for train_test in self.train_test_paths:
            # ready data
            if self.data_loaded:
                X_trn = self.X_trn
                y_trn = self.y_trn
                X_tst = self.X_tst
                y_tst = self.y_tst
            else:
                X_trn,\
                y_trn,\
                X_tst,\
                y_tst = get_train_test(train_test)
            # fit & evaluate estimator
            est.fit(X_trn, y_trn)
            trn_pred = est.predict_proba(X_trn)
            trn_score= metric_func(y_trn, trn_pred[:,1])
            tst_pred = est.predict_proba(X_tst)
            tst_score= metric_func(y_tst, tst_pred[:,1])
            # 
            trn_scores.append(trn_score)
            tst_scores.append(tst_score)
        # record parameters and metrics
        trn_scores_avg = np.mean(trn_scores)
        tst_scores_avg = np.mean(tst_scores)
        self.report['trn_score'].append(trn_scores_avg)
        self.report['tst_score'].append(tst_scores_avg)
        for k,v in params.items():
            self.report[k].append(v)        
        return tst_scores_avg
        
        
    def _validator_helper(self, **params):
        '''
        Converts real-valued parameter values to integers
        or string categories based on stored param types.
        Then passes fixed params to validator function.
        '''
        #params = self.params
        for k,v in params.items():
            if   self.param_types[k]=='int':
                params[k] = int(v)
            elif self.param_types[k]=='cat':
                params[k] = self.cat_levs[k][int(v)]
        return self._validator(**params)
                
        
    def maximize(self, init_points=2, n_iter=3, **kwargs):
        '''
        Wrapper for optimizer's maximize method.
        '''
        if self.optimizer is None:
            self._init_optimizer()
        if self.report is None:
            self._init_report()
        self.optimizer.maximize(init_points=init_points,
                                n_iter=n_iter,
                                **kwargs)
    


