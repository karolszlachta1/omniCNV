#from unittest import result
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as mgs
import numpy as np
import scipy.stats as sts
import scipy.optimize as opt

#from omniCNV import Model as Model
from collections import defaultdict

clonecolor = defaultdict (lambda: 'gray')
clonecolor.update({'A' : 'tab:blue',
                   'LAB' : 'tab:cyan',
                   'AB' : 'tab:green', 
                   'AABB' : 'black',
                   'AAAB' : 'tab:purple',
                   'AA' : 'tab:brown',
                   'AAB' : 'tab:red', 
                   'AAA' : 'tab:orange',
                   'CRAZY' : 'tab:pink',
                   '-' : 'gray'})


class Solution:
    """
    Solution class for specific karyotyping optimization.
    
    Parameters
    ----------
    mbounds : tuple  Initial bounds for m0.
    init_scale : float  Initial scale factor.
    balanced : pd.DataFrame  Balanced segments.
    balanced_small : pd.DataFrame  Small balanced segments.
    imbalanced : pd.DataFrame  Imbalanced segments.
    imbalanced_small : pd.DataFrame  Small imbalanced segments.
    coverage_type : str  Type of coverage ('ai' or 'l2r').
    models : Model  Model object containing karyotyping models.
    config : dict  Configuration dictionary.
    logger : logging.Logger  Logger object.
    """
    def __init__(self, mbounds, init_scale, 
                 balanced, balanced_small,
                 imbalanced, imbalanced_small,
                 coverage_type, models, config, logger):
        """Initialize and optimize one candidate karyotype solution.

        Segment inputs are copied before optimization.  The resulting
        combined table is available through :meth:`get_segments`.
        """
        
        self.mbounds = mbounds
        self.init_scale = init_scale
        self.balanced = balanced.copy()
        self.balanced_small = balanced_small.copy()    
        self.imbalanced = imbalanced.copy()
        self.imbalanced_small = imbalanced_small.copy()
        self.coverage_type = coverage_type
        self.models = models
        self.config = config
        self.logger = logger.getChild(f'{self.__class__.__name__}')
        self.logger.info (f'Initializing solution with m0: {self.mbounds}, scale: {self.init_scale}.')
        if len(self.imbalanced) == 0:
            self.logger.warning ('No imbalanced segments to optimize on!')
            self.m0 = np.mean(self.mbounds)
            self.scale = float (self.config['KARYOTYPING']['initial_scale'])
            self.distance = np.nan
            self.balanced ['cn'] = self.scale* (self.balanced['cv'] - self.m0) + 2
            self.balanced ['dcn'] = self.scale*self.balanced['dcv']
            self.balanced_small ['cn'] = self.scale* (self.balanced_small['cv'] - self.m0) + 2
            self.balanced_small ['dcn'] = self.scale*self.balanced_small['dcv']
            self.imbalanced_small ['cn'] = self.scale* (self.imbalanced_small['cv'] - self.m0) + 2
            self.imbalanced_small ['dcn'] = self.scale*self.imbalanced_small['dcv']

        else:
            self.optimize()
        
        self.calculate_k()

        self.logger.info (f'Optimized solution with m0: {self.m0}, scale: {self.scale}.')


    def calculate_k (self):
        """Calculate clonality fractions and model-consistency flags.

        Adds ``k`` and ``good`` columns to ``self.segments`` after the copy
        number and zygosity status have been assigned.
        """
        self.logger.info (f'Calculating k for segments.')
        self.segments = pd.concat ([self.balanced,
                                    self.imbalanced,
                                    self.imbalanced_small,
                                    self.balanced_small]).sort_values (by = ['arm', 'start'])

        # float column: model['k'](ai) and cn/2-1 are floats (int 0 made pandas warn)
        self.segments['k'] = 0.0
        self.segments['good'] = '-'

        for model in self.models.models:
            self.logger.debug (f'Calculating k for model {model["label"]}.')
            for i, seg in self.segments.loc[self.segments['zygosity'] == model["label"]].iterrows():
                ai = seg['ai']
                cn = seg['cn']
                at = model['ai'](np.array([ai]), np.array([cn]))[0]
                ct = model['m'](np.array([ai]), np.array([cn]))[0]
                self.segments.loc[i, 'k'] = model['k'](ai)
                if (np.abs(at - ai) > 2*seg['dai']) | (np.abs(ct - cn) > 2*seg['dcn']):
                    good = False
                else:
                    good = True

                self.segments.loc[i, 'good'] = 'Yes' if good else 'No'
        
        self.segments.loc[self.segments['zygosity'] == 'AABB', 'k'] = self.segments.loc[self.segments['zygosity'] == 'AABB',
                                                                                         'cn'].values/2 - 1
        self.segments.loc[self.segments['zygosity'] == 'AABB', 'good'] = 'Yes'
        

    def optimize (self):
        """Optimize diploid reference and scale, then calculate copy-number estimates.

        The scale bounds depend on ``coverage_type``.  Updated ``cn`` and
        ``dcn`` columns are written to each segment subset, and the objective
        value is stored in ``self.distance``.
        """
        def total_distance (params, balanced, imbalanced, small):
            """Return the weighted model distance for optimizer parameters."""
            m0, scale = params
            if len (balanced):
                ai = balanced['ai'].to_numpy(dtype = float)
                uai = balanced['dai'].to_numpy(dtype = float)
                cv = balanced['cv'].to_numpy(dtype = float).copy()
                ucv = scale*balanced['dcv'].to_numpy(dtype = float)
                weight = np.sqrt(balanced['size'].to_numpy(dtype = float))
                cv [cv == m0] += 10**-16  #avoid zero division
                balanced_d = np.nansum(np.log10(np.sqrt((ai/uai)**2 + (scale*(cv - m0)/ucv)**2)*weight/np.sum(weight))+1)
            else:
                balanced_d = 0

            ai = imbalanced['ai'].to_numpy(dtype = float)
            uai = imbalanced['dai'].to_numpy(dtype = float)
            cv = imbalanced['cv'].to_numpy(dtype = float).copy()
            ucv = imbalanced['dcv'].to_numpy(dtype = float)
            weight = np.sqrt(imbalanced['size'].to_numpy(dtype = float))
            imbalanced_d = np.nansum(np.log10(self.models._distance ([scale], m0, ai, uai, cv, ucv, weight))+1)
            
            models, dist = self.models._distance ([scale], m0, ai, uai, cv, ucv, weight, return_labels = True)
            imbalanced['zygosity'] =  models

            if self.imbalanced_small is not None:
                ai = self.imbalanced_small['ai'].values
                uai = self.imbalanced_small['dai'].values
                cv = self.imbalanced_small['cv'].values
                ucv = scale*self.imbalanced_small['dcv'].values
                weight = np.sqrt(self.imbalanced_small['size'].to_numpy(dtype = float))
                small['zygosity'], _ = self.models._distance ([scale], m0, ai, uai, cv, ucv,
                                                                weight, return_labels = True)
                
            return (imbalanced_d + balanced_d)/(len(imbalanced) + len(balanced))

        #First analyze scale 
        if self.coverage_type != 'wgs':
            result = opt.minimize (total_distance, 
                                   args = (self.balanced.loc[self.balanced['zygosity'] == 'AB'],
                                           self.imbalanced, self.imbalanced_small), 
                                   x0 = [np.mean(self.mbounds), self.init_scale],
                                   bounds = ((np.mean(self.mbounds)-0.001,
                                              np.mean(self.mbounds)+0.001),
                                             (self.init_scale-0.5, self.init_scale+0.5)),
                                   method = 'L-BFGS-B',
                                   tol = 1e-2)

            result = opt.minimize (total_distance, 
                                   args = (self.balanced.loc[self.balanced['zygosity'] == 'AB'],
                                           self.imbalanced, self.imbalanced_small), 
                                   x0 = [np.sum(self.mbounds)/2, result.x[0]],
                                   bounds = (self.mbounds, (self.init_scale-0.5, self.init_scale+0.5)),
                                   method = 'L-BFGS-B',
                                   tol = 1e-2)
            self.logger.debug(f"Kariotyping termination: {result.message}")

            self.m0 = result.x[0]
            self.scale = result.x[1]
            self.distance = result.fun
            self.logger.info(f'Optimization distance (d): {self.distance}')
        
        elif self.coverage_type == 'wgs':
            result = opt.minimize (total_distance, 
                                   args = (self.balanced.loc[self.balanced['zygosity'] == 'AB'],
                                           self.imbalanced, self.imbalanced_small), 
                                   x0 = [np.mean(self.mbounds), self.init_scale],
                                   bounds = (self.mbounds, (1.96, 2.04)),
                                   method = 'L-BFGS-B',
                                   tol = 1e-2)
            self.m0 = result.x[0]
            self.scale = result.x[1]
            self.distance = result.fun
            self.logger.info(f'Optimization distance (d): {self.distance}')
            
        self.balanced ['cn'] = self.scale* (self.balanced['cv'] - self.m0) + 2
        self.balanced ['dcn'] = self.scale*self.balanced['dcv']
        self.imbalanced ['cn'] = self.scale* (self.imbalanced['cv'] - self.m0) + 2
        self.imbalanced ['dcn'] = self.scale*self.imbalanced['dcv']
        self.balanced_small ['cn'] = self.scale* (self.balanced_small['cv'] - self.m0) + 2
        self.balanced_small ['dcn'] = self.scale*self.balanced_small['dcv']
        self.imbalanced_small ['cn'] = self.scale* (self.imbalanced_small['cv'] - self.m0) + 2
        self.imbalanced_small ['dcn'] = self.scale*self.imbalanced_small['dcv']


    def get_segments (self):
        """Return the combined, annotated segment table."""
        return self.segments


    def get_scale (self):
        """Return the optimized coverage-to-copy-number scale."""
        return self.scale
    

    def get_m0 (self):
        """Return the initial bounds used for the optimized ``m0`` value."""
        return self.mbounds
    

    def get_distance (self):
        """Return the optimization objective value for this solution."""
        return self.distance
