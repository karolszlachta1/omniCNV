#from tokenize import group
import numpy as np
import scipy.optimize as opt
import sys as sys


def distance (ai, uai, cn, ucn, tai, tcn):
    """Calculate the combined normalized distance from a mathematical model
    to observed data.

    ``ai`` and ``cn`` are observed allele imbalance and copy number, while
    ``tai`` and ``tcn`` are model predictions.  ``uai`` and ``ucn`` are their
    uncertainties; array inputs are supported.
    """
    a = (cn - tcn) / (ucn)
    b = (ai - tai) / (uai)
    c = np.sqrt(a**2 + b**2)
    
    return np.abs (a*b)/c 

all_models = []
all_models.append ({'label' : 'AB',
                    'm' : lambda ai, cn : np.repeat(2, len(cn)),
                    'ai' : lambda ai, cn : np.repeat(0, len(cn)),
                    'd' : lambda ai, uai, cn, ucn: np.sqrt (((2-cn)/(ucn))**2 + ((0-ai)/(uai))**2),
                    'k' : lambda ai : np.nan}) #+10**-16
all_models.append ({'label' : 'A',
                    'm' : lambda ai, cn : 1/(0.5+ai),
                    'ai' : lambda ai, cn  : 0.5-(cn-1)/cn,
                    'd' : lambda ai, uai, cn, ucn: distance (ai, uai, cn, ucn, 
                                                             0.5-(cn-1)/cn,
                                                             1/(0.5+ai)),
                    'k' : lambda ai : 4*ai/(1+2*ai)}) #+10**-16
all_models.append ({'label' : 'AA',
                    'm' : lambda ai, cn : np.repeat(2, len(cn)),
                    'ai' : lambda ai, cn : ai,
                    'd' : lambda ai, uai, cn, ucn: np.abs (2-cn)/(ucn),
                    'k' : lambda ai : 2*ai }) #+10**-16
all_models.append ({'label' : 'AAB',
                    'm' : lambda ai, cn : 1/(0.5-ai),
                    'ai' : lambda ai, cn : 0.5-1/cn,
                    'd' : lambda ai, uai, cn, ucn: distance (ai, uai, cn, ucn, 
                                                             0.5-1/cn,
                                                             1/(0.50001-ai)),
                    'k' : lambda ai : 4*ai/(1-2*ai+10**-16)}) #+10**-16
all_models.append ({'label' : 'AAA',
                    'm' : lambda ai, cn : 6/(3-2*ai),
                    'ai' : lambda ai, cn : (3*cn-6)/(2*cn),
                    'd' : lambda ai, uai, cn, ucn: distance (ai, uai, cn, ucn, 
                                                             (3*cn-6)/(2*cn),
                                                             6/(3-2*ai)),
                    'k' : lambda ai : (4*ai)/ (3 - 2*ai + 10**-16)}) #+10**-16
all_models.append ({'label' : 'AABB',
                    'm' : lambda ai, cn : cn,
                    'ai' : lambda ai : 0.0})


class Model:
    """
    Model class for karyotyping optimisation.
    Holds the selected models and provides utility functions.
    """
    def __init__(self, models_labels, config , logger):
        """Initialize modeling."""
        
        self.config = config
        self.logger = logger.getChild(f'{self.__class__.__name__}')
        self.models = []
        self.models_labels = models_labels
        self.all_models_labels = [m['label'] for m in all_models]

        for m in models_labels:
            i = np.where(np.array(self.all_models_labels) == m)[0][0]
            self.models.append(all_models[i])       
        

    def get_m0_range (self, balanced, imbalanced, min_dif):
        """Generate diploid coverage ranges for optimization.

        Balanced-segment coverage gaps are grouped using MAD outlier
        detection; an additional candidate may be derived from the lowest
        imbalanced segment.  Each returned candidate is ordered as
        ``(minimum, lower, upper, maximum)``.
        """
        #first check if there is a clear gap in the balanced segments        
        res = []
       
        cv = (np.sort(balanced['cv'].to_numpy(dtype = float)))
        if len (cv) > 1:
            d = cv[1:]-cv[:-1]
            if any (d > min_dif):
                self.logger.debug ("Checking balanced segments for compactness.")
                mask, m, s, sc = MADoutlier (d, thr = float(self.config["KARYOTYPING"]["DUP_MAD_thr"]),#2,
                                             cov_std = np.sqrt(balanced['dcv'].median()**2/4 + min_dif**2))
                self.logger.debug (f"Distances analysis: m = {m}, s = {s}, sc = {sc}.")
                self.logger.info (f"Measurement uncertainty for grouping {balanced['dcv'].median()/2}")
                self.logger.debug (f"Mask: {mask}")
                if any (mask != 'H'):
                    self.logger.warning ("Balanced segments seems not compact.")
                    group = np.array([0] + list(np.cumsum(mask == 'H')))
                    self.logger.debug (f"Group labels for balanced segments: {group}")
                    mval, mcts = rle_encode (group)
                    self.logger.debug (f"Groups: {mval}, counts: {mcts}")
                    start = 0
                    for i,m in enumerate (mval):
                        ccv = cv[start:start+mcts[i]]
                        self.logger.debug (f"Adding group {ccv} to m0s.")
                        if len (ccv) >= 4:
                            res.append (np.quantile (ccv, q = [0,0.25,0.75,1]))
                        elif len (ccv) > 1:
                            #mtmp = np.median (ccv)
                            res.append (np.array ([ccv.min(), ccv.min(), ccv.max(), ccv.max()]))
                        else:
                            res.append (np.array ([ccv[0] - min_dif/2, ccv[0] - min_dif/2, 
                                                   ccv[0] + min_dif/2, ccv[0] + min_dif/2]))
                        self.logger.info (f"m0 range added: {res[-1]}")
                        start += mcts[i]

                else:
                    self.logger.info ("Balanced segments seems compact. No outliers detected.")
                    res.append (np.quantile (cv, q = [0,0.25,0.75,1]))
                    self.logger.debug (f"m0 range added: {res[-1]}")
            else:
                self.logger.debug ("Balanced segments seems compact. Not passing min required difference.")
                res.append (np.quantile (cv, q = [0,0.25,0.75,1]))
                self.logger.debug (f"m0 range added: {res[-1]}")
        else:
            self.logger.debug ("Only one balanced segment.")
            res.append (np.array([0.75,0.75,1.25,1.25])*cv)
            self.logger.debug (f"m0 range added: {res[-1]}")

        #TODO: needs a fix. 
        if len (imbalanced):
            m = imbalanced['cv'].min()
            ai = imbalanced.loc[imbalanced['cv'] == m, 'ai'].values[0]
            m0 = 2*m*(0.5+ai)
            #m = imbalanced['cv'].min()
            if all([m0 < r[0] for r in res]):
                self.logger.info ("Imbalanced segments based m0 added.")
                res.append ([0.95*m0, 0.95*m0, 1.05*m0, 1.0*m0])
                self.logger.debug (f"m0 range added: {res[-1]}")
        return res


    def _distance (self, scales, m0, ai, uai, cv, ucv, weight, return_labels = False):
        """Score selected models by weighted squared distance and optionally return their labels.

        Copy number is calculated as ``2 + scale * (cv - m0)``.  With
        ``return_labels=False`` this returns weighted squared distances; with
        it enabled, it returns ``(labels, per_model_distances)``.
        """
        scale = scales[0]
        cn = 2 + scale*(cv - m0)
        ucn = scale*ucv
        
        if (ai >= 0.5).any():
            ai = ai.copy()
            ai[ai >= 0.5] = 0.5 - 10**-16

        distance = np.array([m['d'](ai, uai, cn, ucn) for m in self.models])
        min_index = np.nanargmin (distance, axis = 0)
        
        try:
            total_distance = ((distance[min_index,np.arange (distance.shape[1])]*weight/np.sum(weight))**2)
            total_distance[total_distance == 0] = 10**-16
        except IndexError:
            exit (1)
        if return_labels:
            return [self.models_labels[i] for i in min_index], distance
        else:
            return total_distance


    def get_clonality (self, model, ai, sai, cn, scn):
        """Estimate clonality and assess whether a segment fits a model.

        Returns ``(k, good)``, where ``k`` is the model-specific clonality
        estimate and ``good`` indicates agreement within two uncertainty units.
        """
        which_model = np.where ([m['label'] == model for m in self.models])[0][0]
        model_fun = self.models[which_model]
        at = model_fun['ai'](ai, cn)
        ct = model_fun['m'](ai, cn)

        #ai, uai, cn, ucn, tai, tcn
        def dist (a):
            """Return distance for a candidate allele-imbalance value."""
            return distance (ai, sai, cn, scn, a, model_fun['m'](a, cn))

        res = opt.minimize_scalar (dist, bounds = np.sort (ai, at), method = 'bounded')

        if (np.abs(at - ai) > 2*sai) | (np.abs(ct - cn) > 2*scn):
            good = False
        else:
            good = True

        return model_fun['k'](res.x), good

def MADoutlier(values, thr = 3, cov_std = 0):
    """
    Identifies outliers in a given array of values using the Median Absolute Deviation (MAD) method.

    Parameters:
    values (array-like): The input array of values.
    thr (float, optional): The threshold value for determining outliers. Defaults to 3.

    Returns:
    tuple: A tuple containing the following elements:
        - mask (array-like): An array of the same length as `values` indicating the outlier status of each value.
            'C' represents a core value, 'L' represents a low outlier, and 'H' represents a high outlier.
        - m (float): The mean of the core values.
        - s (float): The standard deviation of the core values.
        - s/c (float): The corrected standard deviation of the core values.

    Raises:
    None.

    Examples:
    >>> values = [1, 2, 3, 4, 5, 100]
    >>> mask, m, s, s_c = MADoutlier(values)
    >>> print(mask)
    ['C', 'C', 'C', 'C', 'C', 'H']
    >>> print(m)
    3.0
    >>> print(s)
    1.4142135623730951
    >>> print(s_c)
    1.4142135623730951

    """
    
    assert np.isfinite(values).all(), 'Input values must be numeric.'  # Check if values are numeric
    median = np.median(values)
    diff = np.abs((values - median))
    med_abs_deviation = np.median(diff)
    ## Important to ensure no division by zero
    ## In case of good WGS data mostly
    if med_abs_deviation == 0:
        med_abs_deviation = cov_std  
    
    modified_z_score = (0.6745 * diff / med_abs_deviation)
    coreMask = modified_z_score < thr
    m = np.mean (values[coreMask])
    s = np.std (values[coreMask])
    n = len (values)
    try:
        c = 1 - 1/(4*n) - 7/(32*n**2) - 19/(128*n**3)
    except (ZeroDivisionError):
        c = 1

    mask = np.repeat ('C', len(values))
    mask[(diff >= cov_std)&(modified_z_score >= thr)&(values < m)] = 'L'
    mask[(diff >= cov_std)&(modified_z_score >= thr)&(values > m)] = 'H'

    return mask, m, s, s/c


def rle_encode (string):
    """RLE encoder for a one-dimensional sequence."""
    i = 0
    values = []
    counts = []
    while i < len(string):
        cur_char = string[i]
        count = 1
        while (i < len(string) - 1 and string[i] == string[i+1]):
            count += 1
            i += 1
        values.append (cur_char)
        counts.append (count)
        i += 1
    return np.array(values), np.array(counts)

