
import numpy as np
import scipy.optimize as opt


def haar_basis(l,base):
    res = np.zeros((len(base),l))
    for i, wavelet in enumerate(base):
        res[i,wavelet[2]:wavelet[2]+wavelet[3][1]+wavelet[4][1]] = generate_wavelet_function(wavelet)
    return res
    

def decompose (signal, base, threshold = None, k = 1.4826, ms = 2.0):
        
    basis = haar_basis(len(signal),base)
    coefficients = np.matmul(basis,signal)
    #normalized_coefficients = coefficients / coefficients.sum()
    if threshold is None:
        threshold = estimate_threshold (coefficients, k, ms)

    if np.isnan (threshold):
        threshold = 0.1

    drop = np.argwhere(np.abs(coefficients) <= threshold).flatten()
    all_coefficients = coefficients.copy()
    coefficients[drop] = 0.0
    coefficients[0] = 1.0
    y = generate_function_from_wavelets (coefficients, base,len(signal))
    
    return {'basis' : basis,
            'coefficients' : coefficients,
            'all_coefficients' : all_coefficients,
            'threshold' : threshold,
            'rle' : rle(y)}

def rle(s):
    counts = {}
    for c in s:
        if counts.get(c) is None:
            counts[c] = 1
        else:
            counts[c] = counts[c] + 1
    return [(k,counts[k]) for k in counts.keys()]



def estimate_threshold (X, k = 1.4826, ms = 2.0):
    d = np.abs(X[1:]-X[:-1])/np.sqrt(2)
    MAD = np.median(d)
    sigma = k*MAD
    thr = ms*sigma*np.sqrt(2*np.log(len(X)))
    return thr

def generate_wavelet_function (wavelet):
    """
    Generate a wavelet function from a wavelet.
    """
    wavelet_parts = [np.repeat (v, l) for v, l in wavelet[3:]]
    return np.concatenate (wavelet_parts)
 
def generate_function_from_wavelets (coefficients, base, l):
    """
    Generate a function from a set of wavelets.
    """
    wf = haar_basis (l, base) 
    return (wf* coefficients[ :, np.newaxis]).sum(axis = 0)
                              
def test_decompose ():
    SA = np.sqrt (1/(250 - 0) - 1/(1000 - 0 + 1))
    SB = np.sqrt (1/(1000 - 250) - 1/(1000 - 0 + 1))
    base = [[0,0,0, (10,1000),(0,0),(0,0)],
            [1,0,0, (SA,250), (SB, 750), (0,0)]]

    coefficients = [1, 10]

    res = decompose (generate_function_from_wavelets (coefficients, base), base)
    return res.x - np.array (coefficients)

def rle_to_array(rle):
    return np.concatenate([np.repeat(k,v) for k,v in rle])