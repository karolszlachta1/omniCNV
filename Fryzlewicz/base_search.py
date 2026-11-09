#pure numpy module for the base search algorithm. 



import numpy as np

def haar_high(s,b,e):
    """
    This is the first term in the page 10 equation.
    """
    return np.sqrt(1/(b - s + 1) - 1/(e - s + 1))

def haar_low(s,b,e):
    """
    This is the second term in the page 10 equation.
    """
    return - np.sqrt(1/(e - b) - 1/(e - s + 1))

def basis_vector(s,b,e):
    """
    This creates a single basis vector for the haar wavelet, paramaterized by s,b,e and l.
    """
    high = haar_high(s,b,e)
    low = haar_low(s,b,e)
    array = np.zeros(e-s)
    array[0:b-s] = high
    array[b-s:] = low #should be b+1?
    return array

def haar_matrix(s,e):
    """
    This creates the matrix of basis vectors caused by iterating all possible break points.
    """
    matrix = np.zeros((e-s-1,e-s))
    for i, b in enumerate(range(s+1,e)):
        matrix[i,:] = basis_vector(s,b,e)
    return matrix

def haar_vectors(s,e):
    length = e - s    
    v_low = np.array([haar_low(s,b,e) for b in range(s+1,e)])
    v_high = np.array([haar_high(s,b,e) for b in range(s+1,e)])
    return v_low, v_high


def group_means(x, n):
        """
        Returns n mean values of input array grouped by len(x)/n.
        """
        assert n <= 2*len(x), "n must be less than or equal to len(x)/2."
                
        indexes = np.linspace (0, len(x), n+1, dtype=int)
        means = []
        for i,j in zip (indexes[0:-1], indexes[1:]):
            means.append(np.mean(x[i:j]))
        return np.array(means)


def _running_sum_excluding_fraction_numpy_cum(values, f):
    """
    This function calculates the running sum of the input array, excluding at most f fraction of the values that are equal to 0.5.
    """
    n = len (values)
    cumsum = np.cumsum (values)
    fractionHO = np.cumsum (values == 0.5)
    thrHO = np.floor (f * np.arange(1, n+1))
    corrHO = np.min (np.array([fractionHO, thrHO]), axis = 0)
    return cumsum - 0.5*corrHO


def choose_break(signal,s,e,p0 = .95, debug = False, fHO = 0):
    """
    This function chooses the best break point for the signal between s and e.
    It resolves ties by selecting the break point closest to the center of the signal.
    """
    if not (.5 <= p0 < 1):
        raise ValueError("p0 must be between [.5,1).")
    
    v_low, v_high = haar_vectors(s,e)

    if fHO > 1/(e-s):
        cumsignal = _running_sum_excluding_fraction_numpy_cum(signal[s:e], fHO)
        revcumsignal = _running_sum_excluding_fraction_numpy_cum(signal[s:e][::-1], fHO)[::-1]
    else:
        cumsignal = np.cumsum(signal[s:e])
        revcumsignal = np.cumsum(signal[s:e][::-1])[::-1]
    scores = cumsignal[:-1]*v_high + revcumsignal[:-1]*v_low
    
    trunc_scores = np.abs(scores[int((1-p0)*len(scores)):int(p0*len(scores))])
    best_options = np.argwhere(trunc_scores == np.nanmax(trunc_scores)).flatten() + 1 + s + int((1-p0)*len(scores))
    solution = best_options[np.abs(best_options - signal.size //2).argmin()]
    if debug:
        return scores, best_options, solution
    else:
        return solution
    

def create_basis_form(s,b,e):
    """
    This function creates the basis form of the haar wavelet.
    """
    high = haar_high(s,b,e)
    low = haar_low(s,b,e)
    return [0,e-s, s,(high, b-s),(low,e-b)]

    
def generate_haar_basis (signal, p0 = .95, length = 20, fHO = 0):
    """
    This function generates the haar basis for the signal.
    """
    if callable(length):
        d = int(length(signal.size))
    elif isinstance(length, int):
        d = length
    else:
        raise ValueError("Depth must be an integer or a function.")
    
    if not (.5 <= p0 < 1):
        raise ValueError("p0 must be in [.5,1).")
    
    d = max(d,2)
    s = 0
    e = signal.size
    
    done = [[0,len(signal),0,(signal.mean(),len(signal)),(0,0)]]
    todo = [(0, s,e)]
    while len(todo) > 0:
        depth, s,e = todo.pop(0)
        if e - s >= d:
            break_point = choose_break(signal, s, e, p0, fHO = fHO)
            if break_point == s or break_point == e:
                break_point = (s+e)//2
            solution = create_basis_form(s,break_point,e)
            solution[0] = depth
            if solution not in done:
                done.append(solution)
                todo.append((depth+1, s,break_point))
                todo.append((depth+1, break_point,e))
    #TODO: configurable size threshold
    return [done[0]] + [done[i] for i in np.where([(d[3][1] > 10)&((d[4][1] > 10))  for d in done])[0]]

