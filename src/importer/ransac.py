import cPickle
import copy
import numpy as np
import scipy # use np if scipy unavailable
import scipy.linalg # use np if scipy unavailable
import scipy.optimize
import matplotlib.pyplot as plt

from sets import Set

def ransac(data,model,n,k,t,d,debug=False,return_all=False):
    """fit model parameters to data using the RANSAC algorithm
    
This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

{{{
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
"""
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs,:]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error( test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if debug:
            print 'test_err.min()',test_err.min()
            print 'test_err.max()',test_err.max()
            print 'np.mean(test_err)',np.mean(test_err)
            print 'iteration %d:len(alsoinliers) = %d'%(
                iterations,len(alsoinliers))
        if len(alsoinliers) > d:
            betterdata = np.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = np.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
        iterations+=1
    if bestfit is None:
        raise ValueError("did not meet fit acceptance criteria")
    if return_all:
        test_err = model.get_error(data, bestfit)
        test_idxs = np.arange(data.shape[0])
        also_idxs = test_idxs[test_err < t]
        return bestfit, {'inliers':also_idxs}
    else:
        return bestfit

def random_partition(n,n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.
    
    """
    def __init__(self, debug=False):
        self.debug = debug
    def fit(self, data):
        A = np.matrix(data[:,0])
        print A
        B = np.matrix(data[:,1])
        x,resids,rank,s = scipy.linalg.lstsq(A,B)
        return x
    def get_error( self, data, model):
        A = np.array(data[:,0])
        B = np.array(data[:,1])
        B_fit = scipy.dot(A,model)
        err_per_point = np.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point

class LinerCurveModel:
    """
        fit:
            x = a0*t + b0
            y = a1*t + b1
    """
    def __init__(self, debug=False):
        self.debug = debug
    
    def fit(self, data):
        ones = np.ones(len(data))
        A = np.vstack((data[:,0], data[:,1], ones)).T

        u, s, v = np.linalg.svd(A)
        
        return v[2,:]

    def get_error(self, data, model):
        error = []
        EPSILON = 1e-5
        for pt in data:
            x_target = 0.0
            y_target = 0.0
            if np.abs(model[0]) <= EPSILON:
                x_target = pt[0]
                y_target = -1 * model[2] / model[1]
            else:
                if np.abs(model[1]) <= EPSILON:
                    x_target = -1 * model[2] / model[0]
                    y_target = pt[1]
                else:
                    x_target = pt[0]
                    y_target = -1 * model[2] - model[0]*pt[0]
                    y_target /= model[1]

            tmp_error = np.abs(x_target-pt[0])*1e5 + np.abs(y_target-pt[1])*1e6
            error.append(tmp_error)
        err_per_point = np.array(error)
        return err_per_point

def f_lin(t, model):
    x = model[0][0]*t + model[0][1]
    y = model[1][0]*t + model[1][1]
    return x,y

class QuadraticLeastSquaresModel:
    """
        fit:
            x = a0 + b0*t + c0*t^2
            y = a1 + b1*t + c1*t^2
    """
    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        x = data[:, 0]
        sort_idxs = np.argsort(x)
        x = x[sort_idxs]
        y = data[:, 1]
        y = y[sort_idxs]
        #y = np.sort(y)
        t = np.linspace(0.0, 1.0, len(x))
        z_x = np.polyfit(t, x, 2)
        z_y = np.polyfit(t, y, 2)
        return np.array([z_x, z_y])

    def get_error(self, data, model):
        error = []
        for pt in data:
            x_opt = scipy.optimize.minimize(f, 
                            0.0, 
                            args=(model, pt),
                            method='TNC',
                            bounds=[(0.0,1.0)])
            error.append(x_opt.fun[0])
        err_per_point = np.array(error)
        return err_per_point
        
def f(x, *args):
    model = args[0]
    target_point = args[1]

    value = (model[0][0]*x**2 + model[0][1]*x + model[0][2] - target_point[0])**2 \
            + (model[1][0]*x**2 + model[1][1]*x + model[1][2] - target_point[1])**2
    return value

def fprime(x, **args):
    model = args[0]
    target_point = args[1]
    value = 2*(model[0][1] + 2*model[0][0]*x)*\
            (model[0][2] + model[0][1]*x + model[0][0]*x**2 - target_point[0]) + \
            2*(model[1][1] + 2*model[1][0]*x)*\
            (model[1][2] + model[1][1]*x + model[1][0]*x**2 - target_point[1])
    return value

def test():
    with open("test_region_points.dat", "r") as fin:
        point_collection = cPickle.load(fin)
    
    # setup model
    data = np.array([[pt[0] for pt in point_collection], [pt[1] for pt in point_collection]]).T
    data[:,0] /= 1e5
    data[:,1] /= 1e6

    debug = False
    model = LinerCurveModel(debug=debug)


    # run RANSAC algorithm
    current_step = 0
    results = []
    while True:
        this_step_result = {}
        if current_step >=3:
            break
        goal = np.floor(0.1*len(data))
        ransac_fit, ransac_data = ransac(data,
                                         model,
                                         goal, 
                                         100, 
                                         20, 
                                         10, # misc. parameters
                                         debug=debug,
                                         return_all=True)
        print ransac_fit

        this_step_result['model'] = copy.deepcopy(ransac_fit)
        this_step_result['inliers'] = copy.deepcopy(ransac_data['inliers'])
        this_step_result['data'] = copy.deepcopy(data)
        results.append(copy.deepcopy(this_step_result))

        # Remove inlier
        inlier_ind = Set(ransac_data['inliers'])
        all_ind = Set(np.arange(data.shape[0]))
        remaining_ind = all_ind.difference(inlier_ind)
        new_data = np.array([[data[ind][0] for ind in remaining_ind], 
                            [data[ind][1] for ind in remaining_ind]]).T

        print "hello, before %d"%(len(data))
        data = new_data
        print "hello, after %d"%(len(new_data))

        current_step += 1
        
    if 1:
        import pylab

        sort_idxs = np.argsort(data[:,0])
        A_col0_sorted = data[sort_idxs] # maintain as rank-2 array

        if 1:
            pylab.plot([pt[0] for pt in point_collection],
                       [pt[1] for pt in point_collection],
                       'k.', 
                       label='data' )

        EPSILON = 1e-5
        for result in results:
            data = result['data']
            ransac_fit = result['model']
            ransac_data = result['inliers']
            
            min_x = np.min(data[:,0])
            max_x = np.max(data[:,0])
            x = np.linspace(min_x, max_x, 100, endpoint=True)

            if np.abs(ransac_fit[0]) <= EPSILON:
                # a == 0
                y = np.ones(100) * (-1) * ransac_fit[2] / ransac_fit[1]
            else:
                if np.abs(ransac_fit[1]) <= EPSILON:
                    min_y = np.min(data[:,1])
                    max_y = np.max(data[:,1])
                    y = np.linspace(min_y, max_y, 100, endpoint=True)
                    x = np.ones(100) * (-1) * ransac_fit[2] / ransac_fit[0]
                else:
                    y = -1 * ransac_fit[2] - ransac_fit[0]*x
                    y /= ransac_fit[1]

            x *= 1e5
            y *= 1e6

            pylab.plot( x,
                        y,
                        label='RANSAC fit' )
            data[:,0] *= 1e5
            data[:,1] *= 1e6

            pylab.plot( [data[ind,0] for ind in ransac_data], 
                        [data[ind,1] for ind in ransac_data],
                        'r.')
        pylab.legend()
        pylab.show()

if __name__=='__main__':
    test()
    
