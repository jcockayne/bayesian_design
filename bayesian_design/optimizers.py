__author__ = 'benorn'
import numpy as np


def gpyopt_optimizer(optimizer_bounds_function, debug=False):
    """
    Optimizer which uses Bayesian Optimization to minimize a function
    :param optimizer_bounds_function: Function which provides optimization bounds when moving a given sensor.
    :param debug: Whether or not to display debug plots from GPyOpt
    :return: The optimizer function
    """
    import GPyOpt

    def __gpyopt_optimizer(function, cur_point, other_points):
        bounds = optimizer_bounds_function(cur_point)
        opt = GPyOpt.methods.BayesianOptimization(function, bounds, X=np.array([cur_point]))
        opt.run_optimization(verbose=False)
        if debug:
            import matplotlib.pyplot as plt
            opt.plot_convergence()
            plt.scatter(opt.x_opt[0], opt.x_opt[1], c='green')
            plt.scatter(cur_point[0], cur_point[1], c='red')
            plt.scatter(other_points[:,0], other_points[:, 1], c='black')

            plt.show()
            plt.close()
        return opt.x_opt
    return __gpyopt_optimizer