__author__ = 'benorn'
import numpy as np

MAX_ITER = 10

def gpyopt_optimizer(optimizer_bounds_function):
    """
    Optimizer which uses Bayesian Optimization to minimize a function
    :param optimizer_bounds_function: Function which provides optimization bounds when moving a given sensor.
    :param debug: Whether or not to display debug plots from GPyOpt
    :return: The optimizer function
    """
    import GPyOpt

    def __gpyopt_optimizer(function, cur_point, other_points, debug=False, return_opt=False):
        if debug > 1:
            print("Received cur_point: {}, other_points: {}".format(cur_point, other_points))
        X = np.array([cur_point])

        bounds = optimizer_bounds_function(cur_point)
        # todo: this doesn't seem to work with an input x
        opt = GPyOpt.methods.BayesianOptimization(function, bounds)
        opt.run_optimization(MAX_ITER, verbosity=debug)
        if debug > 2:
            import matplotlib.pyplot as plt
            opt.plot_convergence()
            plt.scatter(opt.x_opt[0], opt.x_opt[1], c='green')
            plt.scatter(cur_point[0], cur_point[1], c='red')
            plt.scatter(other_points[:,0], other_points[:, 1], c='black')

            plt.show()
            plt.close()
        if not return_opt:
            return opt.x_opt
        return opt
    return __gpyopt_optimizer