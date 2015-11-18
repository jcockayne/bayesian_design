__author__ = 'benorn'
import numpy as np
import inspect


def numpy_to_scipy_bounds(b):
    # degenerate 1D case
    if len(b.shape) == 1 and b.shape[0] == 2:
        return [(b[0], b[1])]
    else:
        return [(b[i,0], b[i, 1]) for i in range(b.shape[0])]


def deletion(points, loss):
    base_loss = loss(points)
    indices = np.arange(len(points))
    new_losses = np.empty(points.shape[0])
    for i in indices:
        flags = indices != i
        new_losses[i] = loss(points[flags])

    return new_losses, base_loss


def a_optimality(get_cov):
    return lambda points: np.sum(np.diag(get_cov(points))**2)


def __plot_loss(cur_location, new_location, bounds, emulator, emu_points, other_points):
    if len(bounds) == 1:
        __plot_loss_1d(cur_location, new_location, bounds, emulator, emu_points, other_points)
    elif len(bounds) == 2:
        __plot_loss_2d(cur_location, new_location, bounds, emulator, emu_points, other_points)
    else:
        raise Exception('Cannot plot loss when problem has dimension > 2')


def __plot_loss_1d(cur_location, new_location, bounds, emulator, emu_points, other_points):
    pass


def __plot_loss_2d(cur_location, new_location, bounds, emulator, emu_points, other_points):
    import matplotlib.pyplot as plt
    x,y = np.mgrid[bounds[0][0]:bounds[0][1]:PLOT_POINTS, bounds[1][0]:bounds[1][1]:PLOT_POINTS]
    locs = np.c_[x.ravel(), y.ravel()]
    emu_values = emulator(locs)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.scatter(cur_location[0], cur_location[1], c='red', linewidths=0, label='Original Location')
    plt.scatter(new_location[0], new_location[1], c='green', linewidths=0, label='New Location')
    plt.scatter(emu_points[:,0], emu_points[:,1], c='gray', label='Emulator Design Points', marker='x')
    arrow_jitter = new_location-cur_location
    arrow_start = cur_location + 0.1*arrow_jitter
    arrow_delta = arrow_jitter*0.8
    plt.arrow(
        arrow_start[0], arrow_start[1],
        arrow_delta[0], arrow_delta[1],
        alpha=0.5,
        color='black',
        linewidth=0,
        length_includes_head=True
    )
    other_points_in_range = other_points[
        (other_points[:,0] > bounds[0][0])
        & (other_points[:,0] < bounds[0][1])
        & (other_points[:,1] > bounds[1][0])
        & (other_points[:,1] < bounds[1][1]), :]
    plt.scatter(other_points_in_range[:,0], other_points_in_range[:,1], c='black', alpha=0.5, label='Other Design Points')
    plt.contour(x, y, emu_values.reshape(x.shape))
    plt.legend(loc=3, bbox_to_anchor=(1,1.05))

    plt.subplot(122)
    plt.scatter(other_points[:,0], other_points[:,1], c='black')
    plt.scatter(cur_location[0], cur_location[1], c='red', linewidths=0)
    plt.scatter(new_location[0], new_location[1], c='green', linewidths=0)

    plt.show()
    plt.close()


def ace(initial_design, k, max_iter, loss_fn, optimizer, debug=False):
    # initial_design should have shape 2; else reshape it
    if len(initial_design.shape) == 1:
        initial_design = initial_design[:,None]
    if len(initial_design.shape) > 2:
        raise Exception("Initial design should be a 2D array, with one row per point and one column per dimension.")

    cur_design = initial_design.copy()
    cur_loss = loss_fn(cur_design)
    for ix in xrange(max_iter):
        losses, base_loss = deletion(cur_design, loss_fn)

        points_to_modify = np.argsort(losses-base_loss)[:k]
        for i in points_to_modify:
            cur_point = cur_design[i,:]
            # dump out the point we are modifying
            index_mask = np.arange(len(cur_design)) != i
            other_points = cur_design[index_mask, :]

            # proxy for the loss function, as a function of the point we are modifying
            def __partial_loss(d):
                tmp_points = np.concatenate((other_points, d.reshape((1, other_points.shape[1]))))
                return loss_fn(tmp_points)

            def __partial_loss_vectorized(points):
                ret = np.array([[__partial_loss(points[i1, :])] for i1 in xrange(points.shape[0])])
                return ret

            # construct the emulator
            new_point = optimizer(__partial_loss_vectorized, cur_point, other_points)

            # now decide whether to _use_ that point.
            # In Overstall & Woods we have an acceptance probability for the proposal which is not included here.
            # This is because our loss function is exact (it is the trace of posterior covariance) so we can just look
            # at whether the new location is a better design than the old one, rather than looking at probabilities.
            new_loss = __partial_loss(new_point)
            accept = new_loss < cur_loss
            if accept:
                if debug > 0:
                    print ix, "Moved {} from {} to {} (new loss {:.2e} < {:.2e})".format(i, cur_point, new_point, new_loss, cur_loss)
                cur_design[i, :] = new_point
                cur_loss = new_loss
            else:
                # no-op
                if debug > 0:
                    print ix, "Rejected move of {} from {} to {}, (new loss {:.2e} > {:.2e})".format(i, cur_point, new_point, new_loss, cur_loss)

    return cur_design