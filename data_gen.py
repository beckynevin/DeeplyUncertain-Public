import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt


def pendulum(n=10000, n_t=10, g_range=None, theta_range=None, ell_range=None, m_range=None, t_spread=None,
             ell_spread=None, seed=42, plot=True):
    """_summary_

    Args:
        n (int, optional): Number of data points. Defaults to 10000.
        n_t (int, optional): Number of moments in period oscillation. Defaults to 10.
        g_range (_type_, optional): Range of gravitational constant values?. Defaults to None.
        theta_range (_type_, optional): Range of theta values for pendulum. 
            This range sets range of error for statistical error. Defaults to None.
        ell_range (_type_, optional): Range of L, length values. 
            This range represents the systemic error. Defaults to None.
        m_range (_type_, optional): Range of mass values. Defaults to None.
        t_spread (_type_, optional): Period of oscillation. Defaults to None.
        ell_spread (_type_, optional): ??. Defaults to None.
        seed (int, optional): Random seed for generating data points, if you change this you will get different numbers. Defaults to 42.
        plot (boolean, optional): Defaults to plotting
        
    Returns:
        _type_: _description_
    """
    
    
    if g_range is None:
        g_range = [5, 15]
    if theta_range is None:
        theta_range = [5, 15]
    if ell_range is None:
        ell_range = [0.2, 0.8]
    if m_range is None:
        m_range = [0.02, 0.1]
    if t_spread is None:
        # Used to define t_scale, which is the std of error added to t
        # ^I think this is experimental or systematic error
        t_spread = [0.03, 0.03]
    if ell_spread is None:
        ell_spread = [0., 0.]

    np.random.seed(seed)

    # for the input number of datapoints (n) choose a randomized g value between the range
    # of allowed values:
    # np.random.rand(n) gives random numbers in an uniform range
    g = (g_range[1] - g_range[0]) * np.random.rand(n) + g_range[0]
    # do the same for theta and L and m:
    theta = ((theta_range[1] - theta_range[0]) * np.random.rand(n) + theta_range[0]).reshape((n, 1))
    ell = ((ell_range[1] - ell_range[0]) * np.random.rand(n) + ell_range[0]).reshape((n, 1))
    m = ((m_range[1] - m_range[0]) * np.random.rand(n) + m_range[0]).reshape((n, 1))
    
    # This is the equation of motion of a pendulum under the small angle approx,
    # which produces simple harmonic motion:
    t = (2 * np.pi * np.sqrt(ell / g.reshape((n, 1))))
    t_og = t
    # ^ This is cool because the period of oscillation depends only on length
    # and gravity, and not on the angle of displacement or mass
    # Pendulums with the same length but different masses will oscillate
    # with the same period: https://www.acs.psu.edu/drussell/Demos/Pendulum/Pendulum.html
    
    if plot:
        # Plot these values:
        plt.clf() 
        plt.scatter(theta, ell, s=0.1, color='black')
        plt.xlabel(r'$\Theta$')
        plt.ylabel(r'L')
        plt.show()  
        
        plt.clf()
        plt.scatter(ell, g, c = t, s=0.2)
        plt.colorbar(label='period of oscillation, this is what we are solving for')
        plt.xlabel('L')
        plt.ylabel('g')
        plt.show()
          
    

    t_scales = np.random.uniform(t_spread[0], t_spread[1], n)
    t_scales = np.repeat(t_scales, n_t).reshape((n, n_t))
    # Drawing from a random normal, 'scale' is the standard deviation
    
    t_spreads = np.random.normal(scale=t_scales, size=(n, n_t))
    t_sigma = t_spreads * t
    t = t + t_sigma
    print('shape of t before', np.shape(t_og),'shape after', np.shape(t[:,0]))
    
    if plot:
        # Plot these values:
        plt.clf()
        fig = plt.figure()
        ax0 = fig.add_subplot(311)
        im0 = ax0.scatter(ell, g, c = t_og, s=0.4)
        plt.colorbar(im0, label='period')
        ax0.set_xlabel('L')
        ax0.set_ylabel('g')
        
        ax1 = fig.add_subplot(312)
        im1 = ax1.scatter(ell, g, c = t[:,0], s=0.4)
        plt.colorbar(im1, label='period + error')
        
        ax2 = fig.add_subplot(313)
        diff = [x - y for x,y in zip(t[:,0],t_og)]
        im2 = ax2.scatter(ell, g, c = diff, s=0.4)
        plt.colorbar(im2, label='difference')
        
        
        plt.show()
    
    
    
    

    ell_scales = np.random.uniform(ell_spread[0], ell_spread[1], n).reshape((n, 1))
    ell_spreads = np.random.normal(scale=ell_scales, size=(n, 1))
    ell_sigma = ell_spreads * ell
    ell = ell + ell_sigma

    # Feature vector is filled with:
    # 1) theta values with an uniform scatter (systematic?)
    # 2) L + scatter (aleatoric systematic, normal, comes from measurement error)
    # 3) m values also have an uniform scatter (systematic?)
    # 4) t, or period + scatter (aleatoric statistical, normal)
    # g is independent of m and theta, although you can use them to introduce
    # additional systematics, which they didn't do in the paper
    # i.e. moving away from the small angle approximation
    # epistemic uncertainty is in theta and g being outside the allowed values?!
    feat = np.concatenate([theta, ell, m, t], axis=1)
    y = g

    # [Aleatoric] statistical uncertainty (OG comment)
    # is then estimated from adding noise to the period
    delta_t = np.std(t_sigma, axis=1) / np.sqrt(2) * gamma((n_t-1)/2) / gamma(n_t/2)
    mean_t = np.mean(t, axis=1)
    ell = ell.reshape(n, )
    delta_ell = ell * ell_scales.reshape(n, )
    calc_y = 4 * np.pi ** 2 * ell / mean_t ** 2
    # propagating noise in each input variable to the final calculation of g:
    # total predicted uncertainty is added in quadrature
    delta_y = 4 * np.pi ** 2 / mean_t ** 2 * np.sqrt((2 * ell * delta_t / mean_t) ** 2 + delta_ell ** 2)

    return feat, y, calc_y, delta_y
