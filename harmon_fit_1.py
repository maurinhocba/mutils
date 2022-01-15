
import numpy as np


def harmon_1(C, S, om, tini=0.0, tend=100.0, npoin=1000, noise_level=0.0):
    '''
    Creates discrete data of the form
        f(t) = C*cos(om*t) + S*sin(om*t)
    '''
    # crate base data
    t=np.linspace(tini, tend, npoin)
    f=C*cos(om*t) + S*sin(om*t)
    
    # add noise
    rng = np.random.default_rng()
    f=f + noise_level*rng.normal(size=t.size)
    
    return t, f


def harmon_fit_1(t, f):
    '''
    Fit a 2-term harmonic function of the form
        f(t) = C*cos(om*t) + S*sin(om*t)
    from discrete data.
    C, S and om are unknown.
    The name ends in "1" because f(t) can be represented with a unique sine or cosine term adding the corresponding phase.
    '''
    # define de form of the searched function
    def objective_func(t, C, S, om):
        return C*cos(om*t) + S*sin(om*t)
    
    popt, pcov = curve_fit(objective_func, t, f)
    
    return popt, pcov

if __name__=__main__:
    # try harmon_fit_1
    if True:
        # data
        C=2.0
        S=3.0
        om=1.5
        t,f = harmon_1(C, S, om, noise_level=1.0)
        
        # fit
        popt, pcov = harmon_fit_1(t, f)
        
        # plot
        plt.plot(harmon_1(C, S, om, noise_level=0.0), 'k-', label='data without noise')
        plt.plot(t, f, 'b-', label='data with noise')
        plt.plot(harmon_1(*popt, noise_level=0.0), 'r-', label='fit: C=%5.3f, S=%5.3f, om=%5.3f' % tuple(popt))
