
'''
Set of utilities to reconstruct harmonic functions from discrete data

Mauro S. Maza
2022-01-15
'''

# ##############
# IMPORTING ZONE
# ##############
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cfit
from scipy.optimize import differential_evolution as difevo


# ###################
# AUXILIARY FUNCTIONS
# ###################
def harmon_1(C, S, om, tini=0.0, tend=100.0, npoin=1000, noise_level=0.0):
    '''
    Creates discrete data of the form
        f(t) = C*cos(om*t) + S*sin(om*t)
    '''
    # crate base data
    t=np.linspace(tini, tend, npoin)
    f=C*np.cos(om*t) + S*np.sin(om*t)
    
    # add noise
    rng = np.random.default_rng()
    f=f + noise_level*rng.normal(size=t.size)
    
    return t, f


def harmon_2(C1, S1, om1, C2, S2, om2, tini=0.0, tend=100.0, npoin=1000, noise_level=0.0):
    '''
    Creates discrete data of the form
        f(t) = C1*cos(om1*t) + S1*sin(om1*t)  +  C2*cos(om2*t) + S2*sin(om2*t)
    '''
    # crate base data
    t=np.linspace(tini, tend, npoin)
    f=C1*np.cos(om1*t) + S1*np.sin(om1*t)  +  C2*np.cos(om2*t) + S2*np.sin(om2*t)
    
    # add noise
    rng = np.random.default_rng()
    f=f + noise_level*rng.normal(size=t.size)
    
    return t, f


def harmon_2e(C1, S1, om1, C2, S2, om2, D, A, B, tini=0.0, tend=100.0, npoin=1000, noise_level=0.0):
    '''
    Creates discrete data of the form
        f(t) = [  C1*cos(om1*t) + S1*sin(om1*t)  +  C2*cos(om2*t) + S2*sin(om2*t)  ]*exp(Dt)  +  A*t + B
    '''
    # crate base data
    t=np.linspace(tini, tend, npoin)
    f=(  C1*np.cos(om1*t) + S1*np.sin(om1*t)  +  C2*np.cos(om2*t) + S2*np.sin(om2*t)  )*np.exp(D*t)  +  A*t + B
    
    # add noise
    rng = np.random.default_rng()
    f=f + noise_level*rng.normal(size=t.size)
    
    return t, f


# #########
# FUNCTIONS
# #########
def harmon_fit_1(t, f):
    '''
    Fit a 2-term harmonic function of the form
        f(t) = C*cos(om*t) + S*sin(om*t)
    from discrete data using non-linear least squares (scipy.optimize.curve_fit).
    C, S and om are unknown parameters.
    The name ends in "1" because f(t) can be represented with a unique sine or cosine term adding the corresponding phase.
    
    Have a big problem with frequency: the intial guess for that parameter must be very good.
    '''
    
    # define de form of the searched function
    def objective_func(t, C, S, om):
        return C*np.cos(om*t) + S*np.sin(om*t)
    
    popt, pcov = cfit(objective_func, t, f, p0=[1,1,1.3])
    
    return popt, pcov
    

def harmo_genalg_1(t, f, bounds):
    '''
    Fit a 2-term harmonic function of the form
        f(t) = C*cos(om*t) + S*sin(om*t)
    from discrete data using genetic algorithms (scipy.optimize.differential_evolution).
    C, S and om are unknown parameters.
    The name ends in "1" because f(t) can be represented with a unique sine or cosine term adding the corresponding phase.
    
    It is sensitive to the initial guess for om, but not so baddly as harmon_fit_1.
    
    t and f are vectors with data to be fit
    bounds=[(C_min,C_max), (S_min,S_max), (om_min,om_max)] determines searching intervals for the unknown parameters.
    '''
    
    def test(C, S, om):
        return C*np.cos(om*t) + S*np.sin(om*t)
    
    def minimize_func(parameters, *args):
        '''
        Determines the residual in the least squares sense
        '''
        C, S, om = parameters
        t, f = args
        
        residual = (np.sum(  np.power( test(C, S, om) - f, 2 )  ))**0.5
        return residual
    
    result = difevo(minimize_func, bounds, args=(t,f))
    
    if result.success==True:
        C, S, om = result.x
        # Mean squared error
        mse = (np.sum(  np.power( test(C, S, om) - f, 2 )  ))/t.size
        return (C, S, om, mse)
    else:
        print('problems with scipy.optimize.differential_evolution trying to fit data')
    

def harmo_genalg_2(t, f, bounds):
    '''
    Fit a 2-term harmonic function of the form
        f(t) = C1*cos(om1*t) + S1*sin(om1*t)  +  C2*cos(om2*t) + S2*sin(om2*t)
    from discrete data using genetic algorithms (scipy.optimize.differential_evolution).
    C1, S1, om1, C2, S2 and om2 are unknown parameters.
    The name ends in "2" because f(t) can be represented with 2 sine or cosine terms adding the corresponding phase.
    
    It is sensitive to the initial guess for om1 and om2.
    In case de input signal is a 1 term harmonic function (the type of function it is NOT meant for)
    - works perfect if the signal has no noise (C=S=0 for one of the terms)
    - but finds a second term if there is noise in the signal
    
    t and f are vectors with data to be fit
    bounds=[(C1_min,C1_max), (S1_min,S1_max), (om1_min,om1_max), (C2_min,C2_max), (S2_min,S2_max), (om2_min,om2_max)]
        determines searching intervals for the unknown parameters.
    '''
    
    def test(C1, S1, om1, C2, S2, om2):
        return C1*np.cos(om1*t) + S1*np.sin(om1*t)  +  C2*np.cos(om2*t) + S2*np.sin(om2*t)
    
    def minimize_func(parameters, *args):
        '''
        Determines the residual in the least squares sense
        '''
        C1, S1, om1, C2, S2, om2 = parameters
        t, f = args
        
        residual = (np.sum(  np.power( test(C1, S1, om1, C2, S2, om2) - f, 2 )  ))**0.5
        return residual
    
    result = difevo(minimize_func, bounds, args=(t,f))
    
    if result.success==True:
        C1, S1, om1, C2, S2, om2 = result.x
        if om1>om2:
            C1, S1, om1, C2, S2, om2 = C2, S2, om2, C1, S1, om1
        # Mean squared error
        mse = (np.sum(  np.power( test(C1, S1, om1, C2, S2, om2) - f, 2 )  ))/t.size
        return (C1, S1, om1, C2, S2, om2, mse)
    else:
        print('problems with scipy.optimize.differential_evolution trying to fit data')
    

def harmo_genalg_2sh(t, f, omBounds):
    '''
    Easy to use version of harmo_genalg_2.
    As the intial guesses for the amplitudes seem not to be a problem for stability, only the discrete data and bounds for the frequencies are required by this function. Other bounds are determined internally.
    
    t and f are vectors with data to be fit
    omBounds=[(om1_min,om1_max), (om2_min,om2_max)]
        determines searching intervals for the unknown frequencies.
    '''
    
    # find bounds
    max_abs = np.abs(f).max()
    bounds = [(-max_abs,max_abs), (-max_abs,max_abs), omBounds[0], (-max_abs,max_abs), (-max_abs,max_abs), omBounds[1]]
    
    return harmo_genalg_2(t, f, bounds)
    

def harmo_genalg_2e(t, f, bounds):
    '''
    Fit a 2-term harmonic function with varing amplitude and mean value of the form
        f(t) = [  C1*cos(om1*t) + S1*sin(om1*t)  +  C2*cos(om2*t) + S2*sin(om2*t)  ]*exp(Dt)  +  A*t + B
    from discrete data using genetic algorithms (scipy.optimize.differential_evolution).
    C1, S1, om1, C2, S2, om2, D, A and B are unknown parameters.
    The name ends in "2e" because it is the extended version of a function f(t) that can be represented with 2 sine or cosine terms adding the corresponding phase.
    
    It is VERY sensitive to the initial guesses.
    Maybe another approach is needed, for example, finding approximations for A and B and detrending the signal based on those values in a first step, and then find the other parameters.
    
    t and f are vectors with data to be fit
    bounds=[(C1_min,C1_max), (S1_min,S1_max), (om1_min,om1_max), (C2_min,C2_max), (S2_min,S2_max), (om2_min,om2_max), (D_min,D_max), (A_min,A_max), (B_min,B_max)]
        determines searching intervals for the unknown parameters.
    '''
    
    def test(C1, S1, om1, C2, S2, om2, D, A, B):
        return (  C1*np.cos(om1*t) + S1*np.sin(om1*t)  +  C2*np.cos(om2*t) + S2*np.sin(om2*t)  )*np.exp(D*t)  +  A*t + B
    
    def minimize_func(parameters, *args):
        '''
        Determines the residual in the least squares sense
        '''
        C1, S1, om1, C2, S2, om2, D, A, B = parameters
        t, f = args
        
        residual = (np.sum(  np.power( test(C1, S1, om1, C2, S2, om2, D, A, B) - f, 2 )  ))**0.5
        return residual
    
    result = difevo(minimize_func, bounds, args=(t,f))
    
    if result.success==True:
        C1, S1, om1, C2, S2, om2, D, A, B = result.x
        if om1>om2:
            C1, S1, om1, C2, S2, om2 = C2, S2, om2, C1, S1, om1
        # Mean squared error
        mse = (np.sum(  np.power( test(C1, S1, om1, C2, S2, om2, D, A, B) - f, 2 )  ))/t.size
        return (C1, S1, om1, C2, S2, om2, D, A, B, mse)
    else:
        print('problems with scipy.optimize.differential_evolution trying to fit data')


# ##############
# RUNNING THINGS
# ##############
if __name__=="__main__":
    
    # try harmon_fit_1
    if False:
        # data
        C=5.0
        S=3.0
        om=0.7
        t,f = harmon_1(C, S, om, noise_level=0.0)
        
        # fit
        popt, pcov = harmon_fit_1(t, f)
        
        # plot
        taux, faux = harmon_1(C, S, om, noise_level=0.0)
        plt.plot(taux, faux, 'k-', label='data without noise')
        plt.plot(t, f, 'b-', label='data with noise')
        taux, faux = harmon_1(*popt, noise_level=0.0)
        plt.plot(taux, faux, 'r-', label='fit: C=%5.3f, S=%5.3f, om=%5.3f' % tuple(popt))
        plt.legend()
    
    # try harmo_genalg_1
    if False:
        # data
        C=5.0
        S=3.0
        om=0.5
        t,f = harmon_1(C, S, om, noise_level=0.5)
        
        # fit
        bounds = [(-100,100), (-100,100), (0,10)]
        res = harmo_genalg_1(t, f, bounds)
        
        # plot
        taux, faux = harmon_1(C, S, om, noise_level=0.0)
        plt.plot(taux, faux, 'k-', label='data without noise')
        plt.plot(t, f, 'b-', label='data with noise')
        taux, faux = harmon_1(res[0], res[1], res[2], noise_level=0.0)
        plt.plot(taux, faux, 'r-', label='fit: C=%5.3f, S=%5.3f, om=%5.3f' % res[0:3])
        plt.legend()
    
    # try harmo_genalg_2
    # with a 2 term harmonic function (the type of function it is meant for)
    if False:
        # data
        C1=-5.0
        S1=3.0
        om1=0.5
        
        C2=1.0
        S2=3.0
        om2=1.0
        
        t,f = harmon_2(C1, S1, om1, C2, S2, om2, noise_level=0.5)
        
        # fit
        bounds = [(-100,100), (-100,100), (0,10), (-100,100), (-100,100), (0,10)]
        res = harmo_genalg_2(t, f, bounds)
        
        # plot
        taux, faux = harmon_2(C1, S1, om1, C2, S2, om2, noise_level=0.0)
        plt.plot(taux, faux, 'k-', label='data without noise')
        plt.plot(t, f, 'b-', label='data with noise')
        taux, faux = harmon_2(res[0], res[1], res[2], res[3], res[4], res[5], noise_level=0.0)
        plt.plot(taux, faux, 'r-', label='fit: C1=%5.3f, S1=%5.3f, om1=%5.3f, C2=%5.3f, S2=%5.3f, om2=%5.3f' % res[0:6])
        plt.legend()
    
    # try harmo_genalg_2
    # with a 1 term harmonic function (the type of function it is NOT meant for)
    # works perfect if the signal has no noise (C=S=0 for one of the terms)
    # but finds a second term if there is noise in the signal
    if False:
        # data
        C1=5.0
        S1=3.0
        om1=0.5
        
        C2=0.0
        S2=0.0
        om2=0.0
        
        t,f = harmon_2(C1, S1, om1, C2, S2, om2, noise_level=0.0)
        
        # fit
        bounds = [(-100,100), (-100,100), (0,10), (-100,100), (-100,100), (0,10)]
        res = harmo_genalg_2(t, f, bounds)
        
        # plot
        taux, faux = harmon_2(C1, S1, om1, C2, S2, om2, noise_level=0.0)
        plt.plot(taux, faux, 'k-', label='data without noise')
        plt.plot(t, f, 'b-', label='data with noise')
        taux, faux = harmon_2(res[0], res[1], res[2], res[3], res[4], res[5], noise_level=0.0)
        plt.plot(taux, faux, 'r-', label='fit: C1=%5.3f, S1=%5.3f, om1=%5.3f, C2=%5.3f, S2=%5.3f, om2=%5.3f' % res[0:6])
        plt.legend()
    
    # try harmo_genalg_2sh
    if True:
        # data
        C1=-5.0
        S1=3.0
        om1=0.5
        
        C2=1.0
        S2=3.0
        om2=1.0
        
        t,f = harmon_2(C1, S1, om1, C2, S2, om2, noise_level=0.3)
        
        # fit
        omBounds = [(0,10), (0,10)]
        res = harmo_genalg_2sh(t, f, omBounds)
        
        # plot
        plt.plot(t, f, 'b-', label='data with noise')
        taux, faux = harmon_2(C1, S1, om1, C2, S2, om2, noise_level=0.0)
        plt.plot(taux, faux, 'k-', label='data without noise')
        taux, faux = harmon_2(res[0], res[1], res[2], res[3], res[4], res[5], noise_level=0.0)
        plt.plot(taux, faux, 'r-', label='fit: C1=%5.3f, S1=%5.3f, om1=%5.3f, C2=%5.3f, S2=%5.3f, om2=%5.3f' % res[0:6])
        plt.legend()
        print(res[-1])
    
    # try harmo_genalg_2e
    # with a 2 term harmonic function (the type of function it is meant for)
    if False:
        # data
        C1=-5.0
        S1=3.0
        om1=0.5
        
        C2=1.0
        S2=3.0
        om2=1.0
        
        D=-0.01
        
        A=0.06
        B=-1
        
        t,f = harmon_2e(C1, S1, om1, C2, S2, om2, D, A, B, noise_level=0.1)
        
        # fit
        bounds = [(-10,10), (-10,10), (0,1), (-10,10), (-10,10), (0,2), (-10,10), (-10,10), (-10,10)]
        res = harmo_genalg_2e(t, f, bounds)
        
        # plot
        plt.plot(t, f, 'b-', label='data with noise')
        taux, faux = harmon_2e(C1, S1, om1, C2, S2, om2, D, A, B, noise_level=0.0)
        plt.plot(taux, faux, 'k-', label='data without noise')
        taux, faux = harmon_2e(res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], noise_level=0.0)
        plt.plot(taux, faux, 'r-', label='fit')
        plt.legend()
        print(res)