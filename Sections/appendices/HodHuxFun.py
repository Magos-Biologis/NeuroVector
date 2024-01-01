import numpy                as np
from numpy              import exp


# The system of ODEs that defines the HH model of AP
def HodgkinnyHuxer(t, xs, *pars):
    # Tries to fit parameters if they are not labeled well
    # Allows the use before their definitions
    try:
       Vrest            = pars['Vrest'].value
       I, Inj_t         = pars['I'].value,    pars['Inj_t'].value
       C_m              = pars['C_m'].value  
       G_Na, G_K, G_L   = pars['G_Na'].value, pars['G_K'].value, pars['G_L'].value
       E_Na ,E_K, E_l   = pars['E_Na'].value, pars['E_K'].value, pars['E_l'].value
       dt               = pars['dt'].value
    except:
       Vrest, I, Inj_t, C_m, G_Na, G_K, G_L, E_Na, E_K, E_l, dt = pars
       
    # Attributes the relevant variable to the array term
    V, n, m, h  = xs 
      
    I_Na        = G_Na * (m**3) * h * (V - E_Na)
    I_K         = G_K  * (n**4)     * (V - E_K)
    I_L         = G_L               * (V - E_l)
    
    # Allows one to choose an injection time
    while t < Inj_t:
        I = 0

    ddt_V       = (I - (I_Na + I_K + I_L))*(C_m**(-1))
    ddt_n       = alphN(V - Vrest) * (1.0 - n) - betaN(Vrest - V) * n 
    ddt_m       = alphM(V - Vrest) * (1.0 - m) - betaM(Vrest - V) * m 
    ddt_h       = alphH(Vrest - V) * (1.0 - h) - betaH(V - Vrest) * h 
    
    return [ddt_V, ddt_n, ddt_m, ddt_h]


# Using Python's lambda feature 
# To define our alpha and beta functions
# Functionally(ish) the same as 
# 'def alphH(v): return 0.07 * exp( -(v / 20) )'
alphN = lambda Vm: ((10.0 - Vm) / 100.0) / ( exp((10.0 - Vm) / 10.0) - 1.0 ) 
betaN = lambda Vm: 0.125 *  exp( Vm / 80.0 )

alphM = lambda Vm: ((25.0 - Vm) / 10.0 ) / ( exp((25.0 - Vm) / 10.0) - 1.0 )
betaM = lambda Vm: 4.0   *  exp( Vm / 18.0 )

alphH = lambda Vm: 0.07  *  exp( Vm / 20.0 )
betaH = lambda Vm: 1.0   / ( 1 + exp((30.0 - Vm) / 10.0) )


# Representative of the probabilities as 
# the limit of the channel count approaches infinity 
n_inf = lambda Vm: alphN(Vm) / (alphN(Vm) + betaN(Vm))
m_inf = lambda Vm: alphM(Vm) / (alphM(Vm) + betaM(Vm))
h_inf = lambda Vm: alphH(Vm) / (alphH(Vm) + betaH(Vm))


# Nerst equations
nrnt    = lambda c,z,inn,out: ((c + 273.15) / z)*np.log(out / inn)*11.61**(-1) 
nrnt_K  = lambda c: nrnt(c,  1, 115, 4) # Calculates K  Epote
nrntNa  = lambda c: nrnt(c, 1, 16, 127) # Calculates Na Epote
nrntCl  = lambda c: nrnt(c,-1, 10, 131) # Calculates Cl Epote
nrntCa  = lambda c: nrnt(c, 2, 1e-4, 1) # Calculates Ca Epote


# Constants definitions
# # Resting Potential, we use a funny number to avoid division by 0
Vrest           = -64.99584 
# Current
I               =  15.0 
# Capacitance
C_m             =  1.0 

# Human body temp; 37
# Squid axon temp; 6
Temp_Celcius    = 6

E_Na            = nrntNa(Temp_Celcius) # Na Erev
E_K             = nrnt_K(Temp_Celcius) #  K Erev
E_Cl = E_l      = nrntCl(Temp_Celcius) # Cl Erev and by extension, L Erev
E_Ca            = nrntCa(Temp_Celcius) # Ca Erev
# Max Conductances for Na, K, and leakage
G_Na, G_K,  G_L = 120.0,   36.0,   0.3 

dt              = 0.01   # Time increments 
Inj_t           = 50     # When to inject in ms 

#Initial conditions
init_cond   = [Vrest,    # V initial
               n_inf(0), # n initial
               m_inf(0), # m initial
               h_inf(0)  # h initial
               ]
# init_cond   = [Vinit, 0.318, 0.053, 0.596]


# Time span for solution evaluated at selected time points
# Span of time in milliseconds
time_span   = [0, 150]  
# Divides time_span up by 0.01 step increments
time_eval   = np.arange(*time_span, dt) 

# Packages all the constants into a single list
pars = (Vrest, I, Inj_t, C_m, G_Na, G_K, G_L, E_Na, E_K, E_l, dt)
