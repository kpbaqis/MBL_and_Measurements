import pickle
import numpy as np
import time

from qutip import basis, qeye, sigmax, sigmay, sigmaz, sigmap, sigmam, tensor, expect
from qutip import mesolve, mcsolve, ket2dm, Options


## Functions for saving and loading pickled data ##

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

## Routines for defining basis ##

def pauli_operators(N):
    # Define lists of Pauli operators from QuTiP built in functions
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    sp = sigmap()
    sx_list = []
    sy_list = []
    sz_list = []
    sm_list = []
    sp_list = []

    for n in range(N):
        op_list = [si for m in range(N)]
            
        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))
        
        op_list[n] = sz
        sz_list.append(tensor(op_list))
        
        op_list[n] = sm
        sm_list.append(tensor(op_list))    
        
        op_list[n] = sp
        sp_list.append(tensor(op_list))
        
    return sx_list, sy_list, sz_list, sm_list, sp_list

def fermi_operators(N):
    # Construct fermion operators from Pauli operators using Jordan-Wigner
    _,_,pz,pp,_ = pauli_operators(N)
    ad = []
    a = []
    n=[]
    for j in range(N):
        adj = pp[j]
        for k in range(j):
            adj=pz[j-1-k]*adj
        ad.append(adj)
        a.append(adj.dag())
        n.append(adj*adj.dag())
    return ad,a,n

## Routines for intialising, evolution and calculating imbalance ##

def site_fermions(N,state):
    #N is the number of physical qubits
    #state is the input state you wish to measure Sz at each site
    _, _, n = fermi_operators(N)
    return [expect(n[j],state) for j in range(N)]

def imbalance(N,state):
    #Ne(No) number of excitation on even(odd) sites
    #imbalance is I = (Ne-No)/(Ne+No)
    excitations = site_fermions(N,state)
    Ne = sum(excitations[1::2])
    No = sum(excitations[0::2])
    return (No-Ne)/(Ne+No)   


def L_a(N, gamma, site):    
    _, a, _ = fermi_operators(N)
    return gamma * a[site]

def L_ad(N, gamma, site):    
    ad, _, _ = fermi_operators(N)        
    return gamma * ad[site]

def L_n(N, gamma, site):
    _, _, n = fermi_operators(N)
    return gamma * n[site]

def hamiltonian(N,J,V,d):
    ad,a,n = fermi_operators(N)
    
    rng = np.random.default_rng()
    dis = rng.uniform(-d,d,N)
    
    H = 0
    for j in range(N-1):
        H += -J*(ad[j]*a[j+1] + ad[j+1]*a[j]) + V*n[j]*n[j+1]
    for j in range(N):
        H += dis[j]*n[j]
    
    ## Add closed boundaries
    #H += -J*(ad[N-1]*a[0] + ad[0]*a[N-1]) + V*n[N-1]*n[0]
    
    ## Add tilt to stop steady state collapsing to identity
    #d = 0.2
    #H += sum([j*d*n[j] for j in range(N)])
        
    return H        

def trajectory(N, H, gamma_n, gamma, psi0, tlist, solver, opts):

    ## Lindblad operators
    c_op_list = []
    
    ## spin dephasing 
    for j in range(N):
        if gamma != 0:
            c_op_list.append(L_a(N, gamma, j))
            c_op_list.append(L_ad(N, gamma, j))
            c_op_list.append(L_n(N, gamma_n, j))
            print("Site ",j+1," Lindblad operator constructed")
        else:
            c_op_list.append(L_n(N, gamma_n, j))
            print("Site ",j+1," Lindblad operator constructed")
            
    # evolve without collapse operators
    if solver == "closed":
        result = mesolve(H, psi0, tlist, options=opts) 

    # evolve using master equation with collapse operators
    if solver == "me":
        result = mesolve(H, psi0, tlist, c_op_list, options=opts) 
    
    return result.states

def initial_neel(N):
    ## intial state, Neel state |010101...>
    psi_list = []
    for n in range(N):
        if (n%2) == 1:
            psi_list.append(basis(2,0))
        else:
            psi_list.append(basis(2,1))
    psi0 = tensor(psi_list)
    return psi0

## Main routines for producing time evolution data ##

def data_script():
    print('\nBegin simulation for master equation solver')

    N = 10
    J = 1
    V = 2.

    gamma = 0

    solver = "me"
    psi0 = initial_neel(N)
    
    opts = {"nsteps": 5000, "progress_bar": True}

    cut = N/2

    tnum = 100
    tlist = np.logspace(-2, 2, num=tnum)

    rstart = 0
    realisations = 300

    dis = 5.0
    dis = round(dis,1)

    for gamma_n in [5e-4,5e-3,5e-2,5e-1]:
        gamma_n = round(gamma_n,4)

        for realisation in range(rstart,rstart+realisations):
            print("\nRun for disorder = ", dis)
            print("\nRun for gamma_n = ", gamma_n)
            print('\nRealisation number: ',realisation)

            while True:
                try:
                    t0 = time.time()
                    H = hamiltonian(N,J,V,dis)

                    states = trajectory(N, H, gamma_n**2, gamma, psi0, tlist, solver, opts)
                    print('\nStates produced from master equation approach')

                    im = []
                    for rho in states:
                        im.append(imbalance(N,rho))

                    print('\nTime taken for this realisation: ',time.time()-t0)
                except Exception as e:
                    print('Exception found, retrying this realisation.')
                    print('Exception is: \n',e)
                else:
                    break


            savedata = {'N': N, 'gamma_n': gamma_n, 'gamma': gamma, 'tlist': tlist, 'Im': im, 'rho': states}
            save_obj(savedata, '../data/N{}_dis_{}_gamman_{}_gamma_{}_realisation_{}'.format(N,str(dis).replace('.','_'),str(gamma_n).replace('.','_'),str(gamma).replace('.','_'),str(realisation)))
            print('\nData saved to file.\n')


def processing():
    # Routine for averaging over disorder realisations
    N = 10
    dis = 5.
    dis = round(dis,1)
    realisations = 300

    gamma = 0
 
    tnum = 100
    tlist = np.logspace(-2, 2, num=tnum)
    
    imbalance = [] 
    dm = []
    for gamma_n in [0.]:
        gamma_n = round(gamma_n,4)
        print('Measurement rate: ', gamma_n)
            
        imb = []
        rho = []
        for realisation in range(realisations):
            print('Realisation number: ', realisation)
            file = load_obj('../data/N{}_dis_{}_gamman_{}_gamma_{}_realisation_{}'.format(N,str(dis).replace('.','_'),str(gamma_n).replace('.','_'),str(gamma).replace('.','_'),str(realisation)))
            imb.append(file['Im'])
            rho.append(file['rho'])
        imbalance.append([np.array(np.average([item[n] for item in imb])) for n in range(tnum)])
        dm.append([np.average([ket2dm(item[n]) for item in rho]) for n in range(tnum)])
        del rho
        del imb

    for gamma_n in [5e-4,5e-3,5e-2,5e-1]:
        gamma_n = round(gamma_n,4)
        print('Measurement rate: ', gamma_n)

        imb = []
        rho = []
        for realisation in range(realisations):
            print('Realisation number: ', realisation)
            file = load_obj('../data/N{}_dis_{}_gamman_{}_gamma_{}_realisation_{}'.format(N,str(dis).replace('.','_'),str(gamma_n).replace('.','_'),str(gamma).replace('.','_'),str(realisation)))
            imb.append(file['Im'])
            rho.append(file['rho'])

        imbalance.append([np.array(np.average([item[n] for item in imb])) for n in range(tnum)])
        dm.append([np.average([item[n] for item in rho]) for n in range(tnum)])        
        del rho
        del imb

    savedata = {'N': N, 'Im': imbalance, 'rho': dm, 'tlist': tlist}    
    save_obj(savedata, '../data/im_rho_data_N{}_dis_{}'.format(N,str(dis).replace('.','_')))    



def __main__():
    data_script()
    #processing()

if __name__ == '__main__':
    __main__()
