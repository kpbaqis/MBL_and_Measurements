import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.optimize import leastsq

from qutip import qeye, sigmax, sigmay, sigmaz, sigmap, sigmam, tensor
from qutip import partial_transpose, ket2dm, entropy_vn, Options

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "lines.linewidth": 2
})

## Functions for saving and loading pickled data ##

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

## Main routines for analysing and plotting data ##

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

def ent_entropy(state, cut):
    # Partial trace density matrix to find von Neumann entropy
    red_dm = state.ptrace(list(range(int(cut))))
    return entropy_vn(red_dm)

def log_neg(rho, N, cut):
    # Partial transpose density matrix for logarithmic negativity
    partition = [1]*int(cut) + [0]*(N-int(cut))
    return np.log(partial_transpose(rho, mask=partition).norm(norm='tr'))

def fermion_log_neg(rho, N, cut):
    # Define routine for calculating fermionic log neg
    # Based on qutip routine for bosonic log neg
    # Main change: inclusion of phases for exchange of fermions
    
    from qutip.core.states import state_number_index, state_number_enumerate
    from qutip.core.qobj import Qobj
    
    partition = [1]*int(cut) + [0]*(N-int(cut))
    
    A_pt = np.zeros(rho.shape, dtype=complex)

    for psi_A in state_number_enumerate(rho.dims[0]):
        m = state_number_index(rho.dims[0], psi_A)
        at1 = sum([psi_A[j] for j in range(len(partition)) if partition[j]==1])
        at2 = sum([psi_A[j] for j in range(len(partition)) if partition[j]==1])

        for psi_B in state_number_enumerate(rho.dims[1]):
            n = state_number_index(rho.dims[1], psi_B)
            bt1 = sum([psi_B[j] for j in range(len(partition)) if partition[j]==1])
            bt2 = sum([psi_B[j] for j in range(len(partition)) if partition[j]==1])

            m_pt = state_number_index(
                rho.dims[1], np.choose(partition, [psi_A, psi_B]))
            n_pt = state_number_index(
                rho.dims[0], np.choose(partition, [psi_B, psi_A]))
            
            phase = 0.5*((at1+bt1)%2) + (at1+bt1)*(at2+bt2)

            A_pt[m_pt, n_pt] = ((-1)**phase)*rho.data.as_ndarray()[m, n]

    return np.log(Qobj(A_pt, dims=rho.dims).norm(norm='tr'))

## Define functions used in fitting algorithms ##

def double_gaussian(x, params):
    (c1, mu1, sigma1, c2, mu2, sigma2) = params
    res = c1*np.exp(-(x-mu1)**2.0/(2.0*sigma1**2)) + c2*np.exp(-(x-mu2)**2.0/(2.0*sigma2**2))
    return res
    
def linear(x, A, B):
    y = A*x + B
    return y

def quartic(x, A, B, C, D, E):
    y = A*x**4 + B*x**3 + C*x**2 + D*x + E
    return y

## Main plotting scripts below ##

def plot_script():
    # Supplementary plot
    N = 10
    dis = 5.
    dis = round(dis,1)
    
    tnum = 100
    tlist = np.logspace(-2, 2, num=tnum)

    imbalance = load_obj('../data/im_rho_data_N{}_dis_{}'.format(N,str(dis).replace('.','_')))['Im'] 
 
    fig, ax = plt.subplots(figsize=(3,2), dpi=400.0)
    
    cmap = cm.get_cmap('tab10')
    colours = np.linspace(0,1,num=10)

    window = 0 ## No moving-average smoothing routine when = 0 
    idx = 0
    for cl, gamma_n in zip(np.append([0],colours),['$0.0$','$2.5 \cdot 10^{-7}$','$2.5 \cdot 10^{-5}$','$2.5 \cdot 10^{-3}$','$2.5 \cdot 10^{-1}$']):      
        
        av_data = []
        for ind in range(window,len(imbalance[idx])-window):
            av_data.append(np.mean(imbalance[idx][ind-window:ind+window+1]))   
        
        if gamma_n == '$0.0$':
            ax.semilogx(tlist[window:len(imbalance[idx])-window],av_data,label=r'{}'.format(gamma_n),c = 'black',marker = 'x',ms=2,markevery=2,lw=1)
        else:
            ax.semilogx(tlist[window:len(imbalance[idx])-window],av_data,label=r'{}'.format(gamma_n),c = cmap(cl),lw=1)
        idx += 1
  
    fs=7
    
    leg = ax.legend(title=r'$\gamma$',loc='lower left', bbox_to_anchor=(0.01, 0.01),fontsize=fs-1,title_fontsize=fs,frameon=False) 
    ax.set_ylabel(r'$\mathcal{I}(\rho(t))$',fontsize=fs) 
    ax.set_xlabel(r'$t$',fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs)
    ax.set_ylim([-0.05,1.05])
    ax.set_xlim(left=3e-2,right=4e1)
    ax.margins(0)    
    
    plt.tight_layout()
    plt.savefig('Sup1.pdf')

def combined1():
    # Figure 1 plot
    N = 10
    dis = 5.
    dis = round(dis,1)
    
    tnum = 100
    tlist = np.logspace(-2, 2, num=tnum)
    
    for idx,num in enumerate(tlist):
        if num > 3:
            tidx = idx - 1
            break

    imbalance = load_obj('../data/scaling_data/im_rho_data_N{}_dis_{}'.format(N,str(dis).replace('.','_')))['Im'] 
 
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7,4), dpi=400.0, constrained_layout = True)
    
    cmap = cm.get_cmap('tab10')
    colours = np.linspace(0,1,num=10)
    
    lw = 1.
    fs = 7

    window = 10
  
    axins = ax[0,0].inset_axes([0.45, 0.5, 0.45, 0.45], xlim=(3, 8), ylim=(0.3, 0.45))
    axins.xaxis.set_tick_params(which='major', labeltop=False, labelbottom=False)
    axins.xaxis.set_tick_params(which='minor', labeltop=False, labelbottom=False)
    axins.yaxis.set_ticklabels([])
    
    idx = 0        
    for cl,gamma_n in zip(np.append([0],colours),['$0.0$','$2.5 \cdot 10^{-7}$','$2.5 \cdot 10^{-5}$','$2.5 \cdot 10^{-3}$','$2.5 \cdot 10^{-1}$']):      
        
        av_data = []
        for ind in range(window,len(imbalance[idx])-window):
            av_data.append(np.mean(imbalance[idx][ind-window:ind+window+1]))   
        
        if gamma_n == '$0.0$':
            ax[0,0].semilogx(tlist[window:len(imbalance[idx])-window],av_data,label=r'{}'.format(gamma_n), 
                c = 'black', marker = 'x', ms = 2, markevery=2,lw=lw)
            axins.semilogx(tlist[window:len(imbalance[idx])-window],av_data, c = 'black', marker = 'x', ms = 2, markevery=2,lw=lw)
        else:
            ax[0,0].semilogx(tlist[window:len(imbalance[idx])-window],av_data,label=r'{}'.format(gamma_n), c = cmap(cl), lw=lw)
            axins.semilogx(tlist[window:len(imbalance[idx])-window],av_data, c = cmap(cl), lw=lw)
        if gamma_n == '$2.5 \cdot 10^{-3}$':
            ypoint = av_data[tidx]+0.015
        idx += 1
    
    ax[0,0].vlines(x=tlist[tidx], ymin=-.05, ymax=ypoint, lw=lw, color = 'black', ls = '--')
    axins.vlines(x=tlist[tidx], ymin=-.05, ymax=ypoint, lw=lw, color = 'black', ls = '--')
        
    leg = ax[0,0].legend(title=r'$\gamma$',loc='lower left', bbox_to_anchor=(0.01, 0.01),fontsize=fs-1,title_fontsize=fs,frameon=False) 
    ax[0,0].text(.95,.9,r'(a)',fontsize=fs, horizontalalignment='center',
     verticalalignment='center', transform=ax[0,0].transAxes)
    ax[0,0].set_ylabel(r'$\mathcal{I}(\rho(t))$',fontsize=fs) 
    ax[0,0].set_xlabel(r'$t$',fontsize=fs)
    ax[0,0].tick_params(axis='both', labelsize=fs)
    ax[0,0].set_ylim([-0.05,1.05])
    ax[0,0].set_xlim(left=3e-2,right=4e1)
    ax[0,0].margins(0)    
    

    imbalance = []
    for N in [4, 6, 8, 10]:
        gamma_n0 = 0
        gamma_n1 = 3

        file = load_obj('../data/im_rho_data_N{}_dis_{}'.format(N,str(dis).replace('.','_')))['Im'][gamma_n0]
        av_data = []
        for ind in range(window,len(file)-window):
            av_data.append(np.mean(file[ind-window:ind+window+1]))                
        imb0 = av_data[tidx]
        
        file = load_obj('../data/im_rho_data_N{}_dis_{}'.format(N,str(dis).replace('.','_')))['Im'][gamma_n1]
        av_data = []
        for ind in range(window,len(file)-window):
            av_data.append(np.mean(file[ind-window:ind+window+1])) 
        imb1 = av_data[tidx]
        imbalance.append((imb1-imb0))

    popt, pcov = curve_fit(linear, [1./4.,1./6.,1./8.,1./10.], imbalance)
    ax[0,1].plot(np.linspace(0.,0.33,num=100), linear(np.linspace(0.,0.33,num=100), *popt), 'b--', lw=lw)

    ax[0,1].plot([1./4.,1./6.,1./8.,1./10.], imbalance,marker='.',markersize=5,lw=0,c='black')
    ax[0,1].text(.95,.9,r'(b)',fontsize=fs, horizontalalignment='center',
     verticalalignment='center', transform=ax[0,1].transAxes)
    ax[0,1].set_xlim(left=0.,right=0.301)
    ax[0,1].set_xlabel(r'$1/N$',fontsize=fs)
    ax[0,1].set_ylabel(r'$\Delta\mathcal{I}$',fontsize=fs)
    ax[0,1].tick_params(axis='both', labelsize=fs)    
    
    
    N = 6
  
    filename = '../data/im_rho_data_N{}_dis_{}'.format(str(N),str(dis).replace('.','_'))

    trajectories = load_obj(filename)['im'][0]
    av_data = []
    for ind in range(window,len(trajectories)-window):
       av_data.append(np.mean(trajectories[ind-window:ind+window+1]))
    
    ax[1,0].semilogx(tlist[window:len(trajectories)-window],av_data,label=r'{}'.format(gn), c='black', lw=lw)   

    glist = list(range(2,11,2)) + [20]

    for cl,gamma_n,gidx in zip(colours,glist,range(1,len(glist)+1)):
        gamma_n= round(float(gamma_n),4)

        filename = '../data/im_rho_data_N{}_dis_{}_MIC'.format(str(N),str(dis).replace('.','_'))
        trajectories = load_obj(filename)['im'][gidx]

        av_data = []
        for ind in range(window,len(trajectories)-window):
            av_data.append(np.mean(trajectories[ind-window:ind+window+1]))
        ax[1,1].semilogx((1./gamma_n**2)*tlist[window:len(trajectories)-window],av_data,label=r'{}'.format(gamma_n), c = cmap(cl), lw=lw)
        ax[1,0].semilogx(tlist[window:len(trajectories)-window],av_data,label=r'{}'.format(gamma_n), c = cmap(cl), lw=lw)

    ax[1,0].legend(title=r'$\gamma$',loc='lower left', bbox_to_anchor=(0.01, 0.01),fontsize=fs-1,title_fontsize=fs,frameon=False)    
    ax[1,0].text(.95,.9,r'(c)',fontsize=fs, horizontalalignment='center',
     verticalalignment='center', transform=ax[1,0].transAxes)
    ax[1,0].set_ylabel(r'$\mathcal{I}(\rho(t))$',fontsize=fs) 
    ax[1,0].set_xlabel(r'$t$',fontsize=fs)
    ax[1,0].tick_params(axis='both', labelsize=fs)
    ax[1,0].set_ylim([-0.05,1.05])
    ax[1,0].set_xlim(left=3e-2,right=4e1)
    ax[1,0].margins(0)    
    
    ax[1,1].legend(title=r'$\gamma$',loc='lower left', bbox_to_anchor=(0.01, 0.01),fontsize=fs-1,title_fontsize=fs,frameon=False)    
    ax[1,1].text(.95,.9,r'(d)',fontsize=fs, horizontalalignment='center',
     verticalalignment='center', transform=ax[1,1].transAxes)
    ax[1,1].set_ylabel(r'$\mathcal{I}(\rho(t))$',fontsize=fs) 

    ax[1,1].set_xlabel(r'$t/\gamma$',fontsize=fs)
    ax[1,1].tick_params(axis='both', labelsize=fs)
    ax[1,1].set_ylim([-0.05,1.05])
    ax[1,1].set_xlim(left=2e-3,right=3e0)
    ax[1,1].margins(0)  

    plt.savefig('Fig1.pdf')
    
def combined2():
    # Figure 2 plot
    
    def double_gaussian_fit(params):
        fit = double_gaussian(x_new,params)
        return fit - y_proc

    N = 6
    dis = 5.
    dis = round(dis,1)
    gamma = 0
    
 
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4), dpi=400.0, constrained_layout = True)
    
    cmap = cm.get_cmap('tab10')
    colours = np.linspace(-1,1,num=11)
    
    lw = 1.
    fs = 14

    cut = int(N/2)
    
    tnum = 200
    tlist = np.logspace(-4, 4, num=tnum)
    
    window = 10

    _, _, n = fermi_operators(N)
    total_N = sum(n)

    glist = [0.0, 0.0005, 0.005, 0.05, 0.5]
    glabel = ['$0.0$','$2.5 \cdot 10^{-7}$','$2.5 \cdot 10^{-5}$','$2.5 \cdot 10^{-3}$','$2.5 \cdot 10^{-1}$']

    for cl,gamma_n,gidx,glabel in zip(np.append([0],colours),glist,range(len(glist)),glabel):
        gamma_n= round(gamma_n,4)
        print("\nRun for gamma_n = ", gamma_n)
        print("\nRun for disorder = ", dis)
        
        
        filename = '../data/im_rho_data_N{}_dis_{}_scan'.format(str(N),str(dis).replace('.','_'))

        trajectories = load_obj(filename)['dms'][gidx]            
        trajectories_ln =[fermion_log_neg(dm, N, cut) for dm in trajectories]
        
        av_data = []
        for ind in range(window,len(trajectories_ln)-window):
           av_data.append(np.mean(trajectories_ln[ind-window:ind+window+1]))

        if gamma_n == 0.0:
            ax[0].semilogx(tlist[window:len(trajectories)-window],av_data,label=r'${}$'.format(glabel),c='black')
        else:
            ax[0].loglog(tlist[window:len(trajectories_ln)-window],av_data,label=r'{}'.format(glabel),c = cmap(cl)) 
   
    
    ax[0].legend(title=r'$\gamma$',loc='lower left', bbox_to_anchor=(0.1, 0.01),fontsize=fs-1,title_fontsize=fs,frameon=False) 
    ax[0].set_ylabel(r'$S_n(t)$',fontsize=fs) 
    ax[0].set_xlabel(r'$t$',fontsize=fs)
    ax[0].tick_params(axis='both', labelsize=fs)
    ax[0].set_xlim(left=8e-4,right=2e3)
    ax[0].set_ylim(top=.4)
    ax[0].margins(0)
    ax[0].text(.05,.9,r'(a)',fontsize=fs, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)


    tselect = [110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
    glist = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]
    glist += [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]
    glist += [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    glist += [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    for cl,indx in zip(np.append(colours,[0]),tselect):
        data = []
        y_new = []

        for gidx in range(len(glist):
            filename = '../data/im_rho_data_N{}_dis_{}_many'.format(str(N),str(dis).replace('.','_'))
            
            trajectories = load_obj(filename)['dms'][gidx]
            trajectories =[fermion_log_neg(dm, N, cut) for dm in trajectories]

            av_data = []
            for ind in range(window,len(trajectories)-window):
                av_data.append(np.mean(trajectories[ind-window:ind+window+1]))

            data.append(av_data[indx-window])
        window = 2
        av_data = []
        for ind in range(window,len(data)-window):
            av_data.append(np.mean(data[ind-window:ind+window+1]))
            
        y_proc = np.copy(av_data)[11:]
        x_new = np.linspace(np.log10(glist[window:len(glist)-window][0]), np.log10(glist[window:len(glist)-window][-1]), 32)[11:]
        fit = leastsq(double_gaussian_fit, [0.225, -4, 1.0, 0.20, -1.0, 1.0])
        x_new = np.linspace(np.log10(glist[window:len(glist)-window][0]), np.log10(glist[window:len(glist)-window][-1]), 32)
        parameters, covariance = curve_fit(linear, np.log10(glist[window:len(glist)-window])[:11], av_data[:11])
        y_new.append(list(linear(x_new[:11],parameters[0],parameters[1]))+list(double_gaussian(x_new[11:],fit[0])))
        
        ax[1].semilogx(np.array(glist[window:len(glist)-window])**2,av_data,label=r'$t={}$'.format(round(tlist[indx],2)),c = cmap(cl),marker='x',linestyle='')
        ax[1].semilogx(np.array([10**num for num in x_new])**2, y_new[0], c=cmap(cl),linestyle='--')
    
    ax[1].legend(loc='lower left',bbox_to_anchor= (0, 0),fontsize=fs-1,frameon=False) 
    ax[1].set_ylabel(r'$S_n(t)$',fontsize=fs)
    ax[1].set_xlabel(r'$\gamma$',fontsize=fs)
    ax[1].tick_params(axis='both', labelsize=fs)
    ax[1].margins(0)
    ax[1].set_ylim(top=0.25)
    ax[1].text(0.95,.9,r'(b)',fontsize=fs, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)

    plt.savefig('Fig2.pdf')
    
def combined3():
    # Figure 3 plot

    def double_gaussian_fit(params):
        fit = double_gaussian(x_new,params)
        return fit - y_proc

    N = 6
    dis = 5.
    dis = round(dis,1)
    gamma = 0

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4), dpi=400.0, constrained_layout = True)
    
    cmap = cm.get_cmap('tab10')
    colours = np.linspace(0,1,num=10)
    
    lw = 1.
    fs = 14

    cut = int(N/2)
    
    tnum = 200
    tlist = np.logspace(-4, 4, num=tnum)

    tselect = [110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]
    
    glist = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009]
    glist += [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]
    glist += [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    glist += [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    window = 10
   
    ad,a,n = fermi_operators(N)
    total_N = sum(n)

    indx = 125
 
    data2 = []   
    for gidx,gamma_n in range(len(glist))):
        
        fp = '../data/im_rho_data_N{}_dis_{}_many'.format(str(N),str(dis).replace('.','_'))
        dm = load_obj(fp)['dms'][gidx][indx-window:indx+window+1]
        dm = (1./len(dm))*sum(dm)
        data2.append([((dm*ad[0]*a[j]).tr()).real for j in range(1,N)])
    
    data = []
    data_error = []
    y_new = []
    window = 2
    runindex = 0
    for lst in np.array(data2).T:
        data.append([np.mean(lst[ind-window:ind+window+1]) for ind in range(window,len(lst)-window)])
        data_error.append([np.std(lst[ind-window:ind+window+1]) for ind in range(window,len(lst)-window)])

        x_new = np.linspace(np.log10(glist[window:len(glist)-window][0]), np.log10(glist[window:len(glist)-window][-1]), 32)

        if runindex != 0:
            parameters, covariance = curve_fit(quartic, np.log10(glist[window:len(glist)-window]), data[-1])
            y_new.append(quartic(x_new,parameters[0],parameters[1],parameters[2],parameters[3],parameters[4]))

        if runindex==0:
            y_proc = np.copy(data[-1])[11:]
            x_new = np.linspace(np.log10(glist[window:len(glist)-window][0]), np.log10(glist[window:len(glist)-window][-1]), 32)[11:]
            fit = leastsq(double_gaussian_fit, [0.04, -4, 1.0, 0.04, -1.0, 1.0])
            x_new = np.linspace(np.log10(glist[window:len(glist)-window][0]), np.log10(glist[window:len(glist)-window][-1]), 32)
            parameters, covariance = curve_fit(linear, np.log10(glist[window:len(glist)-window])[:11], data[-1][:11])
            y_new.append(list(linear(x_new[:11],parameters[0],parameters[1]))+list(double_gaussian(x_new[11:],fit[0])))

        runindex += 1 

    markerset = ['x','+','.','.',"."]
    for point,dataj,errorj,cl,ynew,markerpoint in zip(range(2,7),data,data_error,colours,y_new,markerset):
        ax[0].semilogx([item**2 for item in glist[window:len(glist)-window]],dataj,label=r'$\langle c_1^\dagger c_{}\rangle$'.format(point),c=cmap(cl),linestyle = '',marker=markerpoint)
        ax[0].semilogx([10**(2*num) for num in x_new], ynew, c=cmap(cl),linestyle='--')

    ax[0].set_xlabel(r'$\gamma$',fontsize=fs)
    ax[0].margins(0)
    ax[0].tick_params(axis='both', labelsize=fs)
    ax[0].set_ylim(bottom=-0.02,top=0.055)
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.01, 0.85),fontsize=fs-1)
    ax[0].text(.05,.9,r'(a)',fontsize=fs, horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)
    ax[0].tick_params(axis='both', labelsize=fs)
 
    
    window = 10  
    
    glist = [0.0, 0.0005, 0.005, 0.05, 0.5, 5.0]
    glabel = ['0.0', '$2.5 \cdot 10^{-7}$', '$2.5 \cdot 10^{-5}$', '$2.5 \cdot 10^{-3}$', '$2.5 \cdot 10^{-1}$']

    for cl,gamma_n,gidx,gl in zip(colours,glist,range(len(glist)),glabel):
        gamma_n= round(gamma_n,4)
        print("\nRun for gamma_n = ", gamma_n)
        print("\nRun for disorder = ", dis)

        filename = '../data/im_rho_data_N{}_dis_{}_scan'.format(str(N),str(dis).replace('.','_'))

        trajectories = load_obj(filename)['rho'][gidx]
        trajectories = [ent_entropy(dm, cut) for dm in trajectories]

        av_data = []
        for ind in range(window,len(trajectories)-window):
           av_data.append(np.mean(trajectories[ind-window:ind+window+1]))

        ax[1].semilogx(tlist[window:len(trajectories)-window],av_data,label=r'${}$'.format(gl),c=cmap(cl))  

    ax[1].legend(title=r'$\gamma$',loc='lower left', bbox_to_anchor=(0.01, 0.1),fontsize=fs-1,title_fontsize=fs,frameon=False) 
    ax[1].set_ylabel(r'$S(t)$',fontsize=fs) 
    ax[1].set_xlabel(r'$t$',fontsize=fs)
    ax[1].tick_params(axis='both', labelsize=fs)
    ax[1].set_xlim(left=8e-4,right=2e3)
    ax[1].set_ylim(top=2.1)   
    ax[1].text(.05,.9,r'(b)',fontsize=fs, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
    ax[1].margins(0)  

    plt.savefig('Fig3.pdf')


def __main__():
    plot_script()
    #combined1()
    #combined2()
    #combined3()
   
    
if __name__ == '__main__':
    __main__()
