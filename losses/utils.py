import torch
import numpy as np
import warnings
from typing import Union

from scipy.linalg import logm
from copy import deepcopy as copy

import random

warnings.filterwarnings("ignore")

EPSILON = 1E-20
EPS = 1.0E-40

def triu(X):
    # Sum of strictly upper triangular part
    return X.triu(diagonal=1).sum()

def atleast_epsilon(X, eps=EPSILON):
    """
    Ensure that all elements are >= `eps`.

    :param X: Input elements
    :type X: th.Tensor
    :param eps: epsilon
    :type eps: float
    :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
    :rtype: th.Tensor
    """
    return torch.where(X < eps, X.new_tensor(eps), X)

    
def p_dist(x, y):
    # x, y should be with the same flatten(1) dimensional
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    x_norm2 = torch.sum(x**2, -1).reshape((-1, 1))
    y_norm2 = torch.sum(y**2, -1).reshape((1, -1))
    dist = x_norm2 + y_norm2 - 2*torch.mm(x, y.t())
    return torch.where(dist<0.0, torch.zeros(1).to(dist.device), dist)

def calculate_gram_DG(domains:Union[list,tuple,torch.Tensor], 
                      sigmas: Union[torch.Tensor,np.ndarray,float]=None,
                      **kwargs)->torch.Tensor:
    dn, bn = len(domains), len(domains[0])
    PD = torch.zeros([dn,dn,bn,bn], device=domains[0].device)
    out_sigmas=[]
    for t in range(dn):
        for k in range(t+1):
            pd_tk = p_dist(domains[t], domains[k])
            if sigmas==None:
                sigma = (0.15*pd_tk[np.triu_indices(len(pd_tk))].median())
            else:
                sigma = sigmas[t,k] if type(sigmas) in [torch.Tensor,np.ndarray] else sigmas
            PD[t,k,...] = PD[k,t,...] = -pd_tk/sigma
            out_sigmas.append(sigma)
    
    return PD.exp(), out_sigmas

def  calculate_gram_mat(*data, sigma=1):
    if len(data) == 1:
        x, y = data[0], data[0]
    elif len(data) == 2:
        x, y = data[0], data[1]
    else:
        print('size of input not match')
        return []
    dist = p_dist(x, y)    
    # dist /= torch.max(dist+EPSILON)
    # dist /= torch.trace(dist)
    return torch.exp(-dist / sigma)

def reyi_entropy(x, sigma, alpha=1.001):
    k = calculate_gram_mat(x, sigma=sigma)
    k = k/torch.trace(k)
    # eigv = torch.abs(torch.linalg.eigh(k)[0])
    try:
        eigv = torch.abs(torch.linalg.eigh(k)[0])
    except:
        eigv = torch.diag(torch.eye(k.shape[0]))
    entropy = (1/(1-alpha))*torch.log2((eigv**alpha).sum(-1))
    return entropy

def joint_entropy(x, y, s_x, s_y, alpha=1.001):
    x = calculate_gram_mat(x, sigma=s_x)
    y = calculate_gram_mat(y, sigma=s_y)
    k = torch.mul(x, y)
    k = k/torch.trace(k)
    try:
        eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    except:
        eigv = torch.diag(torch.eye(k.shape[0]))
        
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def mi(x, y, s_x, s_y):

    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx+Hy-Hxy
    Ixy = Ixy/(torch.max(Hx,Hy)+1e-16)

    return Ixy

def IsPositiveDefinite(A):
    """ Checks if the matrix A is positive semidefinite by checking that all eigenvalues are >=0
    A : np.array
    """
    if isinstance(A, torch.Tensor):
        A = A.numpy()   
    return np.all(np.linalg.eigvals(A) >= 0)

# conditional divergency discrepency
def SampledCenteredCorrentropy(Sample, Sigma=1):
    n = Sample.shape[0]
    m = Sample.shape[1]
    C = torch.zeros(m,m)
    for i in range(m):
        for j in range(i+1):
            gram = calculate_gram_mat(Sample[:,i], Sample[:,j], sigma=Sigma)
            C[i,j] = C[j,i] = torch.trace(gram)/n  -   gram.sum()/(n**2)    
    # if not IsPositiveDefinite(C.numpy()) or torch.abs(C.det())<1.0e-32:
    #     eigv, _ = torch.linalg.eigh(C)
    #     eigv, _ = torch.abs(eigv).sort(0)
    #     # print(eigv)
    #     index = torch.nonzero(eigv, as_tuple=True)
    #     C += torch.eye(C.shape[0])*eigv[index[0]]*1.0e-5
    #     # print(IsPositiveDefinite(C.numpy()))
    # # print('C = {}'.format(C.numpy()))
    return C
    
# Bregman Divengency
def BregmanDivengency(T1, T2, divFuncType):
    # if not IsPositiveDefinite(T1) or not IsPositiveDefinite(T2):
    #     print('Sampled_Center_Correntropy is not positive nefinite')

    # if  divFuncType == 'Von_Neumann':       
    #     d = torch.abs( torch.trace( T1.dot( torch.tensor(logm(T1.numpy())-logm(T2.numpy()) )) - T1 + T2 ) )
    # elif  divFuncType == 'LogDet':
    #     d  = torch.trace(torch.mm(torch.linalg.pinv(T2),T1)) + \
    #          torch.log(torch.linalg.det(T2)/torch.linalg.det(T1)) - T1.shape[0]
    
    if  divFuncType == 'Von_Neumann':       
        d = np.abs( np.trace( T1.dot( logm(T1.numpy())-logm(T2.numpy()) ) - T1 + T2 ) )
    elif  divFuncType == 'LogDet':
        d  = np.trace(np.matmul(np.linalg.pinv(T2),T1.numpy())) + \
             np.log(np.linalg.det(T2.numpy())/np.linalg.det(T1.numpy())+EPSILON) - T1.shape[0]
        
    return d

def SymmetricalBregmanDivengency(T1, T2, divFuncType): 
    # if  divFuncType == 'Von_Neumann':       
    #     d = torch.trace( tensor((logm(T1.div(T2).numpy())).dot(T1-T2)) )
    # elif  divFuncType == 'LogDet':        
    #     d = torch.trace( torch.mm(torch.linalg.pinv(T1),T2) + torch.mm(torch.linalg.pinv(T2),T1)) - 2.0*T1.shape[0] 
    if  divFuncType == 'Von_Neumann':       
        d = np.trace( (logm(T1.div(T2).numpy())).dot(T1-T2) )
    elif  divFuncType == 'LogDet':       
        d = np.trace( np.matmul(np.linalg.pinv(T1),T2) + np.matmul(np.linalg.pinv(T2),T1)) - 2.0*T1.shape[0] 
    return 0.5*d

# Conditional Distribution Discrepency with correntropy
def cdd(input1, input2, sigma, divFuncType):
    # default the last column output
    Cxy1  = SampledCenteredCorrentropy(input1, sigma)
    Cxy2  = SampledCenteredCorrentropy(input2, sigma)
    Cx1   = Cxy1[:-1,:-1]
    Cx2   = Cxy2[:-1,:-1]

    d1 = SymmetricalBregmanDivengency(Cxy1, Cxy2, divFuncType) 
    d2 = SymmetricalBregmanDivengency(Cx1, Cx2, divFuncType)
    # d1 = (BregmanDivengency(Cxy1, Cxy2, divFuncType) + BregmanDivengency(Cxy2, Cxy1, divFuncType))*0.5
    # d2 = (BregmanDivengency(Cx1, Cx2, divFuncType)   + BregmanDivengency(Cx2, Cx1, divFuncType))*0.5
    d  = (d1-d2)
    # if d.is_cuda:
    #     d = d.cpu()
    return d, d2, Cxy1, Cxy2
    
def idd(input1, input2, sigma, divFuncType):

    C1  = SampledCenteredCorrentropy(input1, sigma)
    C2  = SampledCenteredCorrentropy(input2, sigma)

    d = SymmetricalBregmanDivengency(C1, C2, divFuncType)
    # if d.is_cuda:
    #     d = d.cpu()
    return d

def mmd(input1, input2, sigma=1):
    m = input1.shape[0]
    n = input2.shape[0]

    kxx = calculate_gram_mat(input1, input1, sigma=sigma)
    kyy = calculate_gram_mat(input2, input2, sigma=sigma)
    kxy = calculate_gram_mat(input1, input2, sigma=sigma)

    d  = (kxx.sum((0,1)) - kxx.trace())/(m*(m-1))
    d += (kyy.sum((0,1)) - kyy.trace())/(n*(n-1))
    d -=  kxy.sum((0,1))*2/(n*m) + torch.tensor(1e-6, device=input1.device)

    # if d.is_cuda:
    #     d = d.cpu()
    return d

def mcc(error, sigma=1, reduction: str='mean'):
    # return ((-(error/sigma)**2).exp()/(sigma*torch.tensor(2*np.pi).sqrt())).mean()
    crit = ((-(error/sigma)**2).exp())
    if reduction in ['mean', 'max', 'min']:
        return getattr(crit, reduction)()
    else:
        return crit  
#     return metrics if len(metrics)>1 else metrics.values[0]
def autocorrelation(x, y, EPSILON=1e-10):
    x, y = x.view(-1), y.view(-1)
    xm = x.sub(x.mean().expand_as(x))
    ym = y.sub(y.mean().expand_as(y))
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)# + EPSILON
    # r_den = xm.norm(2) * ym.norm(2)# + EPSILON
    # r_den = ((xm**2).mean() * (ym**2).mean()).sqrt()# + EPSILON
    r_val = r_num / r_den
    return r_val
    
def selfcorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array(
        [(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result

def consist_loss(A, B):
    K = A.shape[-1]

    loss_consist = 0
    for k in range(1,K+1):
        As1 = A[:,:k]
        Bs1 = B[:k,:]
        As2 = A[:k,:]
        Bs2 = B[:,:k]

        Ik = torch.eye(k,device=A.device).float()
        loss_consist = loss_consist + \
                         (torch.sum((torch.mm(Bs1, As1) - Ik)**2) + \
                          torch.sum((torch.mm(As2, Bs2) - Ik)**2) ) / (2.0*k)
    return loss_consist

def seq_decorrelation(inputs, EPSILON: float = 1.0e-10):          
    seq_len, feat_dim = tuple(inputs.shape[1:])
    if feat_dim>1 and seq_len>1:  
        temp = inputs.view(-1, feat_dim)
        CovMat_abs = (temp.T.matmul(temp)).abs()
        abs_CovMat = (temp.abs().T.matmul(temp.abs()))
        loss = (CovMat_abs/(abs_CovMat+torch.tensor(EPSILON, device=inputs.device))).mean() - 1.0/(feat_dim-1)
    else:
        loss = 0.0
    return loss

def PFIB(current, past, future, beta: float=1.0, kernelsize: list=[3, 1, 3] ):    
    sp, sc, sf = tuple(kernelsize)
    Ipc = mi(past, current, sp, sc)
    Ifc = mi(future, current, sf, sc)
    pfib = Ipc - beta*Ifc

    return pfib, Ipc, Ifc

def hsic(x, y, s_x, s_y):
    m = x.shape[0]  # batch size
    K = calculate_gram_mat(x, sigma=s_x)
    L = calculate_gram_mat(y, sigma=s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m, m))
    H = H.float().to(x.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H))))/((m-1)**2)
    return HSIC

def multi_hsic(variates: torch.Tensor, kernels):
    if len(variates.shape)<2:
        warnings("please at least input 2 sample sets with no less than 2 samples in every set")
        return
    elif len(variates.shape)==2:
        variates = variates.reshape(variates.shape[0], variates.shape[1], 1)
    
    if len(variates.shape)>3:
        warnings("cannot calculate hsic for samplesets with features of more than two dimensions")
        return
        
    num, dim = variates.shape[0], variates.shape[1]
    kernels = np.vstack(tuple(kernels)) if isinstance(kernels, list) else kernels
    if not isinstance(kernels, torch.Tensor):
        kernels = torch.tensor(kernels).squeeze()
    if not dim == kernels.shape[0]:
        warnings("quatity of kernels should be the same as of variates")
        kernels = kernels[:dim] if kernels.shpe[0]>dim else torch.cat((kernels,torch.ones(kernels.shpe[0]-dim)),0)
    if dim==0 or num<=1:
        warnings("please at least input 2 sample sets with no less than 2 samples in every set")
        return
    
    H = torch.eye(num) - 1.0/num * torch.ones((num, num))
    H = H.float().to(variates.device)
    K_sum, self_hsic = torch.zeros((num, num),device=variates.device), torch.zeros(1,device=variates.device)
    for i in range(dim):
        K = calculate_gram_mat(variates[:,i,:].squeeze(), sigma=kernels[i])
        self_hsic  = self_hsic + torch.trace(torch.mm(K, torch.mm(H, torch.mm(K, H)))) 
        K_sum = K_sum + K   
    cross_hsic = torch.trace(torch.mm(K_sum, torch.mm(H, torch.mm(K_sum, H))))
    multi_hsic = (cross_hsic-self_hsic)/dim/(num-1)**2
    return multi_hsic
   
def integ_density_2(set1, set2, sigma=1.0):
    distance = p_dist(set1, set2)
    gram = torch.exp(-distance / sigma)
    return gram.sum()/torch.tensor(distance.shape).prod()

def csd(sampleSet1, sampleSet2, weight1, weight2, sigma=1.0):
    n1, n2 = sampleSet1.shape[0], sampleSet2.shape[0]
    sampleSet1, sampleSet2 = sampleSet1.view(n1, -1), sampleSet2.view(n2, -1)
    if not sampleSet1.shape[1] == sampleSet2.shape[1]:
        Warning("dimetions of the two input sets isn't consistent")
    
    p1_square = integ_density_2(sampleSet1, sampleSet1, sigma)
    p2_square = integ_density_2(sampleSet2, sampleSet2, sigma)
    p12_cross = integ_density_2(sampleSet1, sampleSet2, sigma)

    cs = -torch.log(p1_square*weight1**2+2*p12_cross*weight1*weight2+p2_square*weight2**2) \
        + torch.log(p1_square)+torch.log(p2_square)
    return cs

def gjrd(sampleSet: list, weights: list, sigma=1.0):
    n_cluste = len(sampleSet)
    # n_sample = [set.shape[0] for set in sampleSet]
    
    p = torch.zeros(n_cluste, n_cluste)
    for i in range(n_cluste):
        for j in range(i+1):
            p[i,j]=p[j,i]=integ_density_2(sampleSet[i], sampleSet[j], sigma)*weights[i]*weights[j]

    gjrd = -torch.log(p.sum()) + torch.log(p.trace())

    return gjrd     

def gcsd_cluster(cluster_assignment, kernel_matrix):
    G, A = kernel_matrix, cluster_assignment
    m = cluster_assignment.shape[1]
    
    k=m-1
    Ak = atleast_epsilon(A**k, EPS)
    Ga = atleast_epsilon(G.matmul(A), EPS)
    Gak = atleast_epsilon(Ga**k, EPS)   

    # # version_1
    # Ap = row_prod_except(A)
    # cross_entropy = -atleast_epsilon((Ga*Ap).sum()/m, EPS).log() # - m*torch.log(torch.tensor(n))
    # power_entropy = -atleast_epsilon((Ak.T.matmul(Gak)).diag(), EPS).log().mean() # - m*torch.log(torch.tensor(n))
    
    ## version_2
    Gap = row_prod_except(Ga)
    cross_entropy = -atleast_epsilon((Ak*Gap).sum()/m, EPS).log() # - m*torch.log(torch.tensor(n))
    power_entropy = -atleast_epsilon((Ak.T.matmul(Gak)).diag(), EPS).log().mean() # - m*torch.log(torch.tensor(n))
    
    return torch.exp(-(cross_entropy - power_entropy)/k)

def gjrd_cluster(cluster_assignment, kernel_matrix, order=2):    
    G, A = kernel_matrix, cluster_assignment
    n, m = tuple(cluster_assignment.shape)
    k = order-1
    if k==1:
        AGA = atleast_epsilon(A.T.matmul(G).matmul(A)/n**2, eps=EPSILON)
        cross_entropy = -AGA.mean().log()
        power_entropy = -AGA.diag().log().mean()

    else:
        AkT = atleast_epsilon(A**k, EPS).T
        Gak = atleast_epsilon((G.matmul(A))**k, EPS)        

        cross_entropy = -atleast_epsilon(((G.sum(1)/m)**k).sum()/m, EPS).log()
        power_entropy = -atleast_epsilon(AkT.matmul(Gak).diag(), EPS).log().mean()

    return torch.exp(-(cross_entropy-power_entropy)/k)
    # return torch.exp(-(cross_entropy-power_entropy))

def gcsd_every_2_cluster(cluster_assignment, kernel_matrix):
    A, G = cluster_assignment, kernel_matrix
    nom = A.T @ G @ A
    dnom = (nom.diag().view(-1,1) @ nom.diag().view(1,-1)).sqrt()

    m = A.shape[1]
    d = triu(atleast_epsilon(nom) / atleast_epsilon(dnom)) * 2/(m*(m-1))
    
    # return -d.log()
    return d

def row_prod_except(A):    
    res = torch.zeros_like(A)
    for j in range(res.shape[1]):
        if j==0:
            res[:,j] = A[:,1:].prod(1)
        elif j==A.shape[1]-1:
            res[:,j] = A[:,:-1].prod(1)
        else:
            res[:,j] = torch.cat([A[:,:j], A[:,j+1:]],1).prod(1)

    return atleast_epsilon(res, EPS)
    
def get_kernelsize(features: torch.Tensor, selected_param: Union[int, float]=0.15, select_type: str='meadian'):
    ### estimating kernelsize with data with the rule-of-thumb
    features = torch.flatten(features, 1).detach()
    # if features.shape[0]>300:
    #     idx = [i for i in range(0, features.shape[0])]
    #     random.shuffle(idx)
    #     features = features[idx[:300],:]
    k_features = p_dist(features, features)
    
    if select_type=='min':
        kernelsize = k_features.sort(-1)[0][:, :int(selected_param)].mean()
    elif select_type=='max':
        kernelsize = k_features.sort(-1)[0][:, int(selected_param):].mean()
    elif select_type=='meadian':
        kernelsize = selected_param*k_features.view(-1).median()
    else:
        kernelsize = torch.tensor(1.0, device=features.device)
    
    if kernelsize<EPSILON:
        kernelsize = torch.tensor(EPSILON, device=features.device)

    return kernelsize

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def gcsd_inter_domain(gram_mat) -> torch.tensor:
    # GCSD-Inter-Domain discrepancy
    dn = gram_mat.shape[0]
    tri_d = torch.arange(dn)

    Gs=gram_mat.sum(-1)
    Gp=Gs[tri_d,tri_d,...]
    Gc=Gs.prod(1)/Gp

    cross_entropy=-atleast_epsilon(Gc,         EPS).sum(-1).mean().log()    
    power_entropy=-atleast_epsilon(Gp**(dn-1), EPS).sum(-1).log().mean()

    return (cross_entropy - power_entropy)/dn

def gcsd_class_based_inter_domain(gram_mat, assignment_mat) -> torch.tensor:
    # GCSD-class_based_Inter-Domain discrepancy
    # gram_mat: s,s,bn,d
    # assignment_mat: s,bn,c

    dn = gram_mat.shape[0]
    tri_d = torch.arange(dn)

    Ap = assignment_mat**(dn-1)
    GA=gram_mat.matmul(assignment_mat)   
    GAt=GA[tri_d,tri_d,...]
    
    Gc=(GA.prod(1)/GAt)*Ap
    Gp=(GAt**(dn-1))*Ap

    cross_entropy=-atleast_epsilon(Gc, EPS).sum(1).mean().log()
    
    power_entropy=-atleast_epsilon(Gp, EPS).sum(1).log().mean()

    return (cross_entropy - power_entropy)/dn


def gjrd_inter_domain(gram_mat, order:float=None) -> torch.tensor:
    # GJRD-Inter-Domain discrepancy
    if order==None:
        order=2.0
    
    dn = gram_mat.shape[0]
    tri_d = torch.arange(dn)

    Gc=gram_mat.sum([1,-1])**(order-1)    
    cross_entropy=-atleast_epsilon(Gc, EPS).sum().log() \
                  + order*torch.tensor(dn,device=gram_mat.device).log()
    
    Gp=(gram_mat[tri_d,tri_d,...].sum(-1))**(order-1)
    power_entropy=-atleast_epsilon(Gp, EPS).sum(1).log().mean()

    return (cross_entropy - power_entropy)/(order-1)

    
def gjrd_class_based_inter_domain(gram_mat, assignment_mat, order:float=None) -> torch.tensor:
    if order==None:
        order=2.0
    dn = gram_mat.shape[0]
    tri_d = torch.arange(dn)

    Ap = assignment_mat**(dn-1)
    GA=gram_mat.matmul(assignment_mat)  

    Gc = (GA.sum(1))**(order-1)*(Ap)
    cross_entropy=-atleast_epsilon(Gc, EPS).sum(0).log().mean() \
                  + order*torch.tensor(dn,device=gram_mat.device).log()
    
    Gp=(GA[tri_d,tri_d,...]**(order-1))*(Ap)
    power_entropy=-atleast_epsilon(Gp, EPS).log().mean()
    return (cross_entropy - power_entropy)/(order-1)