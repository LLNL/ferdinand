#! /usr/bin/env python3

import os,math,numpy,cmath
import sys
from CoulCF import cf1,cf2,csigma,Pole_Shifts
from pqu import PQU as PQUModule

import fudge.sums as sumsModule
import fudge.styles as stylesModule
import fudge.reactionData.crossSection as crossSectionModule
import fudge.productData.distributions as distributionsModule
import fudge.resonances.resolved as resolvedResonanceModule
from PoPs.groups.misc import *

def nuclIDs (nucl):
    datas = chemicalElementALevelIDsAndAnti(nucl)
    if datas[1] is not None:
        return datas[1]+str(datas[2]),datas[3]
    else:
        return datas[0],0

def quickName(p,t):     #   (He4,Be11_e3) -> a3
    ln = lightnuclei.get(p,p)
    tnucl,tlevel = nuclIDs(t)
    return(ln + str(tlevel) if tlevel>0 else ln)
    
DBLE = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32

# import tensorflow as tf
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
lpd = tf.config.experimental.list_physical_devices('GPU')
print(lpd)
ngpu = len(lpd)
print("\nNum GPUs Available: ", ngpu)


hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 3.1415926536
rsqr4pi = 1.0/(4*pi)**0.5
lightnuclei = {'n':'n', 'H1':'p', 'H2':'d', 'H3':'t', 'He3':'h', 'He4':'a', 'photon':'g'}



@tf.function
def R2T_transformsTF(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans):
# Now do TF:
    GL = tf.expand_dims(g_poles,2);                   #  print('GL',GL.dtype,GL.get_shape())
    GR = tf.expand_dims(g_poles,3);                   #  print('GR',GR.dtype,GR.get_shape())

    GG  = GL * GR;                                    #  print('GG',GG.dtype,GG.get_shape())
    GGe = tf.expand_dims(GG,0)                            # same for all scattering energies  

    POLES = tf.reshape(E_poles, [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
    SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies

    RPARTS = GGe / (POLES - SCAT);   #  print('RPARTS',RPARTS.dtype,RPARTS.get_shape())

    RMATC = tf.reduce_sum(RPARTS,2)  # sum over poles
  #  print('RMATC',RMATC.dtype,RMATC.get_shape())
  #  print('L_diag',type(L_diag),L_diag.shape)

    C_mat = tf.eye(n_chans, dtype=CMPLX) - RMATC * tf.expand_dims(L_diag,2);              #  print('C_mat',C_mat.dtype,C_mat.get_shape())
    
    D_mat = tf.linalg.solve(C_mat,RMATC);                                                 #  print('D_mat',D_mat.dtype,D_mat.get_shape())

#    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
#  T=I-S
    T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
    
# multiply left and right by Coulomb phases:
#   T_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)
    
    return(RMATC,T_mat)
    
@tf.function
def LM2T_transformsTF(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, DiagonalOnly, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles):
# Use Level Matrix A to get T=1-S:
#     print('g_poles',g_poles.dtype,g_poles.get_shape())
    GL = tf.reshape(g_poles,[1,n_jsets,n_poles,1,n_chans]) #; print('GL',GL.dtype,GL.get_shape())
    GR = tf.reshape(g_poles,[1,n_jsets,1,n_poles,n_chans]) #; print('GR',GR.dtype,GR.get_shape())
    LDIAG = tf.reshape(L_diag,[-1,n_jsets,1,1,n_chans]) #; print('LDIAG',LDIAG.dtype,LDIAG.get_shape())
    GLG = tf.reduce_sum( GL * LDIAG * GR , 4)    # giving [ie,J,n',ncd Rf]
    Z = tf.constant(0.0, dtype=DBLE)
    if brune:   # add extra terms to GLG
        SE_poles = S_poles + tf.expand_dims(tf.math.real(E_poles)-EO_poles,2) * dSdE_poles
        POLES_L = tf.reshape(E_poles, [1,n_jsets,n_poles,1,1])  # same for all energies and channel matrix
        POLES_R = tf.reshape(E_poles, [1,n_jsets,1,n_poles,1])  # same for all energies and channel matrix
        SHIFT_L = tf.reshape(SE_poles, [1,n_jsets,n_poles,1,n_chans] ) # [J,n,c] >  [1,J,n,1,c]
        SHIFT_R = tf.reshape(SE_poles, [1,n_jsets,1,n_poles,n_chans] ) # [J,n,c] >  [1,J,1,n,c]
        SCAT  = tf.reshape(E_scat,  [-1,1,1,1,1])             # vary only for scattering energies
#         NUM = SHIFT_L * (SCAT - POLES_R) - SHIFT_R * (SCAT - POLES_L)  # expect [ie,J,n',n,c]
        NUM = tf.complex(SHIFT_L,Z) * (SCAT - POLES_R) - tf.complex(SHIFT_R,Z) * (SCAT - POLES_L)  # expect [ie,J,n',n,c]
#         print('NUM',NUM.dtype,NUM.get_shape()); tf.print(NUM, summarize=-1 )
        DEN = POLES_L - POLES_R
        W_offdiag = tf.math.divide_no_nan( NUM , DEN )  
        W_diag    = tf.reshape( tf.eye(n_poles, dtype=CMPLX), [1,1,n_poles,n_poles,1]) * tf.complex(SHIFT_R,Z) 
        W = W_diag + W_offdiag
        GLG = GLG - tf.reduce_sum( GL * W * GR , 4)

    POLES = tf.reshape(E_poles, [1,n_jsets,n_poles,1])  # same for all energies and channel matrix
    SCAT  = tf.reshape(E_scat,  [-1,1,1,1])             # vary only for scattering energies
    Ainv_mat = tf.eye(n_poles, dtype=CMPLX) * (POLES - SCAT) - GLG    # print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
#     print('GLG',GLG.dtype,GLG.get_shape())
#     tf.print(GLG, summarize=-1 )
#     print('Ainv_mat',Ainv_mat.dtype,Ainv_mat.get_shape())
#     tf.print(Ainv_mat, summarize=-1 )
    if DiagonalOnly is None:
        A_mat = tf.linalg.inv(Ainv_mat)                                 # full inverse
        D_mat = tf.matmul( g_poles, tf.matmul( A_mat, g_poles) , transpose_a=True)     # print('D_mat',D_mat.dtype,D_mat.get_shape())
    elif DiagonalOnly[0]=='M':
        A_mat = tf.linalg.diag( 1.0/ tf.linalg.diag_part(Ainv_mat) )    #  invert just diagonal (coherent)
        D_mat = tf.matmul( g_poles, tf.matmul( A_mat, g_poles) , transpose_a=True)     # print('D_mat',D_mat.dtype,D_mat.get_shape())
    else:
        A_mat = tf.linalg.diag( tf.complex( 1.0/tf.abs(tf.linalg.diag_part(Ainv_mat))**2 , Z) )   #  invert just diagonal (incoherent)
        D_mat = tf.sqrt( tf.matmul( g_poles**2, tf.matmul( A_mat, g_poles**2) , transpose_a=True) )     # print('D_mat',D_mat.dtype,D_mat.get_shape())
    

#    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
#  T=I-S
    T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
    
# multiply left and right by Coulomb phases:
#   T_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)

    return(T_mat)

  
@tf.function
def T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs):
                    
    Tmod2 = tf.math.real(  T_mat * tf.math.conj(T_mat) )   # ie,jset,a1,a2
    T_diag = tf.linalg.diag_part(T_mat)

# sum of Jpi sets:
    G_fac = tf.reshape(gfac, [-1,n_jsets,1,n_chans])
    XS_mat = Tmod2 * G_fac                          # ie,jset,a1,a2   
  #  print('XS_mat',XS_mat.dtype,XS_mat.get_shape())
    
    
    G_fact = tf.reshape(gfac, [-1,n_jsets,n_chans])
    TOT_mat  = tf.math.real(T_diag)   #  ie,jset,a  for  1 - Re(S) = Re(1-S) = Re(T)
    XS_tot  = TOT_mat * G_fact                           #  ie,jset,a
    p_mask1_in = tf.reshape(p_mask, [-1,npairs,n_jsets,n_chans] )   # convert pair,jset,a to  ie,pair,jset,a
    XSp_tot = 2. *  tf.reduce_sum( tf.expand_dims(XS_tot,1) * p_mask1_in , [2,3])     # convert ie,pair,jset,a to ie,pair by summing over jset,a

    S_diag = tf.ones(n_chans, dtype=CMPLX) - T_diag         # S = 1 - T
    REAC_mat = tf.ones(n_chans, dtype=DBLE) -  tf.math.real( S_diag * tf.math.conj(S_diag) )  #  ie,jset,a  for 1 - |S|^2
    XS_reac  = REAC_mat * G_fact                           #  ie,jset,a
    XSp_reac = tf.reduce_sum( tf.expand_dims(XS_reac,1) * p_mask1_in , [2,3])     # convert ie,pair,jset,a to ie,pair by summing over jset,a

    
    p_mask_in = tf.reshape(p_mask,[1,1,npairs,n_jsets,1,n_chans])   # ; print('p_mask_in',p_mask_in.get_shape())   # 1,1,pin,jset,1,cin
    p_mask_out =tf.reshape(p_mask,[1,npairs,1,n_jsets,n_chans,1])   # ; print('p_mask_out',p_mask_out.get_shape()) # 1,pout,1,jset,cout,1
    
    XS_ext  = tf.reshape(XS_mat, [-1,1,1,n_jsets,n_chans,n_chans] ) # ; print('XS_ext',XS_ext.get_shape())
    XS_cpio =  XS_ext * p_mask_in * p_mask_out                      # ; print('XS_cpio',XS_cpio.get_shape())
    XSp_mat  = tf.reduce_sum(XS_cpio,[-3,-2,-1] )               # sum over jset,cout,cin, leaving ie,pout,pin
                            
    XSp_cap = XSp_tot - tf.reduce_sum(XSp_mat,1)  # total - sum of xsecs(pout)
#     return(XSp_mat,XS_tot,XS_ext,XS_cpio,p_mask_in * p_mask_out ) 
    return(XSp_mat,XSp_tot,XSp_cap,XSp_reac) 


def generateEnergyGrid(energies,widths, lowBound, highBound, stride=1):
    """ Create an initial energy grid by merging a rough mesh for the entire region (~10 points / decade)
    with a denser grid around each resonance. For the denser grid, multiply the total resonance width by
    the 'resonancePos' array defined below. """
    thresholds = []
    # ignore negative resonances
    for lidx in range(len(energies)):
        if energies[lidx] > 0: break
    energies = energies[lidx:]
    widths = widths[lidx:]
    # generate grid for a single peak, should be good to 1% using linear interpolation using default stride
    resonancePos = numpy.array([
        5.000e-04, 1.000e-03, 2.000e-03, 3.000e-03, 4.000e-03, 5.000e-03, 6.000e-03, 7.000e-03, 8.000e-03, 9.000e-03, 1.000e-02, 2.000e-02,
        3.000e-02, 4.000e-02, 5.000e-02, 6.000e-02, 7.000e-02, 8.000e-02, 9.000e-02, 1.000e-01, 1.100e-01, 1.200e-01, 1.300e-01, 1.400e-01,
        1.500e-01, 1.600e-01, 1.700e-01, 1.800e-01, 1.900e-01, 2.000e-01, 2.100e-01, 2.200e-01, 2.300e-01, 2.400e-01, 2.500e-01, 2.600e-01,
        2.800e-01, 3.000e-01, 3.200e-01, 3.400e-01, 3.600e-01, 3.800e-01, 4.000e-01, 4.200e-01, 4.400e-01, 4.600e-01, 4.800e-01, 5.000e-01,
        5.500e-01, 6.000e-01, 6.500e-01, 7.000e-01, 7.500e-01, 8.000e-01, 8.500e-01, 9.000e-01, 9.500e-01, 1.000e+00, 1.050e+00, 1.100e+00,
        1.150e+00, 1.200e+00, 1.250e+00, 1.300e+00, 1.350e+00, 1.400e+00, 1.450e+00, 1.500e+00, 1.550e+00, 1.600e+00, 1.650e+00, 1.700e+00,
        1.750e+00, 1.800e+00, 1.850e+00, 1.900e+00, 1.950e+00, 2.000e+00, 2.050e+00, 2.100e+00, 2.150e+00, 2.200e+00, 2.250e+00, 2.300e+00,
        2.350e+00, 2.400e+00, 2.450e+00, 2.500e+00, 2.600e+00, 2.700e+00, 2.800e+00, 2.900e+00, 3.000e+00, 3.100e+00, 3.200e+00, 3.300e+00,
        3.400e+00, 3.600e+00, 3.800e+00, 4.000e+00, 4.200e+00, 4.400e+00, 4.600e+00, 4.800e+00, 5.000e+00, 5.200e+00, 5.400e+00, 5.600e+00,
        5.800e+00, 6.000e+00, 6.200e+00, 6.400e+00, 6.500e+00, 6.800e+00, 7.000e+00, 7.500e+00, 8.000e+00, 8.500e+00, 9.000e+00, 9.500e+00,
        1.000e+01, 1.050e+01, 1.100e+01, 1.150e+01, 1.200e+01, 1.250e+01, 1.300e+01, 1.350e+01, 1.400e+01, 1.450e+01, 1.500e+01, 1.550e+01,
        1.600e+01, 1.700e+01, 1.800e+01, 1.900e+01, 2.000e+01, 2.100e+01, 2.200e+01, 2.300e+01, 2.400e+01, 2.500e+01, 2.600e+01, 2.700e+01,
        2.800e+01, 2.900e+01, 3.000e+01, 3.100e+01, 3.200e+01, 3.300e+01, 3.400e+01, 3.600e+01, 3.800e+01, 4.000e+01, 4.200e+01, 4.400e+01,
        4.600e+01, 4.800e+01, 5.000e+01, 5.300e+01, 5.600e+01, 5.900e+01, 6.200e+01, 6.600e+01, 7.000e+01, 7.400e+01, 7.800e+01, 8.200e+01,
        8.600e+01, 9.000e+01, 9.400e+01, 9.800e+01, 1.020e+02, 1.060e+02, 1.098e+02, 1.140e+02, 1.180e+02, 1.232e+02, 1.260e+02, 1.300e+02,
        1.382e+02, 1.550e+02, 1.600e+02, 1.739e+02, 1.800e+02, 1.951e+02, 2.000e+02, 2.100e+02, 2.189e+02, 2.300e+02, 2.456e+02, 2.500e+02,
        2.600e+02, 2.756e+02, 3.092e+02, 3.200e+02, 3.469e+02, 3.600e+02, 3.892e+02, 4.000e+02, 4.200e+02, 4.367e+02, 4.600e+02, 4.800e+02,
        5.000e+02, 6.000e+02, 7.000e+02, 8.000e+02, 9.000e+02, 1.000e+03, 1.020e+03, 1.098e+03, 1.140e+03, 1.232e+03, 1.260e+03, 1.300e+03,
        1.382e+03, 1.550e+03, 1.600e+03, 1.739e+03, 1.800e+03, 1.951e+03, 2.000e+03, 2.100e+03, 2.189e+03, 2.300e+03, 2.456e+03, 2.500e+03,
        2.600e+03, 2.756e+03, 3.092e+03, 3.200e+03, 3.469e+03, 3.600e+03, 3.892e+03, 4.000e+03, 4.200e+03, 4.367e+03, 4.600e+03, 4.800e+03,
        5.000e+03, 6.000e+03, 7.000e+03, 8.000e+03, 9.000e+03, 1.000e+04
         ][::stride])

    grid = []
    # get the midpoints (on log10 scale) between each resonance:
    # emid = [lowBound] + list(10**( ( numpy.log10(energies[1:])+numpy.log10(energies[:-1]) ) / 2.0)) + [highBound]
    # or get midpoints on linear scale:
    emid = [lowBound] + [(e1+e2)/2.0 for e1, e2 in zip(energies[1:], energies[:-1])] + [highBound]
    for e, w, lowedge, highedge in zip(energies, widths, emid[:-1], emid[1:]):
        points = e-w*resonancePos
        grid += [lowedge] + list(points[points>lowedge])
#         print('Around e,w=',e,w,': below:',list(points[points>lowedge]))
        points = e+w*resonancePos[1:]
        grid += list(points[points < highedge])
#         print('Around e,w=',e,w,': aboveG:',list(points[points < highedge]))
    # also add rough grid, to cover any big gaps between resonances, should give at least 10 points per decade:
    npoints = int(numpy.ceil(numpy.log10(highBound)-numpy.log10(lowBound)) * 10)
    grid += list(numpy.logspace(numpy.log10(lowBound), numpy.log10(highBound), npoints))[1:-1]
    grid += [lowBound, highBound, 0.0253]   # region boundaries + thermal
    # if threshold reactions present, add dense grid at and above threshold
    for threshold in thresholds:
        grid += [threshold]
        grid += list(threshold + resonancePos * 1e-2)
    grid = sorted(set(grid))
    # toss any points outside of energy bounds:
    grid = grid[grid.index(lowBound) : grid.index(highBound)+1]
    return numpy.asarray(grid, dtype=DBLE)
                       
def reconstructTensorFlow(gnd,MatrixL,DiagonalOnly,dE,Averaging,stride,EMAX,LMAX,HS,Scale,
            Global,base,verbose,debug,reconstyle,thin,Xcited,Convolute):

    PoPs = gnd.PoPs
    projectile = gnd.PoPs[gnd.projectile]
    target     = gnd.PoPs[gnd.target]
    elasticChannel = '%s + %s' % (gnd.projectile,gnd.target)
    if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
    if hasattr(target, 'nucleus'):     target = target.nucleus
    pZ = projectile.charge[0].value; tZ =  target.charge[0].value
    charged =  pZ*tZ != 0
    identicalParticles = gnd.projectile == gnd.target
    if debug: print("Charged-particle elastic:",charged,",  identical:",identicalParticles)
    
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    if EMAX is not None: emax = min(emax,EMAX)
    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    brune = BC=='Brune'
    if brune: MatrixL = True
    print('BC =',BC, ' brune =',brune,'MatrixL',MatrixL)
    IFG = RMatrix.reducedWidthAmplitudes
    
    n_jsets = len(RMatrix.spinGroups)
    n_poles = 0
    n_angles = 0     # angles
    n_chans = 0
    
    np = len(RMatrix.resonanceReactions)
    ReichMoore = False
    if RMatrix.resonanceReactions[0].eliminated: 
        ReichMoore = True
        np -= 1   # exclude Reich-Moore channel here
        print('Has Reich-Moore damping')
    prmax = numpy.zeros(np)
    QI = numpy.zeros(np)
    rmass = numpy.zeros(np)
    za = numpy.zeros(np)
    zb = numpy.zeros(np)
    jp = numpy.zeros(np)
    pt = numpy.zeros(np)
    ep = numpy.zeros(np)
    jt = numpy.zeros(np)
    tt = numpy.zeros(np)
    et = numpy.zeros(np)
    hsphrad = numpy.zeros(np)
    cm2lab  = numpy.zeros(np)
    pname = ['' for i in range(np)]
    tname = ['' for i in range(np)]
    
    partitions = {}
    channels = {}
    pair = 0
    ipair = None
    for partition in RMatrix.resonanceReactions:
        kp = partition.label
        if partition.eliminated:  
            partitions[kp] = None
            continue
        partitions[kp] = pair
        channels[pair] = kp
        reaction = partition.reactionLink.link
        p,t = partition.ejectile,partition.residual
        pname[pair] = p
        tname[pair] = t
        projectile = PoPs[p];
        target     = PoPs[t];
        pMass = projectile.getMass('amu');   tMass =     target.getMass('amu');
        rmass[pair] = pMass * tMass / (pMass + tMass)
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus

        za[pair]    = projectile.charge[0].value;  
        zb[pair]  = target.charge[0].value
        if partition.Q is not None:
            QI[pair] = partition.Q.getConstantAs('MeV')
        else:
            QI[pair] = reaction.getQ('MeV')
        if partition.scatteringRadius is not None:
            prmax[pair] =  partition.scatteringRadius.getValueAs('fm')
        else:
            prmax[pair] = Rm_global
            
        if partition.hardSphereRadius is not None:
            hsphrad[pair] =  partition.hardSphereRadius.getValueAs('fm')
        else:
            hsphrad[pair] = prmax[pair]        
        if partition.label == elasticChannel:
            lab2cm = tMass / (pMass + tMass)
            ipair = pair  # incoming
        cm2lab[pair] = (pMass + tMass) / tMass
            
        jp[pair],pt[pair],ep[pair] = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt[pair],tt[pair],et[pair] = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].pqu('MeV').value
        except:
            jt[pair],tt[pair],et[pair] = 0.,1,0.
        parity = '+' if tt[pair] > 0 else '-'
        print("%3i, %s :%s%s"%(pair,kp,jt[pair],parity),',',QI[pair],'radii',prmax[pair],hsphrad[pair])
        pair += 1
    npairs  = pair
    
#  FIRST: for array sizes:
    Lmax = 0
    tot_poles = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        n_poles = max(n_poles,R.nRows)
        n = R.nColumns-1
        if ReichMoore: n -= 1
        n_chans = max(n_chans,n)
        tot_poles += n_poles
        for ch in Jpi.channels:
            Lmax = max(Lmax,min(ch.L,LMAX))
    print('Need %i Jpi sets with %i poles max, and %i channels max. Lmax=%i. Total poles %i' % (n_jsets,n_poles,n_chans,Lmax,tot_poles))

    E_poles = numpy.zeros([n_jsets,n_poles], dtype=DBLE)
    E_damping = numpy.zeros([n_jsets,n_poles], dtype=DBLE)
    has_widths = numpy.zeros([n_jsets,n_poles], dtype=INT)
    g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
    J_set = numpy.zeros(n_jsets, dtype=DBLE)
    pi_set = numpy.zeros(n_jsets, dtype=INT)
    L_val  =  numpy.zeros([n_jsets,n_chans], dtype=INT)
    S_val  =  numpy.zeros([n_jsets,n_chans], dtype=DBLE)
    S_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
    B_chans = numpy.zeros([n_jsets,n_chans], dtype=DBLE)
    p_mask =  numpy.zeros([npairs,n_jsets,n_chans], dtype=DBLE)
    seg_val=  numpy.zeros([n_jsets,n_chans], dtype=INT) - 1
    seg_col=  numpy.zeros([n_jsets], dtype=INT) 
    seg_col=  numpy.zeros([n_jsets], dtype=INT) 

    Spins = [set() for pair in range(npairs)]
    if debug: print('partitions:',partitions)
    
#  SECOND: fill in arrays for channel specifications
    jset = 0
#     Penergies = []
#     Fwidths = []
    EFwidths = []
    All_spins = set()
    for Jpi in RMatrix.spinGroups:
        J_set[jset] = Jpi.spin
        pi_set[jset] = Jpi.parity
        # print('J,pi =',J_set[jset],pi_set[jset])
        R = Jpi.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns - 1  # ignore energy col
        seg_col[jset] = cols if not ReichMoore else cols-1

        E_poles[jset,:rows] = numpy.asarray( R.getColumn('energy','MeV') , dtype=DBLE)   # lab MeV
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
        
        if ReichMoore: E_damping[jset,:rows] = numpy.asarray(widths[0][:],  dtype=DBLE)
        if IFG==1:     E_damping[jset,:] = 2*E_damping[jset,:]**2            
        if ReichMoore: print('Set',jset,'radiative damping',E_damping[jset,:rows])
                    
        c = 0
        for ch in Jpi.channels:
            rr = ch.resonanceReaction
            pair = partitions.get(rr,None)
            if pair is None: continue
            if ch.L > LMAX: continue
            seg_val[jset,c] = pair

            m = ch.columnIndex - 1
            g_poles[jset,:rows,c] = numpy.asarray(widths[m][:],  dtype=DBLE)


            L_val[jset,c] = ch.L
            S = float(ch.channelSpin)
            S_val[jset,c] = S
            has_widths[jset,:rows] = 1
            
            p_mask[pair,jset,c] = 1.0
            Spins[pair].add(S)
            All_spins.add(S)

            if BC == resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction:
                B = None # replace below by S
            elif BC == 'Brune':
                B = 0 # not used
            elif BC == resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
                B = -ch.L
            elif BC == resolvedResonanceModule.BoundaryCondition.Given:              # btype='B'
                B = BV
            if ch.boundaryConditionValue is not None:
                B = ch.boundaryConditionValue
            B_chans[jset,c] = B

            c += 1
            
        for p in range(rows): # scale g by es0 + es1*E+ es2*E^2 + ... 
            E = E_poles[jset,p]
            if E <= 0. or Scale is None: continue
            s = 0.0
            Ep = 1.0
            for es in Scale:
                s += es * Ep
                Ep *= E
            g_poles[jset,p,:] *= s
            
        jset += 1
    
# MAKE ENERGY GRID              
                    
    if dE is not None:
        n_energies = int( (emax - emin)/dE + 1.0)
        E_scat = numpy.linspace(emin,emax, n_energies, dtype=DBLE)
        
    else:  # generate grid bunching at pole peaks using formal width for guidance
    # COULOMB functions at pole energies if needed      
        for jset in range(n_jsets):
            parity = '+' if pi_set[jset] > 0 else '-'
            print('J,pi =',J_set[jset],parity)
            for n in range(n_poles):
                Fwid = 0.0
                obsEnergy = E_poles[jset,n]
                if E_poles[jset,n] == 0.0: continue
                if debug: print('   pairs =',seg_val[jset,:])
                for c in range(n_chans):
                    pair = seg_val[jset,c]
                    if pair < 0: continue
                    E = E_poles[jset,n]*lab2cm + QI[pair]
                    if rmass[pair]!=0:
                        k = cmath.sqrt(fmscal * rmass[pair] * E)
                    else: # photon!
                        k = E/hbc
                    if debug: print('Pole E,k = ',E,k)
                    rho = k * prmax[pair]
                    if abs(rho) <1e-10:
                        if debug:
                            print('rho =',rho,'from pair,E,k,r =',pair,E,k,prmax[pair])
                            print('from E = ', E_poles[jset,n],'*',lab2cm,'+', QI[pair])
                        S_poles[jset,n,c] = 0.0
                        continue
    #                 print('rho =',rho,'from E,k,r =',E,k,prmax[pair])
                    eta  =  etacns * za[pair]*zb[pair] * cmath.sqrt(rmass[pair]/E)
                    negE = E < 0
                    if negE:  eta = -eta  #  negative imaginary part for bound states
                    PM   = complex(0.,1.);
                    EPS=1e-10; LIMIT = 2000000; ACC8 = 1e-12
                    ZL = 0.0
                    DL,ERR = cf2(rho,eta,ZL,PM,EPS,LIMIT,ACC8)
                    LL = L_val[jset,c]
    #                 CF2_val[ie,pair,0] = DL * rho
                    for L in range(1,LL+1):
                        RLsq = 1 + (eta/L)**2
                        SL   = L/rho + eta/L
                        DL = RLsq/( SL - DL) - SL
                    S_poles[jset,n,c] = (DL*rho).real
                    P = (DL*rho).imag
                    if B is None: B_chans[jset,c] = S_poles[jset,n,c]   # B=S approximation
                
                    if IFG:
                        Pwid = 2 * g_poles[jset,n,c]**2 * P
                    else:
                        Pwid = g_poles[jset,n,c]   # IFG=0
                        if  negE: # E<0: calculate fake P for |E| scattering, to convert ENDF value 
                            E = abs(E) 
                            if rmass[pair]!=0:
                                k = cmath.sqrt(fmscal * rmass[pair] * E)
                            else: # photon!
                                k = E/hbc
                            rho = k * prmax[pair]
                            eta  =  etacns * za[pair]*zb[pair] * cmath.sqrt(rmass[pair]/E)
                            DL,ERR = cf2(rho,eta,ZL,PM,EPS,LIMIT,ACC8)
                            for L in range(1,LL+1):
                                RLsq = 1 + (eta/L)**2
                                SL   = L/rho + eta/L
                                DL = RLsq/( SL - DL) - SL
                            S = (DL*rho).real
                            P = (DL*rho).imag                    
                        g_poles[jset,n,c] = (abs(Pwid)/(2*P))**0.5 * (1 if Pwid > 0 else -1)
                
                    Fwid += Pwid
                    if not brune:
                        obsEnergy -= g_poles[jset,n,c]**2 * (S_poles[jset,n,c] - B_chans[jset,c])
                        if verbose: print('Pole at E=',obsEnergy,'from',E_poles[jset,n],'in channel',c,'has partial width',Pwid,'summing to',Fwid)
                
                    if debug: print('S_poles[%i,%i,%i] = %10.5f for L=%i' % (jset,n,c,S_poles[jset,n,c].real,LL))
            
                EFwidths.append((obsEnergy,Fwid))    # TEMP: replace Fwid later by obsWid

        EFwidths.sort(key = lambda x: x[0])            
        if verbose: print('Observed energies + Formal widths:\n',EFwidths)
    #     print('Radiative damping',E_damping)
#         print('Energies + Formal widths sorted:')
#         for e,w in EFwidths:  print(' E = %10.6f, w = %10.6f' % (e,w))
        Penergies,Fwidths = zip(*EFwidths)

        E_scat = generateEnergyGrid(Penergies,Fwidths, emin,emax, stride=stride)
        n_energies = len(E_scat)
    
    print('\nEnergy grid over emin,emax =',emin,emax,'with',n_energies)
    if debug: print('First energy grid:\n',E_scat)
    sys.stdout.flush()
    
#  THIRD: Calculate Coulomb functions on the energy grid for each cross-sections

    rksq_val  = numpy.zeros([n_energies,npairs], dtype=DBLE)
#     velocity  = numpy.zeros([n_energies,npairs], dtype=DBLE)
    
    eta_val = numpy.zeros([n_energies,npairs], dtype=DBLE)   # for E>0 only
    
    CF1_val =  numpy.zeros([n_energies,np,Lmax+1], dtype=DBLE)
    CF2_val =  numpy.zeros([n_energies,np,Lmax+1], dtype=CMPLX)
    csigma_v=  numpy.zeros([n_energies,np,Lmax+1], dtype=DBLE)
    Csig_exp=  numpy.zeros([n_energies,np,Lmax+1], dtype=CMPLX)
#     Shift         = numpy.zeros([n_energies,n_jsets,n_chans], dtype=DBLE)
#     Penetrability = numpy.zeros([n_energies,n_jsets,n_chans], dtype=DBLE)
    L_diag = numpy.zeros([n_energies,n_jsets,n_chans], dtype=CMPLX)
    POm_diag = numpy.zeros([n_energies,n_jsets,n_chans], dtype=CMPLX)
    Om2_mat = numpy.zeros([n_energies,n_jsets,n_chans,n_chans], dtype=CMPLX)
    CS_diag = numpy.zeros([n_energies,n_jsets,n_chans], dtype=CMPLX)
    
    if brune:  # S_poles: Shift functions at pole positions for Brune basis   
        S_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
        dSdE_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
#         EO_poles =  numpy.zeros([n_jsets,n_poles]) 
        EO_poles = E_poles.copy()
        Pole_Shifts(S_poles,dSdE_poles, EO_poles,has_widths, seg_val,lab2cm,QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
    else:
        S_poles = None
        dSdE_poles = None
        EO_poles = None
            
    for pair in range(npairs):
        if debug:
            foutS = open(base + '+3-S%i' % pair,'w')
            foutP = open(base + '+3-P%i' % pair,'w')
        for ie in range(n_energies):
            E = E_scat[ie]*lab2cm + QI[pair]
            if abs(E) < 1e-10:
                E = (E + E_scat[ie+1]*lab2cm + QI[pair]) * 0.5
            k = cmath.sqrt(fmscal * rmass[pair] * E)
            if rmass[pair]!=0:
                k = cmath.sqrt(fmscal * rmass[pair] * E)
            else: # photon!
                k = E/hbc
            rho = k * prmax[pair]
            if abs(rho) <1e-10: 
                print('rho =',rho,'from E,k,r =',E,k,prmax[pair])
            eta  =  etacns * za[pair]*zb[pair] * cmath.sqrt(rmass[pair]/E)
            if E < 0: eta = -eta  #  negative imaginary part for bound states
            PM   = complex(0.,1.); 
            EPS=1e-10; LIMIT = 2000000; ACC8 = 1e-12
            ZL = 0.0
            DL,ERR = cf2(rho,eta,ZL,PM,EPS,LIMIT,ACC8)
            CF2_val[ie,pair,0] = DL
            for L in range(1,Lmax+1):
                RLsq = 1 + (eta/L)**2
                SL   = L/rho + eta/L
                CF2_val[ie,pair,L] = RLsq/( SL - CF2_val[ie,pair,L-1]) - SL

            if E > 0.:
                CF1_val[ie,pair,Lmax] = cf1(rho.real,eta.real,Lmax,EPS,LIMIT)
                for L in range(Lmax,0,-1):
                    RLsq = 1 + (eta.real/L)**2
                    SL   = L/rho.real + eta.real/L
                    CF1_val[ie,pair,L-1] = SL - RLsq/( SL + CF1_val[ie,pair,L]) 

            CF1_val[ie,pair,:] *=  rho.real
            CF2_val[ie,pair,:] *=  rho
            rksq_val[ie,pair] = 1./max(abs(k)**2, 1e-20) 
#             velocity[ie,pair] = k.real/rmass[pair]  # ignoring factor of hbar
            
            if E > 0.:
                eta_val[ie,pair] = eta.real
                csigma_v[ie,pair,:] = csigma(Lmax,eta)
                for L in range(Lmax+1):
                    Csig_exp[ie,pair,L] = cmath.exp(complex(0.,csigma_v[ie,pair,L]-csigma_v[ie,pair,0]))
                    # Csig_exp[ie,pair,L] = cmath.exp(complex(0.,csigma_v[ie,pair,L]))
                    # Csig_exp[ie,pair,L] = 1.
            else:
                eta_val[ie,pair] = 0.0
                Csig_exp[ie,pair,:] = 1.0
        if debug:
            foutS.close()
            foutP.close()
        
    #  FOURTH: fill in more Coulomb-related functions for R-matrix calculations
    jset = 0
    for Jpi in RMatrix.spinGroups:
        rows = Jpi.resonanceParameters.table.nRows
#         print('For jset',jset,'there are',seg_col[jset],'channels')
        for c in range(seg_col[jset]):
            L = L_val[jset,c]  
            if L > LMAX: continue
            pair = seg_val[jset,c]
     
            B = B_chans[jset,c]  # or B=S if B is None: see below
            if verbose: print('jset,c: BC,B=' , jset,c,BC,B )
                
        # Find S and P:
            for ie in range(n_energies):

                DL = CF2_val[ie,pair,L]
                S = DL.real
                P = DL.imag
                F = CF1_val[ie,pair,L]
                Psr = math.sqrt(abs(P))
                phi = - math.atan2(P, F - S)
                if HS: phi = 0.
                Omega = cmath.exp(complex(0,phi))
                if B is None:
                    L_diag[ie,jset,c]       = complex(0.,P)
                else:
                    L_diag[ie,jset,c]       = DL - B

                POm_diag[ie,jset,c]      = Psr * Omega
                Om2_mat[ie,jset,c,c]     = Omega**2
                CS_diag[ie,jset,c]       = Csig_exp[ie,pair,L]


                
                if abs(prmax[pair]-hsphrad[pair]) > 1e-6:     # tediously & reluctantly recalculate hard-sphere phase shifts:

                    E = E_scat[ie]*lab2cm + QI[pair]
                    if E < 0: continue        # phase-shifts = 0 for bound states
                    
                    k = cmath.sqrt(fmscal * rmass[pair] * E)
                    rho = k * hsphrad[pair]
                    if abs(rho) <1e-10: print('rho =',rho,'from E,k,r =',E,k,prmax[pair])
                    eta  =  etacns * za[pair]*zb[pair] * cmath.sqrt(rmass[pair]/E)
                    if E < 0: eta = -eta  #  negative imaginary part for bound states
                    PM   = complex(0.,1.); 
                    EPS=1e-10; LIMIT = 2000000; ACC8 = 1e-12
                    ZL = 0.0
                    DL,ERR = cf2(rho,eta,ZL,PM,EPS,LIMIT,ACC8)
                    CF2_val[ie,pair,0] = DL
                    for L in range(1,Lmax+1):
                        RLsq = 1 + (eta/L)**2
                        SL   = L/rho + eta/L
                        CF2_val[ie,pair,L] = RLsq/( SL - CF2_val[ie,pair,L-1]) - SL

                    if E > 0.:
                        CF1_val[ie,pair,Lmax] = cf1(rho.real,eta.real,Lmax,EPS,LIMIT)
                        for L in range(Lmax,0,-1):
                            RLsq = 1 + (eta.real/L)**2
                            SL   = L/rho.real + eta.real/L
                            CF1_val[ie,pair,L-1] = SL - RLsq/( SL + CF1_val[ie,pair,L]) 

                    CF1_val[ie,pair,:] *=  rho.real
                    CF2_val[ie,pair,:] *=  rho

                    DL = CF2_val[ie,pair,L]
                    S = DL.real
                    P = DL.imag
                    F = CF1_val[ie,pair,L]
                    Psr = math.sqrt(abs(L_diag[ie,jset,c].imag))   # at the original scattering radius!
                    phi = - math.atan2(P, F - S)
                    if HS: phi = 0.
                    Omega = cmath.exp(complex(0,phi))           
                    POm_diag[ie,jset,c]      = Psr * Omega
                    Om2_mat[ie,jset,c,c]     = Omega**2            

        if verbose:
            print('J set %i: E_poles \n' % jset,E_poles[jset,:])
            print('E_damp  \n',E_damping[jset,:])
            print('g_poles \n',g_poles[jset,:,:])
        jset += 1        

#    print('All spins:',All_spins)
#    print('All channel spins',Spins)
    
    gfac = numpy.zeros([n_energies,n_jsets,n_chans])
    for jset in range(n_jsets):
        for c_in in range(n_chans):   # incoming partial wave
            pair = seg_val[jset,c_in]      # incoming partition
            if pair>=0:
                denom = (2.*jp[pair]+1.) * (2.*jt[pair]+1)
                for ie in range(n_energies):
                    gfac[ie,jset,c_in] = pi * (2*J_set[jset]+1) * rksq_val[ie,pair] / denom 

    sys.stdout.flush()

#    START TENSORFLOW CALLS:

    E_cpoles = tf.complex(E_poles,-E_damping*0.5) # tf.constant(0., dtype=DBLE)) 
    g_cpoles = tf.complex(g_poles,tf.constant(0., dtype=DBLE))
    E_cscat = tf.complex(E_scat,tf.constant(Averaging*0.5, dtype=DBLE)) 
    
    if not MatrixL:
        RMATC,T_mat = R2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
    else:
        T_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, DiagonalOnly,n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 

    XSp_mat,XSp_tot,XSp_cap,XSp_reac  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
    
#    END TENSORFLOW CALLS:

# PRINT TOTAL and ANGLE-INTEGRATED CROSS-SECTIONS:

    XSp_mat_n,XSp_tot_n,XSp_cap_n,XSp_reac_n = XSp_mat.numpy(),XSp_tot.numpy(),XSp_cap.numpy(),XSp_reac.numpy()
    G = 'G' if Global else ''        
    if Convolute > 0.:
        from xData import XYs

        def spread(de,s):
            pi = 3.14159
            c = 1/pi**0.5 / s
            return (c* math.exp(-(de/s)**2))
    
        fun = []
        for i in range(100):
            de = (i-50)*Convolute*0.1
            f = spread(de,Convolute)
            fun.append([de,f])
        conv = XYs.XYs1d(fun)
        print("Convolute with Gaussian in %s * [-5,5] with steps of 0.1*%s" % (Convolute,Convolute))
    Ex = numpy.zeros(n_energies)
    Cy = numpy.zeros(n_energies)
    
    for pair in range(npairs):
        pn = quickName(pname[pair],tname[pair]) 
        _,tlevel = nuclIDs(tname[pair])
        if tlevel > 0 and not Xcited: continue
        
        neut = za[pair]*zb[pair] == 0    # calculate total cross-sections for neutrons
        if neut:
            fname = base + '-%stot_%s' % (G,pn)
            print('Total cross-sections for incoming',pair,'to file',fname)
            fout = open(fname,'w')
        cname = base + '-%scap_%s' % (G,pn)
        print('Capture cross-sections for incoming',pair,'to file',cname)
        cout = open(cname,'w')
        for ie in range(n_energies):
#           E_scat[ie]      is lab incident energy in nominal entrance partition  ipair
            E = E_scat[ie]*lab2cm + QI[pair] - QI[ipair]
            Elab = E * cm2lab[pair]
            Eo = E_scat[ie]*lab2cm if Global else Elab
            x = XSp_tot_n[ie,pair] * 10.  # mb
            Ex[ie] = Eo
            Cy[ie] = x
            if neut and Convolute <= 0 and (Global or Elab>0) : print(Eo,x, file=fout)
            
            c = XSp_cap_n[ie,pair]*10.
            if Global or Elab>0.: print(Eo,c, file=cout)
            
        if Convolute>0. and neut:
#             print('Ex:',Ex)
            XSEC = XYs.XYs1d(data=(Ex,Cy), dataForm="XsAndYs"  )                
            XSEC = XSEC.convolute(conv)
            for ie in range(len(XSEC)):
                print(XSEC[ie][0],XSEC[ie][1], file=fout)    
        cout.close()

        rname = base + '-%sreac_%s' % (G,pn)
        print('Reaction cross-sections for incoming',pair,'to file',rname)
        rout = open(rname,'w')
        for ie in range(n_energies):
#           E_scat[ie]      is lab incident energy in nominal entrance partition  ipair
            E = E_scat[ie]*lab2cm + QI[pair] - QI[ipair]
            Elab = E * cm2lab[pair]
            Eo = E_scat[ie]*lab2cm if Global else Elab
            x = XSp_reac_n[ie,pair] * 10.  # mb
            Ex[ie] = Eo
            Cy[ie] = x
            if Convolute <= 0 and (Global or Elab>0) : print(Eo,x, file=rout)
            
        if Convolute>0.:
#             print('Ex:',Ex)
            XSEC = XYs.XYs1d(data=(Ex,Cy), dataForm="XsAndYs"  )                
            XSEC = XSEC.convolute(conv)
            for ie in range(len(XSEC)):
                print(XSEC[ie][0],XSEC[ie][1], file=rout)    
        rout.close()

        if Convolute>0.:
            sum = None
            fnsum = base + '-%ssum_%s' % (G,pn)
            print('Output sum from',pn,': angle-integrated cross-sections to file',fnsum)
            fsum = open(fnsum,'w')
            
            rsum = None
            fnrsum = base + '-%srcs_%s' % (G,pn)

                        
        for pout in range(npairs):
#             if pair==pout and not neut: continue
            po = quickName(pname[pout],tname[pout])
            fname = base + '-%sch_%s-to-%s' % (G,pn,po)
            print('Partition',pn,'to',po,': angle-integrated cross-sections to file',fname)
            fout = open(fname,'w')
            for ie in range(n_energies):
                x = XSp_mat_n[ie,pout,pair] * 10.
                E = E_scat[ie]*lab2cm + QI[pair] - QI[ipair]
                Elab = E * cm2lab[pair]   # Elab for incoming channel (pair, not ipair)
                Eo = E_scat[ie]*lab2cm if Global else Elab
                Ex[ie] = Eo
                Cy[ie] = x
                if Convolute<=0. and (Global or Elab>0): print(Eo,x, file=fout)
                
            if Convolute>0.:
                XSEC = XYs.XYs1d(data=(Ex,Cy), dataForm="XsAndYs"  )                
                XSEC = XSEC.convolute(conv)
                sum = XSEC if sum is None else sum+XSEC           
                if pout != pair:
                    rsum = XSEC if rsum is None else rsum+XSEC           
                for ie in range(len(XSEC)):
                    print(XSEC[ie][0],XSEC[ie][1], file=fout)     
            fout.close()
            
        if Convolute>0.:
            for ie in range(len(sum)):
                print(sum[ie][0],sum[ie][1], file=fsum)     
            fsum.close()
            if rsum is not None:
                print('Output reaction cross-section from',pn,' to file',fnrsum)
                frsum = open(fnrsum,'w')
                for ie in range(len(rsum)):
                    print(rsum[ie][0],rsum[ie][1], file=frsum)     
                frsum.close()    
                    
    if debug:
        if not MatrixL:
            for ie in range(n_energies):
                for jset in range(n_jsets):
                    print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n R-matrix is size',seg_col[jset])
                    for a in range(n_chans):
                        print('   ',a,'row: ',',  '.join(['{:.5f}'.format(RMATC[ie,jset,a,b].numpy()) for b in range(n_chans)]) )
    
        for ie in range(n_energies):
            for jset in range(n_jsets):
                print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n T-matrix is size',seg_col[jset])
                for a in range(n_chans):
                    print('   ',a,'row: ',',  '.join(['{:.5f}'.format(T_mat[ie,jset,a,b].numpy()) for b in range(n_chans)])  )
                print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n S-matrix is size',seg_col[jset])
                for a in range(n_chans):
                    print('   ',a,'row: ',',  '.join(['{:.5f}'.format(numpy.eye(n_chans)[a,b] - T_mat[ie,jset,a,b].numpy()) for b in range(n_chans)])  )

    if reconstyle is None: return

    ## # PROCESS CROSS-SECTIONS
    
    rStyle = reconstyle.label
                    
    egrid = E_scat[:]    # lab MeV
    totalxs = XSp_tot_n[:,ipair] * 0.01   # barns
    if charged:
        elasticxs = None # sig_ni[:] * 1e-3 # barns not mb
    else:
        elasticxs = XSp_mat_n[:,ipair,ipair] * 0.01 # barns
    fissionxs = numpy.zeros(n_energies)
    absorbtionxs = totalxs - numpy.sum(XSp_mat_n[:,:,ipair], axis=1)*0.01  # barns
    chanxs = [elasticxs]
    for pout in range(npairs):
        if pout == ipair:  continue   # skip elastic: that was first.
        chanxs.append( XSp_mat_n[:,pout,ipair] * 0.01)

    crossSectionAxes = crossSectionModule.defaultAxes( 'MeV' )
    total = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, totalxs), dataForm="XsAndYs" )
    if not charged:
        elastic = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, elasticxs), dataForm="XsAndYs" )
    else:
        None
    fission = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, fissionxs), dataForm="XsAndYs" )
    absorbtion = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, absorbtionxs), dataForm="XsAndYs" )

    if not isinstance( reconstyle, stylesModule.crossSectionReconstructed ):
        raise TypeError("style must be an instance of crossSectionReconstructed, not %s" % type(reconstyle))

    haveEliminated = False
    for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
        reaction = rreac.reactionLink.link
        haveEliminated = haveEliminated or rreac.eliminated
        #                  elastic or capture 
        if reaction == gnd.getReaction('capture'): rreac.tag = 'capture'
        elif reaction == gnd.getReaction('elastic'): rreac.tag = 'elastic'
        elif 'fission' in rreac.label: rreac.tag = rreac.label
        else: rreac.tag = 'competitive'
                
    xsecs = {'total':total, 'elastic':elastic, 'fission':fission, 'nonelastic':absorbtion}
    for c in range(1,npairs):  # skip c=1 elastic !! FIXME
#         print('Channel:',c, channels[c],':',len(egrid),len(chanxs[c]) )
#         print(chanxs[c])
        xsecs[channels[c]] = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, chanxs[c]), dataForm="XsAndYs" )
#         print('xsecs[channels[c]]',xsecs[channels[c]].toString() )

    if haveEliminated:
        eliminatedReaction = [rr for rr in gnd.resonances.resolved.evaluated.resonanceReactions if rr.eliminated]
        if len(eliminatedReaction) != 1:
            raise TypeError("Only 1 reaction can be eliminated in Reich-Moore approximation!")
        xsecs[eliminatedReaction[0].tag] = absorbtion - fission
                
    epsilon = 1e-8  # for joining multiple regions together

    # for each reaction, add tabulated pointwise data (ENDF MF=3) to reconstructed resonances:
    possibleChannels = { 'elastic' : True, 'capture' : True, 'fission' : True, 'total' : False, 'nonelastic' : False }
    elasticChannel = gnd.getReaction('elastic')
    derivedFromLabel = ''
    for reaction in gnd :
        if isinstance( reaction, sumsModule.multiplicitySum ): continue
        iselastic = reaction is elasticChannel

        evaluatedCrossSection = reaction.crossSection.evaluated
        if not isinstance( evaluatedCrossSection, crossSectionModule.resonancesWithBackground ):
            continue
        # which reconstructed cross section corresponds to this reaction?
        if( derivedFromLabel == '' ) : derivedFromLabel = evaluatedCrossSection.label
        if( derivedFromLabel != evaluatedCrossSection.label ) :
            print(('WARNING derivedFromLabel = "%s" != "%s"' % (derivedFromLabel, evaluatedCrossSection.label)))
        RRxsec = None
        if str( reaction ) in xsecs:
            RRxsec = xsecs[ str( reaction ) ]
        else :
            for possibleChannel in possibleChannels :
                if( possibleChannels[possibleChannel] ) :
                    if( possibleChannel in str( reaction ) ) : 
                        RRxsec = xsecs[possibleChannel]
                if( RRxsec is None ) :
                    if( reaction is gnd.getReaction( possibleChannel ) ) : 
                        RRxsec = xsecs[possibleChannel]
                if( RRxsec is not None ) : break
        if( RRxsec is None ) :
            if verbose:
                print(( "Warning: couldn't find appropriate reconstructed cross section to add to reaction %s" % reaction ))
            continue

        background = evaluatedCrossSection.background
        background = background.toPointwise_withLinearXYs( accuracy = 1e-3, lowerEps = epsilon, upperEps = epsilon )
        RRxsec = RRxsec.toPointwise_withLinearXYs( accuracy = 1e-3, lowerEps = epsilon, upperEps = epsilon )
        RRxsec.convertUnits( {RRxsec.domainUnit: background.domainUnit,  RRxsec.rangeUnit: background.rangeUnit } )

        background, RRxsec = background.mutualify(0,0,0, RRxsec, -epsilon,epsilon,True)
        RRxsec = background + RRxsec    # result is a crossSection.XYs1d instance
        if thin:
            RRx = RRxsec.thin( 0.001 )
        else:
            RRx = RRxsec
        RRx.label = rStyle

        reaction.crossSection.add( RRx )
       
        # print("Channels ",reaction.label,iselastic,":\n",RRxsec.toString(),"\n&\n",RRx.toString())
        if iselastic:
            effXsc = RRxsec
            
    gnd.styles.add( reconstyle )
    return 


if __name__=="__main__":
    import argparse
    from fudge import reactionSuite as reactionSuiteModule

    parser = argparse.ArgumentParser(description='Translate R-matrix Evaluations')
    parser.add_argument('inFiles', type=str, nargs='+', help='The input file you want to pointwise expand.' )
    parser.add_argument("-M", "--MatrixL", action="store_true", help="Use level matrix method if not already Brune basis")
    parser.add_argument("-D", "--DiagonalOnly", type=str, help="Model S(SLBW) or M(MLBW) for diagonal-only level matrix")

    parser.add_argument("-S", "--Scale", type=float,nargs='+', help="Scale all amplitudes by factor")
    parser.add_argument(      "--dE", type=float, help="Energy step for uniform energy grid, in MeV")
    parser.add_argument("-E", "--EMAX", type=float, help="Maximum Energy (MeV)")
    parser.add_argument("-s", "--stride", type=int, help="Stride for accessing non-uniform grid template")
    parser.add_argument("-L", "--LMAX", type=int, help="Max partial wave L")
    parser.add_argument("-H", "--HardSphere", action="store_true", help="Without hard-sphere phase shift")
    parser.add_argument("-w", "--write", action="store_true", help="Write cross-sections in GNDS file")
    parser.add_argument("-t", "--thin", action="store_true", help="Thin distributions in GNDS form")
    parser.add_argument("-G", "--Global", action="store_true", help="print output excitation functions in cm energy of GNDS projectile ")
    parser.add_argument("-X", "--Xcited", action="store_true", help="calculate and print output for excited initial states")
    parser.add_argument("-A", "--Averaging", type=float, default=0.0, help="Averaging width to all scattering: imaginary = Average/2.")
    parser.add_argument("-C", "--Convolute", type=float, default=0.0, help="Gaussian width to convolute excitation functions")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debugging output (more than verbose)")

    args = parser.parse_args()
    debug = args.debug
    verbose = args.verbose or debug
    print("Reconstruct pointwise cross sections using TensorFlow")
    cmd = ' '.join([t if '*' not in t else ("'%s'" % t) for t in sys.argv[:]])
    print('Command:',cmd ,'\n')
   
    for inFile in args.inFiles:

        gnd=reactionSuiteModule.readXML(inFile)
    #     base = inFile.replace('.xml','_tf')
        base = '_tf'.join(inFile.rsplit('.xml',1))
        LMAX = args.LMAX
        if args.Scale is not None: base += 'x'+str(args.Scale) # .replace('.0','')
        if args.dE is not None: base += '+'+str(args.dE)+'MeV'
        if LMAX is not None: 
            base += '-L%i' % args.LMAX
        else:
            LMAX = 100
        if args.HardSphere: base += '-H'
        if args.DiagonalOnly is not None: args.MatrixL = True
        if args.MatrixL: base += 'M'
        if args.DiagonalOnly is not None: base += '-MLBW' if args.DiagonalOnly[0]=='M' else '-SLBW'
        if args.thin: base += '+th'
        if args.stride is not None: base += '+s%s' % args.stride
        if args.Averaging>0.: base += '+A%s' % args.Averaging
        if args.Convolute>0.: base += '+C%s' % args.Convolute

        if args.write:
            finalStyleName = 'recon'
            reconstructedStyle = stylesModule.crossSectionReconstructed( finalStyleName,
                    derivedFrom=gnd.styles.getEvaluatedStyle().label )
        else:
            reconstructedStyle = None
        print('base:',base,'\n')
    
        reconstructTensorFlow(gnd,args.MatrixL,args.DiagonalOnly,args.dE,args.Averaging,args.stride,args.EMAX,LMAX,args.HardSphere,args.Scale,
            args.Global,base,verbose,debug,reconstructedStyle,args.thin,args.Xcited,args.Convolute)

        if args.write:
            outFile = base + '.xml'
    
            open( outFile, mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
            print('Written',outFile)
        print('Recommended stdout:',base + '.out')

