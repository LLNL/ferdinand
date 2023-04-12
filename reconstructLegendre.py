#! /usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
TF = True

import sys,os,math,numpy,cmath
from CoulCF import cf1,cf2,csigma,Pole_Shifts

from pqu import PQU as PQUModule
# from PoPs.chemicalElements.misc import *

import fudge.sums as sumsModule
import fudge.styles as stylesModule
import fudge.reactionData.crossSection as crossSectionModule
from fudge.productData.distributions import unspecified as unspecifiedModule
import fudge.productData.distributions as distributionsModule
import fudge.resonances.resolved as resolvedResonanceModule
from xData import enums as xDataEnumsModule

DBLE = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32

if TF: 
    # import tensorflow as tf
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    
# TO do
#   Adaptive energy grid
#   Brune basis: level matrix calculation


##############################################  reconstructLegendre

hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 3.1415926536
rsqr4pi = 1.0/(4*pi)**0.5

logMinLegendreValue = 4
extraFloatPrecision = 4
minLegendreValue = 10**(-logMinLegendreValue)

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
    TC_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)
    
    return(T_mat,TC_mat)
        
@tf.function
def T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs):
                    
    Tmod2 = tf.math.real(  T_mat * tf.math.conj(T_mat) )   # ie,jset,a1,a2

# sum of Jpi sets:
    G_fac = tf.reshape(gfac, [-1,n_jsets,1,n_chans])
    XS_mat = Tmod2 * G_fac                          # ie,jset,a1,a2   
  #  print('XS_mat',XS_mat.dtype,XS_mat.get_shape())
    
    G_fact = tf.reshape(gfac, [-1,n_jsets,n_chans])
    TOT_mat = tf.math.real(tf.linalg.diag_part(T_mat))   #  ie,jset,a  for  1 - Re(S) = Re(1-S) = Re(T)
    XS_tot  = TOT_mat * G_fact                           #  ie,jset,a
    p_mask1_in = tf.reshape(p_mask, [-1,npairs,n_jsets,n_chans] )   # convert pair,jset,a to  ie,pair,jset,a
    XSp_tot = 2. *  tf.reduce_sum( tf.expand_dims(XS_tot,1) * p_mask1_in , [2,3])     # convert ie,pair,jset,a to ie,pair by summing over jset,a
        
    p_mask_in = tf.reshape(p_mask,[1,1,npairs,n_jsets,1,n_chans])   # ; print('p_mask_in',p_mask_in.get_shape())   # 1,1,pin,jset,1,cin
    p_mask_out =tf.reshape(p_mask,[1,npairs,1,n_jsets,n_chans,1])   # ; print('p_mask_out',p_mask_out.get_shape()) # 1,pout,1,jset,cout,1
    
    XS_ext  = tf.reshape(XS_mat, [-1,1,1,n_jsets,n_chans,n_chans] ) # ; print('XS_ext',XS_ext.get_shape())
    XS_cpio =  XS_ext * p_mask_in * p_mask_out                      # ; print('XS_cpio',XS_cpio.get_shape())
    XSp_mat  = tf.reduce_sum(XS_cpio,[-3,-2,-1] )               # sum over jset,cout,cin, leaving ie,pout,pin
                            
    return(XSp_mat,XSp_tot) 

@tf.function
def LM2T_transformsTF(g_poles,E_poles,E_scat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles):
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
    A_mat = tf.linalg.inv(Ainv_mat)                                 # full inverse
    D_mat = tf.matmul( g_poles, tf.matmul( A_mat, g_poles) , transpose_a=True)     # print('D_mat',D_mat.dtype,D_mat.get_shape())
    
#    S_mat = Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2);
#  T=I-S
    T_mat = tf.eye(n_chans, dtype=CMPLX) - (Om2_mat + complex(0.,2.) * tf.expand_dims(POm_diag,3) * D_mat * tf.expand_dims(POm_diag,2) )
    
# multiply left and right by Coulomb phases:
    TC_mat = tf.expand_dims(CS_diag,3) * T_mat * tf.expand_dims(CS_diag,2)

    return(T_mat,TC_mat)
        
@tf.function
def T2B_transformsTF(T_mat,AA, n_jsets,n_chans):

# BB[ie,L] = sum(i,j) T[ie,i]* AA[i,L,j] T[ie,j]
#  T= T_mat[:,n_jsets,n_chans,n_chans]

    T_left = tf.reshape(T_mat,  [-1,n_jsets,n_chans,n_chans, 1, 1,1,1])
    T_right= tf.reshape(T_mat,  [-1,1,1,1, 1,  n_jsets,n_chans,n_chans])
    A_mid  = tf.reshape(AA, [1,n_jsets,n_chans,n_chans, -1, n_jsets,n_chans,n_chans] )
    TAT = tf.math.real( tf.math.conj(T_left) * A_mid * T_right )
    BB = tf.reduce_sum(TAT,[ 1,2,3, 5,6,7])    # exlude dim=0 (ie) and dim=4(L)
        
#     BB[:,:] = 0.0
#     for jset1 in range(n_jsets):
#         for c1 in range(n_chans):
#             for c1_out in range(n_chans):
#                 d  = 1.0 if c1==c1_out else 0.0
#                 for jset2 in range(n_jsets):
#                     for c2 in range(n_chans):
#                         for c2_out in range(n_chans):
#                             d2 = 1.0 if c2==c2_out else 0.0
#                             
#                             for ie in range(n_energies):
#                                 T1 = T_mat_n[ie,jset1,c1_out,c1]
#                                 T2 = T_mat_n[ie,jset2,c2_out,c2]
#                                 BB[ie,:] +=  AA[jset2,c2_out,c2, :, jset1,c1_out,c1] * ( T1 * (T2.conjugate()) ).real
                                                            
    return(BB)
    
@tf.function
def B2A_transformsTF(BB_t, Pleg):
                    
    B = tf.expand_dims(BB_t,2)    # so ie,L,1
    A = tf.reduce_sum( B*Pleg, 1 ) # sum over L
    
#     ds = 0.0
#     for L in range(NL):  ds += BB[ie,L] * Pleg[L,ia] * scale           
                    
    return(A)                                    

def SPphiCoulombFunctions(E,rmass,radius,zazb,L):
    Lmax = L
    CF1_val =  numpy.zeros([Lmax+1], dtype=DBLE)
    CF2_val =  numpy.zeros([Lmax+1], dtype=CMPLX)

    if rmass !=0:
        k = cmath.sqrt(fmscal * rmass * E)
    else: # photon!
        k = E/hbc
    rho = k * radius
    if abs(rho) <1e-10: print('rho =',rho,'from E,k,r =',E,k,radius)
    eta  =  etacns * zazb * cmath.sqrt(rmass/E)
    if E < 0: eta = -eta  #  negative imaginary part for bound states
    PM   = complex(0.,1.); 
    EPS=1e-10; LIMIT = 2000000; ACC8 = 1e-12
    ZL = 0.0
    DL,ERR = cf2(rho,eta,ZL,PM,EPS,LIMIT,ACC8)
    CF2_val[0] = DL
    for LL in range(1,Lmax+1):
        RLsq = 1 + (eta/LL)**2
        SL   = LL/rho + eta/LL
        CF2_val[LL] = RLsq/( SL - CF2_val[LL-1]) - SL

    if E > 0.:
        CF1_val[Lmax] = cf1(rho.real,eta.real,Lmax,EPS,LIMIT)
        for LL in range(Lmax,0,-1):
            RLsq = 1 + (eta.real/LL)**2
            SL   = LL/rho.real + eta.real/LL
            CF1_val[LL-1] = SL - RLsq/( SL + CF1_val[LL]) 

    DL = CF2_val[L] * rho
    S = DL.real
    P = DL.imag
    F = CF1_val[L] * rho.real
    phi = - math.atan2(P, F - S)
    return(S,P,phi)
 
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
    grid += [lowBound, highBound, 0.0253e-6, 1e-11]   # region boundaries + thermal + low-E
    # if threshold reactions present, add dense grid at and above threshold
    for threshold in thresholds:
        grid += [threshold]
        grid += list(threshold + resonancePos * 1e-2)
    grid = sorted(set(grid))
    # toss any points outside of energy bounds:
    grid = grid[grid.index(lowBound) : grid.index(highBound)+1]
    return numpy.asarray(grid, dtype=DBLE)

                                        
def reconstructLegendre(gnd,base,verbose,debug,egrid,stride,angles,legendre,thin,reconstyle):

    PoPs = gnd.PoPs
    projectile = gnd.PoPs[gnd.projectile]
    target     = gnd.PoPs[gnd.target]
    elasticChannel = '%s + %s' % (gnd.projectile,gnd.target)
    if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
    if hasattr(target, 'nucleus'):     target = target.nucleus
    pZ = projectile.charge[0].value; tZ =  target.charge[0].value
    charged =  pZ*tZ != 0
    identicalParticles = gnd.projectile == gnd.target
    rStyle = reconstyle.label
    if debug: print("Charged-particle elastic:",charged,",  identical:",identicalParticles,' rStyle:',rStyle)
    
    recons = gnd.styles.findInstancesOfClassInChildren(stylesModule.CrossSectionReconstructed)
#   print('Recons 2:',recons)
    if len(recons) > 0:
        if len(recons) > 1: raise Exception('ERROR: protare with more than one reconstructed cross section style not supported.')
#       print('Remove style',recons[0].label,', to be replaced.')
        gnd.removeStyle(recons[0].label)
            
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.getScatteringRadius()
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes
    brune = BC==resolvedResonanceModule.BoundaryCondition.Brune
    MatrixL = False
    if brune: MatrixL = True
    print('BC =',BC, ' brune =',brune,'MatrixL',MatrixL)
    
    if angles is not None:
        thmin = angles[0]
        thinc = angles[1]
        if charged:
            from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import CoulombPlusNuclearElastic as CoulombPlusNuclearElasticModule
            from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import nuclearPlusInterference as nuclearPlusInterferenceModule
        from fudge.productData.distributions import reference as referenceModule
        muCutoff = math.cos(thmin*pi/180.)
    # 
    if legendre is not None:
        from fudge.productData.distributions import angular as angularModule
        if charged:
            from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import CoulombPlusNuclearElastic as CoulombPlusNuclearElasticModule
            from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import nuclearAmplitudeExpansion as nuclearAmplitudeExpansionModule
            from fudge.productData.distributions import reference as referenceModule

    accuracy = None

    n_jsets = len(RMatrix.spinGroups)
    n_poles = 0
    n_angles = 0     # angles
    n_chans = 0
    
    np = len(RMatrix.resonanceReactions)
    ReichMoore = False
    if RMatrix.resonanceReactions[0].eliminated: 
        ReichMoore = True
        np -= 1   # exclude Reich-Moore channel from scattering channels
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
    calcP = [True] * np

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
        reaction = partition.link.link
        calcP[pair] = 'fission' not in kp # or partition.calculatePenetrability]
#         print('Partition:',kp,'P?',calcP[pair])
        if calcP[pair]:
            p,t = partition.ejectile,partition.residual
            projectile = PoPs[p];
            target     = PoPs[t];
            pMass = projectile.getMass('amu');   tMass =     target.getMass('amu');
            rmass[pair] = pMass * tMass / (pMass + tMass)
            if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
            if hasattr(target, 'nucleus'):     target = target.nucleus

            za[pair]  = projectile.charge[0].value;  
            zb[pair]  = target.charge[0].value
            
            jp[pair],pt[pair],ep[pair] = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
            try:
                jt[pair],tt[pair],et[pair] = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].pqu('MeV').value
            except:
                jt[pair],tt[pair],et[pair] = 0.,1,0.
            
        if partition.Q is not None:
            QI[pair] = partition.Q.getConstantAs('MeV')
        else:
            QI[pair] = reaction.getQ('MeV')
        if partition.scatteringRadius is not None:
            prmax[pair] =  partition.getScatteringRadius().getValueAs('fm')
        else:
            prmax[pair] = Rm_global
        if partition.label == elasticChannel:
            lab2cm = tMass / (pMass + tMass)
            w_factor = 1. #/lab2cm**0.5 if IFG else 1.0
            ipair = pair  # incoming
            
        print(pair,":",kp,rmass[pair],QI[pair],prmax[pair])
        pair += 1
    print("\nElastic channel is",elasticChannel,'so w factor=',w_factor,'as IFG=',IFG)
    npairs  = pair

#  FIRST: for array sizes:
    Lmax = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        n_poles = max(n_poles,R.nRows)
        n = R.nColumns-1
        if ReichMoore: n -= 1
        n_chans = max(n_chans,n)
        for ch in Jpi.channels:
            Lmax = max(Lmax,ch.L)

    E_poles = numpy.zeros([n_jsets,n_poles], dtype=DBLE)
    E_damping = numpy.zeros([n_jsets,n_poles], dtype=DBLE)
    g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
#     P_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
    B_chans = numpy.zeros([n_jsets,n_chans], dtype=DBLE)
    has_widths = numpy.zeros([n_jsets,n_poles], dtype=INT)
    J_set = numpy.zeros(n_jsets, dtype=DBLE)
    pi_set = numpy.zeros(n_jsets, dtype=INT)
    L_val  =  numpy.zeros([n_jsets,n_chans], dtype=INT)
    S_val  =  numpy.zeros([n_jsets,n_chans], dtype=DBLE)
    p_mask =  numpy.zeros([npairs,n_jsets,n_chans], dtype=DBLE)
    seg_val=  numpy.zeros([n_jsets,n_chans], dtype=INT) - 1 
    seg_col=  numpy.zeros([n_jsets], dtype=INT) 
    seg_row=  numpy.zeros([n_jsets], dtype=INT) 
    Spins = [set() for pair in range(npairs)]

# Set up pole data per spingroup and channel:
    jset = 0
    All_spins = set()
    for Jpi in RMatrix.spinGroups:
        J_set[jset] = Jpi.spin
        pi_set[jset] = Jpi.parity
        R = Jpi.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns - 1  # ignore energy col
        seg_col[jset] = cols
        seg_row[jset] = rows
#         print('J,pi =',J_set[jset],pi_set[jset],'R,C =',rows,cols)

        E_poles[jset,:rows] = numpy.asarray( R.getColumn('energy','MeV') , dtype=DBLE)   # lab MeV
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
        
        if ReichMoore: E_damping[jset,:rows] = numpy.asarray(widths[0][:],  dtype=DBLE)
        if IFG==1:     E_damping[jset,:] = 2*E_damping[jset,:]**2            
        if ReichMoore and debug: print('Set',jset,'radiative damping',E_damping[jset,:rows])
        
        n = 0
        for ch in Jpi.channels:
            rr = ch.resonanceReaction
            pair = partitions.get(rr,None)
            if pair is None: continue
            m = ch.columnIndex - 1
            g_poles[jset,:rows,n] = numpy.asarray(widths[m][:],  dtype=DBLE) * w_factor

            L_val[jset,n] = ch.L
            S = float(ch.channelSpin)
            S_val[jset,n] = S
            has_widths[jset,:rows] = 1
            
            seg_val[jset,n] = pair
            p_mask[pair,jset,n] = 1.0
            Spins[pair].add(S)
            All_spins.add(S)
            n += 1
            
        jset += 1

#               convert IFG=1 to rwa.   Duplicate IFG=0 to FPWid for energy grid construction
    EFwidths = []
    jset = 0
    for Jpi in RMatrix.spinGroups:
#         print('J,pMax',jset,seg_row[jset])
        for p in range(seg_row[jset]):
            Fwid = 0.0
            obsEnergy = E_poles[jset,p]

            n = 0
            for ch in Jpi.channels:
                rr = ch.resonanceReaction
                pair = partitions.get(rr,None)
                if pair is None: continue
                chPrmax  = ch.getScatteringRadius().getValueAs('fm')
            
                if BC == resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
                    B = -ch.L
                elif BC == resolvedResonanceModule.BoundaryCondition.Brune:
                    B = 0 # not used
                elif BC == resolvedResonanceModule.BoundaryCondition.Given:                     # btype='B'
                    B = BV
                elif BC == resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction:    # btype='S'
                    B = None
                if ch.boundaryConditionValue is not None:
                    B = float(ch.boundaryConditionValue)
                B_chans[jset,n] = B

                if calcP[pair]:
                    E = E_poles[jset,p]*lab2cm + QI[pair]
#                     print('Pole spec',jset,p,'E,Rm,zz,L=', E_poles[jset,p],abs(E),rmass[pair],chPrmax,za[pair]*zb[pair],ch.L)
                    S,P,phi = SPphiCoulombFunctions(abs(E),rmass[pair],chPrmax,za[pair]*zb[pair],ch.L)
                else:
                    P = 1.0
                    S = 0.0
#                 P_poles[jset,p,n] = P
                if B is None: B_chans[jset,n] = S   # B=S approximation

                if not IFG: # convert from 'ENDF' width to RWA
                    FPwid = g_poles[jset,p,n]
                    g_poles[jset,p,n] = (abs(FPwid)/(2*P))**0.5 * (1 if FPwid > 0 else -1)
                else:  # IFG=1: given rwa, convert to formal widths - needed for energy grid
                    FPwid = 2 * g_poles[jset,p,n]**2 * P  
            
                Fwid += abs(FPwid)
                if not brune:
                    obsEnergy -= g_poles[jset,p,n]**2 * (S -  B_chans[jset,n])
                    if verbose: print('Pole at E=',obsEnergy,'from',E_poles[jset,p],'in channel',n,'has partial width',FPwid,'summing to',Fwid)
                n += 1
            EFwidths.append((obsEnergy,Fwid))    # TEMP: replace Fwid later by obsWid
            
        if verbose:
            print('J set %i: E_poles \n' % jset,E_poles[jset,:seg_col[jset]])
            print('E_damp  \n',E_damping[jset,:seg_col[jset]])
            print('g_poles \n',g_poles[jset,:seg_row[jset],:seg_col[jset]])
#             print('P_poles \n',P_poles[jset,:seg_row[jset],:seg_col[jset]])
        jset += 1        

#    print('All spins:',All_spins)
#    print('All channel spins',Spins)
    
    
    if brune:  # S_poles: Shift functions at pole positions for Brune basis   
        S_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
        dSdE_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
#         EO_poles =  numpy.zeros([n_jsets,n_poles], dtype=DBLE) 
        EO_poles = E_poles.copy()
        Pole_Shifts(S_poles,dSdE_poles, EO_poles,has_widths,   seg_val,lab2cm,QI,fmscal,rmass,prmax, etacns,za,zb,L_val) 
    else:
        S_poles = None
        dSdE_poles = None
        EO_poles = None

# NOW MAKE ENERGY GRID

    de = egrid
    if egrid > 0:       # uniform grid
        emin = max(emin,de)
        n_energies = int( (emax-emin)/de + 1.5)
        de = (emax-emin)/(n_energies-1)
        print('Reconstruction emin,emax =',emin,emax,'with',n_energies,'spaced at',de,'MeV')
        E_scat  = numpy.asarray([emin + ie*de for ie in range(n_energies)], dtype=DBLE)
        
    else:           # non-uniform grid based on resooance formal widths
        EFwidths.sort(key = lambda x: x[0])            
        if verbose: print('Observed energies + Formal widths:\n',EFwidths)
    #     print('Radiative damping',E_damping)
#         print('Energies + Formal widths sorted:')
#         for e,w in EFwidths:  print(' E = %10.6f, w = %10.6f' % (e,w))
        Penergies,Fwidths = zip(*EFwidths)

        E_scat = generateEnergyGrid(Penergies,Fwidths, emin,emax, stride=stride)
        n_energies = len(E_scat)    
    
    
  #  print('Energy grid (lab):',E_scat)
    print('Need %i energies in %i Jpi sets with %i poles max, and %i channels max. Lmax=%i' % (n_energies,n_jsets,n_poles,n_chans,Lmax))
 
    rksq_val  = numpy.zeros([n_energies,npairs], dtype=DBLE)
#   velocity  = numpy.zeros([n_energies,npairs], dtype=DBLE)
    
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

      
#  Calculate Coulomb functions    
    for pair in range(npairs):
#         print('Partition',pair,'Q,mu:',QI[pair],rmass[pair])
#         if not calcP[pair]:
        Csig_exp[:,pair,:] = 1.0
        if not calcP[pair]:
            CF2_val[:,pair,:] = complex(0.,1.)
        else:
                
            if debug:
                foutS = open(base + '+3-S%i' % pair,'w')
                foutP = open(base + '+3-P%i' % pair,'w')
            for ie in range(n_energies):
                E = E_scat[ie]*lab2cm + QI[pair]
                if rmass[pair]!=0:
                    k = cmath.sqrt(fmscal * rmass[pair] * E)
                else: # photon!
                    k = E/hbc
                    
                rho = k * prmax[pair]
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
                rksq_val[ie,pair] = 1./max(abs(k)**2, 1e-20) 
                eta_val[ie,pair] = eta.real
            
                if E > 0.:
                    csigma_v[ie,pair,:] = csigma(Lmax,eta)
                    for L in range(Lmax+1):
                        Csig_exp[ie,pair,L] = cmath.exp(complex(0.,csigma_v[ie,pair,L]-csigma_v[ie,pair,0]))
                if debug:
                    L = 3 # for printing
                    EE = E  # or E_scat[ie]
                    print(EE,CF2_val[ie,pair,L].real, file=foutS)
                    print(EE,CF2_val[ie,pair,L].imag, file=foutP)
        if debug:
            foutS.close()
            foutP.close()

# Allocate Coulomb functions to partial-wave channels
    jset = 0
    for Jpi in RMatrix.spinGroups:
        n = 0
        for ch in Jpi.channels:
            rr = ch.resonanceReaction
            pair = partitions.get(rr,None)
            if pair is None: continue
            chPrmax  = ch.getScatteringRadius().getValueAs('fm')
            chHsprad = ch.getHardSphereRadius().getValueAs('fm')
            chHsprad  = chPrmax # TEST
            revRR = abs(chPrmax - prmax[pair]) > 1e-6 and calcP[pair]  # calculate local scatteringRadius != partition radius
            revHR = abs(chHsprad - chPrmax   ) > 1e-6 and calcP[pair]  # calculate local hardSphereRadius != local scatteringRadius
            
            m = ch.columnIndex - 1

        # Find S and P:
            for ie in range(n_energies):
                E = E_scat[ie]*lab2cm + QI[pair]

                if revRR: # use local radius if necessary
                    S,P,phi = SPphiCoulombFunctions(E,rmass[pair],chPrmax,za[pair]*zb[pair],L_val[jset,n])
                else:  # use partition radius precalculated
                    DL = CF2_val[ie,pair,L_val[jset,n]]
                    S = DL.real
                    P = DL.imag
                    F = CF1_val[ie,pair,L_val[jset,n]]
                    phi = - math.atan2(P, F - S)
                    
                if B is None:
                    L_diag[ie,jset,n]       = complex(0.,P)
                else:
                    L_diag[ie,jset,n]       = DL - B_chans[jset,n]
                Psr = math.sqrt(abs(P))

                if revHR and E>0:     # tediously & reluctantly recalculate hard-sphere phase shifts for scattering states
                    Sh,Ph,phi = SPphiCoulombFunctions(E,rmass[pair],chHsprad,za[pair]*zb[pair],L_val[jset,n])

                Omega = cmath.exp(complex(0,phi))
                POm_diag[ie,jset,n]      = Psr * Omega
                Om2_mat[ie,jset,n,n]     = Omega**2
                CS_diag[ie,jset,n]       = Csig_exp[ie,pair,L_val[jset,n]]

            n += 1      # channel
        jset += 1       # spin group
            
###     TENSORFLOW code:

    E_cpoles = tf.complex(E_poles,-E_damping*0.5) 
    g_cpoles = tf.complex(g_poles,tf.constant(0., dtype=DBLE))
    E_cscat = tf.complex(E_scat,tf.constant(0., dtype=DBLE)) 
    
    if not MatrixL:
        T_mat,TC_mat = R2T_transformsTF (g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 
    else:
        T_mat,TC_mat = LM2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans,brune,S_poles,dSdE_poles,EO_poles ) 


    if debug:
        for ie in range(n_energies):
            for jset in range(n_jsets):
                print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n T-matrix is size',seg_col[jset])
                for a in range(n_chans):
                    print('   ',a,'row: ',',  '.join(['{:.5f}'.format(T_mat[ie,jset,a,b].numpy()) for b in range(n_chans)]) )

    
    gfac = numpy.zeros([n_energies,n_jsets,n_chans])
    for jset in range(n_jsets):
        for c_in in range(n_chans):   # incoming partial wave
            pair = seg_val[jset,c_in]      # incoming partition
            if pair>=0:
                denom = (2.*jp[ipair]+1.) * (2.*jt[ipair]+1)
                for ie in range(n_energies):
                    gfac[ie,jset,c_in] = pi * (2*J_set[jset]+1) * rksq_val[ie,pair] / denom 

    XSp_mat,XSp_tot  = T2X_transformsTF(T_mat,gfac,p_mask, n_jsets,n_chans,npairs)
    
    XSp_mat_n,XSp_tot_n = XSp_mat.numpy(),XSp_tot.numpy()

    print()   
    for pair in range(npairs):
        fname = base + '-tot_%i' % pair
        print('Total cross-sections for incoming',pair,'to file',fname)
        fout = open(fname,'w')
        for ie in range(n_energies):
            x = XSp_tot_n[ie,pair] * 10.  # mb
            E = E_scat[ie]      # lab incident energy
            print(E,x, file=fout)
        fout.close()

        for pout in range(npairs):
            fname = base + '-ch_%i-to-%i' % (pair,pout)
            print('Partition',pair,'to',pout,': angle-integrated cross-sections to file',fname)
            fout = open(fname,'w')
            for ie in range(n_energies):
                x = XSp_mat_n[ie,pout,pair] * 10.
                E = E_scat[ie]
                print(E,x, file=fout)
            fout.close()

    if angles is not None or legendre is not None:

        from numericalFunctions import angularMomentumCoupling
        from xData.series1d  import Legendre

        if angles is not None:
            print('angles:',angles)
            na = int( (180.0 - thmin)/thinc + 0.5) + 1
            NL = 2*Lmax + 1
        
            Pleg = numpy.zeros([NL,na])
            mu_vals = numpy.zeros(na)
            xsc = numpy.zeros(na)
            Rutherford = numpy.zeros(na)
        
            for ia in range(na):
                theta =thmin + ia*thinc
                thrad = theta*pi/180.
                mu = math.cos(thrad)
                mu_vals[ia] = mu
                for L in range(NL):
                    Pleg[L,ia] = Legendre(L, mu)

            print('\n# angles=',na,' to L=',NL)
        NLB = 2*Lmax + 1
        if legendre is not None:
            NL = min(Lmax,legendre) 
            NLA = NL + 1
            print('\n# legendre expansions up to NL=',NL,' (# NLB =',NLB,')')
    
        NS = len(All_spins)
        ZZbar = numpy.zeros([NLB,NS,n_jsets,n_chans,n_jsets,n_chans])

        def n2(x): return(int(2*x + 0.5))
        def i2(i): return(2*i)
        def triangle(x,y,z): return (  abs(x-y) <= z <= x+y )

        for iS,S in enumerate(All_spins):
            for jset1 in range(n_jsets):
                J1 = J_set[jset1]
                for c1 in range(n_chans):
                    L1 = L_val[jset1,c1]
                    if not triangle( L1, S, J1) : continue

                    for jset2 in range(n_jsets):
                        J2 = J_set[jset2]
                        for c2 in range(n_chans):
                            L2 = L_val[jset2,c2]
                            if not triangle( L2, S, J2) : continue

                            for L in range(NLB):                    
                                ZZbar[L,iS,jset2,c2,jset1,c1] = angularMomentumCoupling.zbar_coefficient(i2(L1),n2(J1),i2(L2),n2(J2),n2(S),i2(L))

        if legendre is not None:
            nuclear = [ [] for pair in range(np)] 
            interferenceReal = [ [] for pair in range(np)] 
            interferenceImaginary = [ [] for pair in range(np)] 
            crossSection = [ [] for pair in range(np)] 
                    
    
        # calculate angular distributiones here. Later move to TF kernel.
        sigdd = {}    
        pair = 0
        rr = None
        T_mat_n = T_mat.numpy()
        BB = numpy.zeros([n_energies,NLB])
        AA = numpy.zeros([np, n_jsets,n_chans,n_chans, NLB, n_jsets,n_chans,n_chans], dtype=CMPLX)
        sig_ni = numpy.zeros(n_energies)
        pair = 0
        for rr_out in RMatrix.resonanceReactions:
            if not rr_out.eliminated:
                elastic = rr_out.label == elasticChannel
                reaction = rr_out.link.link
                AA[pair,:,:,:, :, :,:,:] = 0.0
                sigdd[rr_out.label] = []
                for S_out in Spins[pair]:
                    for S_in in Spins[ipair]:
                        for iS,S in enumerate(All_spins):
                            for iSo,So in enumerate(All_spins):
                                if abs(S-S_in)>0.1 or abs(So-S_out)>0.1: continue
                                phase = (-1)**int(So-S) / 4.0
                                if debug: print('\n *** So=%4.1f <- S=%4.1f:' % (So,S), '(',rr_out.label,pair,'<-',ipair,')')
                                for jset1 in range(n_jsets):
                                    J1 = J_set[jset1]
                                    for c1 in range(n_chans):
                                        if seg_val[jset1,c1] != ipair: continue
                                        if abs(S_val[jset1,c1]-S) > 0.1 : continue

                                        L1 = L_val[jset1,c1]
                                        for c1_out in range(n_chans):
                                            if seg_val[jset1,c1_out] != pair: continue
                                            if abs(S_val[jset1,c1_out]-So) > 0.1 : continue
                                            L1_out = L_val[jset1,c1_out]
                                            d  = 1.0 if c1==c1_out else 0.0

                                            for jset2 in range(n_jsets):
                                                J2 = J_set[jset2]
                                                for c2 in range(n_chans):
                                                    if seg_val[jset2,c2] != ipair: continue
                                                    if abs(S_val[jset2,c2]-S) > 0.1 : continue
    #                                                 print('    Conj: J,c =',J2,c2)
                                                    L2 = L_val[jset2,c2]
                                                    for c2_out in range(n_chans):
                                                        if seg_val[jset2,c2_out] != pair: continue
                                                        if abs(S_val[jset2,c2_out]-So) > 0.1 : continue
                                                        L2_out = L_val[jset2,c2_out]
                                                        d2 = 1.0 if c2==c2_out else 0.0

                                                        for L in range(NLB):
                                                            ZZ = ZZbar[L,iS,jset2,c2,jset1,c1] * ZZbar[L,iSo,jset2,c2_out,jset1,c1_out] 
                                                            AA[pair, jset2,c2_out,c2, L, jset1,c1_out,c1] += phase * ZZ 
                
                BB_t = T2B_transformsTF(TC_mat,AA[pair, :,:,:, :, :,:,:], n_jsets,n_chans)
                BB = BB_t.numpy()
                TC_mat_n = TC_mat.numpy()
            
                if angles is not None:
                    A_t = B2A_transformsTF(BB_t, Pleg)
                    Angular= A_t.numpy()
            
                CNamp = pair==ipair and charged
                for ie in range(n_energies):
                    E = E_scat[ie]   # lab incident energy
                    if E < reaction.domainMin - de: continue
                    
                    denom = (2.*jp[ipair]+1.) * (2.*jt[ipair]+1)
                    gfacc_b =  pi * rksq_val[ie,ipair] / denom * 0.01  # b
                    gfacc =  gfacc_b  * 1e3   # mb
                    xs = BB[ie,0]/pi * gfacc * 4*pi

                    dist = []
                    scale = 0.5 / BB[ie,0] if BB[ie,0] !=0 else 1.0
                    if CNamp: sig_ni[ie] = 0.0
                    
                    if legendre is not None:
                        nuclear_E = []; 
                        maxIndex = 0

                        if not CNamp:  # add in Rutherford + interference terms
                            if BB[ie,0] == 0.0: BB[ie,0] = 1.0  # always have distribution even if xs = 0
                            for L in range(NLB):
    #                             nuclear_E.append(BB[ie,L] * 2 * gfacc_b / (2*L+1) / pi)  # not normalized to a0 = 1, but xs/2pi
                                C = BB[ie,L] / (2*L+1) / BB[ie,0]  # normalized to a0=1
                                if  abs(C) > minLegendreValue or L <= 1:
                                      rounded = round(C, logMinLegendreValue + extraFloatPrecision)
                                      nuclear_E.append(rounded)  
                        else:  # CN:: NOT normalized to a0=1 !!
                            for L in range(NLB):
                                nuclear_E.append(BB[ie,L] * 2 * gfacc_b / (2*L+1) / pi)  # not normalized to a0 = 1, but xs/2pi                        
                        
                        nuclear[pair].append(nuclear_E)
                            
                        if CNamp:  # add in Rutherford + interference terms
                            intReal_E = numpy.zeros(NLA)
                            intImag_E = numpy.zeros(NLA)
                            for jset in range(n_jsets):
                                J = J_set[jset]
                                for c in range(n_chans):
                                    if seg_val[jset,c] != ipair: continue
                                    L = L_val[jset,c]
                                    if L >= NL: continue
                                    fac = - gfacc_b * (2*J+1)/(2*L+1) / pi   # * i * TC.conj
                                    intReal_E[L] += fac * TC_mat_n[ie,jset,c,c].imag
                                    intImag_E[L] -= fac * TC_mat_n[ie,jset,c,c].real 
                            interferenceReal[pair].append(intReal_E)
                            interferenceImaginary[pair].append(intImag_E)                                        

                        crossSection[pair].append(xs)  # do not use for charged elastic!!
                        

                    if angles is not None:
                        mulast = 1.0
                        for ia in range(na):
                            theta = thmin + ia*thinc
                            thrad = theta*pi/180.
                            mu = math.cos(thrad)
                            munext = math.cos((theta+thinc)*pi/180.)
                    
                            if pair==ipair and charged:  # add in Rutherford + interference terms
                                eta = eta_val[ie,ipair]
                                shth = math.sin(thrad*0.5)
                                Coulmod = eta.real * rsqr4pi / shth**2
                                CoulAmpl = Coulmod * cmath.exp(complex(0.,-2*eta.real*math.log(shth) ))
                        
                                CT = denom * Coulmod**2
                        
                                IT = 0.0
                                for jset in range(n_jsets):
                                    J = J_set[jset]
                                    for c in range(n_chans):
                                        if seg_val[jset,c] != ipair: continue
                                        L = L_val[jset,c]
                                        IT += (2*J+1) * Pleg[L,ia] * 2 * (- CoulAmpl * TC_mat_n[ie,jset,c,c].conjugate()).imag * rsqr4pi
                                                
                                RT = Angular[ie,ia] / pi
                                xsc[ia]        = gfacc * (CT + IT + RT)
                                Rutherford[ia] = gfacc *  CT
                                NI             = gfacc * (     IT + RT)
    #                            sig_ni[ie] +=  NI * (mulast - mu) * 2*pi
                                sig_ni[ie] +=  NI * (mulast - munext)/2. * 2*pi
                            else:
                                ds = Angular[ie,ia] * scale        
                    
                                dist.insert(0,ds)
                                dist.insert(0,mu)
                                theta = thmin + ia*thinc
                            mulast = mu
                        if pair==ipair and charged:  # find normalized difference
                            for ia in range(na):
                                mu = mu_vals[ia]
                                ds = 2*pi*( xsc[ia] - Rutherford[ia] ) / sig_ni[ie]
                                dist.insert(0,ds)
                                dist.insert(0,mu)
                                theta = thmin + ia*thinc
                                thrad = theta*pi/180.
                                mu = math.cos(thrad)

                        sigdd[rr_out.label].append([E,dist])
                               
                pair += 1
    
## # PROCESS CROSS-SECTIONS
    
                    
    egrid = E_scat[:]    # lab MeV
    totalxs = XSp_tot_n[:,ipair] * 0.01   # barns
    if charged and (angles is not None):
        elasticxs = sig_ni[:] * 1e-3 # barns not mb
    elif charged and (legendre is not None):
        elasticxs = numpy.ones(n_energies)  
    else:
        elasticxs = XSp_mat_n[:,ipair,ipair] * 0.01 # barns
    fissionxs = numpy.zeros(n_energies)
    absorbtionxs = totalxs - numpy.sum(XSp_mat_n[:,:,ipair], axis=1)*0.01  # barns
    chanxs = []
    for pout in range(npairs):
        if pout == ipair:    # elastic
            chanxs.append(elasticxs)
        else:                # other channels
            chanxs.append( XSp_mat_n[:,pout,ipair] * 0.01)
        if debug: print('chanxs',pout,ipair,':',chanxs[-1])

    crossSectionAxes = crossSectionModule.defaultAxes( 'MeV' )
    total = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, totalxs), dataForm="XsAndYs" )
    elastic = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, elasticxs), dataForm="XsAndYs" )
    fission = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, fissionxs), dataForm="XsAndYs" )
    absorbtion = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, absorbtionxs), dataForm="XsAndYs" )

    if not isinstance( reconstyle, stylesModule.CrossSectionReconstructed ):
        raise TypeError("style must be an instance of crossSectionReconstructed, not %s" % type(reconstyle))

    haveEliminated = False
    for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
        reaction = rreac.link.link
        haveEliminated = haveEliminated or rreac.eliminated
        #                  elastic or capture 
        if reaction == gnd.getReaction('capture'): rreac.tag = 'capture'
        elif reaction == gnd.getReaction('elastic'): rreac.tag = 'elastic'
        elif 'fission' in rreac.label: rreac.tag = rreac.label
        else: rreac.tag = 'competitive'
                
    xsecs = {'total':total, 'elastic':elastic, 'fission':fission, 'nonelastic':absorbtion}
#     print('channels:',list(channels.keys() )) 
#     print('# chanxs:',len(chanxs))
    for c in range(npairs):  
        if debug: print('\nXS',channels[c],'is',chanxs[c],' (Elastic:',channels[c] == elasticChannel,')' )
        xsecs[channels[c]] = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, chanxs[c]), dataForm="XsAndYs" )

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
#     
#     recons = gnd.styles.findInstancesOfClassInChildren(stylesModule.CrossSectionReconstructed)
#     print('Recons 2 for',reaction,':',recons)
#     if len(recons) > 0:
#         if len(recons) > 1: raise Exception('ERROR: protare with more than one reconstructed cross section style not supported.')
#         print('Remove style',recons[0].label,', to be replaced.')
#         gnd.removeStyle(recons[0].label)
#             
            
    for reaction in gnd :
        if isinstance( reaction, sumsModule.MultiplicitySum ): continue
        iselastic = reaction is elasticChannel

        evaluatedCrossSection = reaction.crossSection.evaluated
        if not isinstance( evaluatedCrossSection, crossSectionModule.ResonancesWithBackground ):
#             print('Channel',reaction.label,'is not crossSectionModule.ResonancesWithBackground')
            continue
        # which reconstructed cross section corresponds to this reaction?
        if( derivedFromLabel == '' ) : derivedFromLabel = evaluatedCrossSection.label
        if( derivedFromLabel != evaluatedCrossSection.label ) :
            print(('WARNING derivedFromLabel = "%s" != "%s"' % (derivedFromLabel, evaluatedCrossSection.label)))
#         print('Look for recon',reaction.label,'in',xsecs.keys(),':',str( reaction ) in xsecs)
        RRxsec = None
        if str( reaction ) in xsecs:
            RRxsec = xsecs[ str( reaction ) ]
#             print('Got', str(reaction) )
        else :
            for possibleChannel in possibleChannels :
                if( possibleChannels[possibleChannel] ) :
                    if( possibleChannel in str( reaction ) ) : 
                        RRxsec = xsecs[possibleChannel]
#                         print('Got',str(reaction),'as',possibleChannel)
                if( RRxsec is None ) :
                    if( reaction is gnd.getReaction( possibleChannel ) ) : 
#                       RRxsec = xsecs[possibleChannel]
                        RRxsec = xsecs.get(possibleChannel,None)
                if( RRxsec is not None ) : break
        if( RRxsec is None ) :
            if True:
                print(( "Error: couldn't find reconstructed cross section to add to reaction %s" % reaction ))
            continue

        background = evaluatedCrossSection.background
        background = background.toPointwise_withLinearXYs( accuracy = 1e-3, lowerEps = epsilon, upperEps = epsilon )
        RRxsec = RRxsec.toPointwise_withLinearXYs( accuracy = 1e-3, lowerEps = epsilon, upperEps = epsilon )
        RRxsec.convertUnits( {RRxsec.domainUnit: background.domainUnit,  RRxsec.rangeUnit: background.rangeUnit } )

        background, RRxsec = background.mutualify(0,0,0, RRxsec, -epsilon,epsilon,True)
        RRxsec = background + RRxsec    # result is a crossSection.XYs1d instance
        if thin:
            RRx = RRxsec.thin( accuracy or .001 )
        else:
            RRx = RRxsec
        RRx.label = rStyle

        reaction.crossSection.add( RRx )
       
        if debug: print("Channels ",reaction.label,iselastic,":\n",RRxsec.toString(),"\n&\n",RRx.toString())
        if iselastic:
            effXsc = RRxsec
            
    gnd.styles.add( reconstyle )

## # PROCESS DISTRiBUTIONS

    if legendre is not None:
        pair = 0
        for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
            if rreac.eliminated or not calcP[pair]: continue
            productName = rreac.ejectile
            elastic = productName == gnd.projectile and rreac.residual == gnd.target
            print("Add angular distribution for",productName," in",rreac.label,"channel (elastic=",elastic,")")

            reaction = rreac.link.link
            firstProduct = reaction.outputChannel.getProductWithName(productName)

            reconDist = firstProduct.distribution['recon']
            
            angularAxes = distributionsModule.angular.defaultAxes( 'MeV' )  # for Elab outerDomainValue
            dist = distributionsModule.angular.XYs2d( axes = angularAxes )
            
            elab_max = 0.; elab_min = 1e10
            iee = 0
            for ie in range(n_energies):
                elab = E_scat[ie]
                elab_max = max(elab,elab_max); elab_min = min(elab,elab_min)
                if elab < reaction.domainMin - de: continue
                angdist = distributionsModule.angular.Legendre( coefficients = nuclear[pair][iee], outerDomainValue = elab, axes = angularAxes ) 
                dist.append(angdist)
                iee += 1
                
#             print('Channel:',rreac.label,charged,elastic)

            if charged and elastic:
                
                interferenceRealDist = distributionsModule.angular.XYs2d( axes = angularAxes )
                interferenceImaginaryDist = distributionsModule.angular.XYs2d( axes = angularAxes )
                
                for ie in range(n_energies):
                    elab = E_scat[ie]   
                    interferenceRealDist.append(distributionsModule.angular.Legendre( coefficients = interferenceReal[pair][ie], outerDomainValue = elab, axes = angularAxes ))         
                    interferenceImaginaryDist.append(distributionsModule.angular.Legendre( coefficients = interferenceImaginary[pair][ie], outerDomainValue = elab, axes = angularAxes )) 
                            
                nuclearPart      = nuclearAmplitudeExpansionModule.NuclearTerm(  dist )
                interferenceRealPart = nuclearAmplitudeExpansionModule.RealInterferenceTerm( interferenceRealDist )
                interferenceImaginaryPart = nuclearAmplitudeExpansionModule.ImaginaryInterferenceTerm( interferenceImaginaryDist )
                               
                nuclearAmplitudeExpansion = nuclearAmplitudeExpansionModule.NuclearAmplitudeExpansion(
                            nuclearTerm=nuclearPart, realInterference=interferenceRealPart, imaginaryInterference=interferenceImaginaryPart )
                            
                dSigma_form = CoulombPlusNuclearElasticModule.Form( productName, rStyle, nuclearPlusInterference = None,
                        nuclearAmplitudeExpansion = nuclearAmplitudeExpansion, identicalParticles = identicalParticles )
                reaction.doubleDifferentialCrossSection.add( dSigma_form )

                reaction.crossSection.remove( rStyle )
                reaction.crossSection.add( crossSectionModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )
                # also make a link from 'normal' distribution to differential part:
                firstProduct.distribution.remove( rStyle )
                firstProduct.distribution.add( referenceModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )
                    
            else:  # non-elastic ejectile

                newForm = distributionsModule.angular.TwoBody( label = rStyle, productFrame = xDataEnumsModule.Frame.centerOfMass, angularSubform = dist )

                evalDist = firstProduct.distribution['eval']   # existing distributions
                
                if isinstance(evalDist,unspecifiedModule.Form):
                    firstProduct.distribution.add( newForm )
                else:
                    if evalDist.domainMin < newForm.domainMin:    
                        newForm.insert(0, newForm[0].__class__([1], outerDomainValue=evalDist.domainMin))

                    reconData = newForm.data
                    regions2d = angularModule.Regions2d(axes=reconData.axes)
                    regions2d.append(reconData)
                
                    evalData = evalDist.data
                    regions2d.append(evalData.domainSlice(reconData.domainMax))
                    print('Angular regions for distributions split at',reconData.domainMax)
                    newRecon = reconDist.__class__(reconDist.label, reconDist.productFrame, regions2d)
                    firstProduct.distribution.replace(newRecon)
            pair += 1
                    
    if angles is None: return
        
    angularAxes = distributionsModule.angular.defaultAxes( 'MeV' )  # for Elab outerDomainValue
    pair = 0
    for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
        if not rreac.eliminated and calcP[pair]:
            productName = rreac.ejectile
            elastic = productName == gnd.projectile and rreac.residual == gnd.target
            print("Add angular distribution for",productName," in",rreac.label,"channel (elastic=",elastic,'Style=',rStyle,")")

            reaction = rreac.link.link
            firstProduct = reaction.outputChannel.getProductWithName(productName)
            evalDist = firstProduct.distribution['eval']
            evalData = evalDist.data
            reconDist = firstProduct.distribution['recon']

            effDist = distributionsModule.angular.XYs2d( axes = angularAxes )

            elab_max = 0.; elab_min = 1e10; nangles=0
            ne = 0
            for elab,dist in sigdd[rreac.label]:
                if debug: print('E=',elab,'has',len(dist),' angles')
                if len(dist) <= 3: 
                    print('   E=',elab,'has',len(dist),' angles')
                    continue
                angdist = distributionsModule.angular.XYs1d( data = dist, outerDomainValue = elab, axes = angularAxes, dataForm = 'list' ) 
                if thin:
                    angdist = angdist.thin( accuracy or .001 )
                norm = angdist.integrate()
                if norm != 0.0:
                    if debug: print(rreac.label,elab,norm)
                    effDist.append( angdist ) 
                elab_max = max(elab,elab_max); elab_min = min(elab,elab_min); nangles = max(len(dist)//2,nangles)
                ne += 1
            print("   Angles reconstructed at %i energies from %s to %s MeV with up to %i angles at each energy" % (ne,elab_min,elab_max,nangles))

            newForm = distributionsModule.angular.TwoBody( label = rStyle,
                productFrame = firstProduct.distribution.evaluated.productFrame, angularSubform = effDist )
                
            if evalDist.domainMin < newForm.domainMin:    
                newForm.insert(0, newForm[0].__class__([1], outerDomainValue=evalDist.domainMin))

            regions2d = angularModule.Regions2d(axes=newForm.axes)
            regions2d.append(newForm)
            regions2d.append(evalData.domainSlice(newForm.domainMax))
            print('Angular regions for distributions split at',newForm.domainMax)
            newRecon = reconDist.__class__(reconDist.label, reconDist.productFrame, regions2d)
            firstProduct.distribution.replace(newRecon)

#             firstProduct.distribution.add( newForm )

            if elastic and charged:   #    dCrossSection_dOmega for charged-particle elastics:
   
                NCPI = nuclearPlusInterferenceModule.NuclearPlusInterference( muCutoff=muCutoff,
                        crossSection=nuclearPlusInterferenceModule.CrossSection( effXsc),
                        distribution=nuclearPlusInterferenceModule.Distribution( effDist)
                        )
#                Rutherford = RutherfordScatteringModule.RutherfordScattering()

                CoulombElastic = CoulombPlusNuclearElasticModule.Form( gnd.projectile, rStyle, nuclearPlusInterference = NCPI, identicalParticles=identicalParticles )
                reaction.doubleDifferentialCrossSection.add( CoulombElastic )
    
                reaction.crossSection.remove( rStyle )
                reaction.crossSection.add( crossSectionModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )
                firstProduct.distribution.remove( rStyle )
                firstProduct.distribution.add( referenceModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )
            pair += 1

            # secondProduct = reaction.outputChannel[1]
            # secondProduct.distribution[rStyle].angularSubform.link = firstProduct.distribution[rStyle]    ## Fails
            # give 'recoil' distribution!
    return 


if __name__=="__main__":
    import argparse
    from reconstructLegendre import reconstructLegendre
    from fudge import reactionSuite as reactionSuiteModule

    parser = argparse.ArgumentParser(description='Translate R-matrix Evaluations')
    parser.add_argument('inFile', type=str, help='The input file you want to pointwise expand.' )
    parser.add_argument("dE", type=float, default='0', nargs='?', help="Reconstruct angle-integrated cross sections using TensorFlow for given E step (in eV). 0 (default) to use resonance-based grid.")
    parser.add_argument("-s", "--stride", type=int, help="Stride for accessing non-uniform grid template")

    parser.add_argument("-A", "--Angles", metavar='Ang', type=float, nargs=2, help="Reconstruct also angle-dependent cross sections, given thmin, thinc (in deg)")
    parser.add_argument("-L", "--Legendre", metavar='Ang', type=int, help="Reconstruct also angle-dependent cross sections as Legendre expansion up to this Lmax")
    parser.add_argument("-t", "--thin", action="store_true", help="Thin distributions")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debugging output (more than verbose)")

    args = parser.parse_args()
    debug = args.debug
    verbose = args.verbose or debug

    gnd=reactionSuiteModule.ReactionSuite.readXML_file(args.inFile)
    gnd.convertUnits( {'eV':'MeV'} )
    base = args.inFile.replace('.xml','')+'_RC'
    if args.dE>0: base += '+'+str(args.dE)
    if args.stride is not None: base += '+s%s' % args.stride
#     recons = gnd.styles.findInstancesOfClassInChildren(stylesModule.CrossSectionReconstructed)
#     print('\nRecons 1:',recons)
        
    print("\nReconstruct pointwise cross sections using TensorFlow")
    thin = args.thin
    finalStyleName = 'recon'
    reconstructedStyle = stylesModule.CrossSectionReconstructed( finalStyleName,
            derivedFrom=gnd.styles.getEvaluatedStyle().label )
        
    reconstructLegendre(gnd,base,verbose,debug,args.dE,args.stride,args.Angles,args.Legendre,thin,reconstructedStyle)

    if args.Legendre: args.Angles = None

    suffix = ''
    if args.Angles: suffix = 'A'+str(args.Angles[0]).replace('.0','')+','+str(args.Angles[1]).replace('.0','')+suffix
    if args.Legendre: suffix = 'L'+str(args.Legendre)+suffix
#     if args.stride is not None: base += '+s%s' % args.stride
    outFile = base + suffix + '.xml'
    
    open( outFile, mode='w' ).writelines( line+'\n' for line in gnd.toXML_strList( ) )
    print('Written',outFile)
