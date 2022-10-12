#! /usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################
TF = True

import os,math,numpy,cmath
import sys
from CoulCF import cf1,cf2,csigma
from pqu import PQU as PQUModule
# from PoPs.groups.misc import *

import fudge.sums as sumsModule
import fudge.styles as stylesModule
import fudge.reactionData.crossSection as crossSectionModule
import fudge.productData.distributions as distributionsModule
import fudge.resonances.resolved as resolvedResonanceModule

DBLE = numpy.double
CMPLX = numpy.complex128
INT = numpy.int32

if TF: 
    # import tensorflow as tf
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    
# TO do

#   Verify cp elastic distributions: cf Fresco, sig(th) reconstructions for data
#   Consider storing cp Legendre coeffficients.
#
#   Brune basis: level matrix calculation


##############################################  reconstructTensorFlow

hbc =   197.3269788e0             # hcross * c (MeV.fm)
finec = 137.035999139e0           # 1/alpha (fine-structure constant)
amu   = 931.4940954e0             # 1 amu/c^2 in MeV

coulcn = hbc/finec                # e^2
fmscal = 2e0 * amu / hbc**2
etacns = coulcn * math.sqrt(fmscal) * 0.5e0
pi = 3.1415926536
rsqr4pi = 1.0/(4*pi)**0.5


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
    
    return(RMATC,T_mat,TC_mat)
        
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
                                        
def reconstructTensorFlow(gnd,base,verbose,debug,egrid,angles,thin,reconstyle):

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
    
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')
    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes
    

    if angles is not None:
        thmin = angles[0]
        thinc = angles[1]
        if charged:
            from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import CoulombPlusNuclearElastic as  CoulombPlusNuclearElasticModule
            from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import nuclearPlusInterference as nuclearPlusInterferenceModule
    #         from fudge.reactionData.doubleDifferentialCrossSection.chargedParticleElastic import RutherfordScattering as RutherfordScatteringModule
            from fudge.productData.distributions import reference as referenceModule
    # 
            muCutoff = math.cos(thmin*pi/180.)
    accuracy = None

 
    de = egrid
    emin = max(emin,de)
    n_energies = int( (emax-emin)/de + 2.5)
    de = (emax-emin)/(n_energies-1)
    print('Reconstruction emin,emax =',emin,emax,'with',n_energies,'spaced at',de,'MeV')
    E_scat  = numpy.asarray([emin + ie*de for ie in range(n_energies)], dtype=DBLE)
  #  print('Energy grid (lab):',E_scat)
    n_jsets = len(RMatrix.spinGroups)
    n_poles = 0
    n_angles = 0     # angles
    n_chans = 0
    
    np = len(RMatrix.resonanceReactions)
    ReichMoore = False
    if RMatrix.resonanceReactions[0].eliminated: 
        print('Exclude Reich-Moore channel')
        ReichMoore = True
        np -= 1   # exclude Reich-Moore channel here
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
        if partition.label == elasticChannel:
            lab2cm = tMass / (pMass + tMass)
            w_factor = 1. #/lab2cm**0.5 if IFG else 1.0
            ipair = pair  # incoming
            
        jp[pair],pt[pair],ep[pair] = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt[pair],tt[pair],et[pair] = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].pqu('MeV').value
        except:
            jt[pair],tt[pair],et[pair] = 0.,1,0.
        print(pair,":",kp,rmass[pair],QI[pair],prmax[pair])
        pair += 1
    # print("\nElastic channel is",elasticChannel,'so w factor=',w_factor,'as IFG=',IFG)
    npairs  = pair
    if not IFG:
        print("Not yet coded for IFG =",IFG)
        sys.exit()
    
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
    print('Need %i energies in %i Jpi sets with %i poles max, and %i channels max. Lmax=%i' % (n_energies,n_jsets,n_poles,n_chans,Lmax))

    E_poles = numpy.zeros([n_jsets,n_poles], dtype=DBLE)
    g_poles = numpy.zeros([n_jsets,n_poles,n_chans], dtype=DBLE)
    J_set = numpy.zeros(n_jsets, dtype=DBLE)
    pi_set = numpy.zeros(n_jsets, dtype=INT)
    L_val  =  numpy.zeros([n_jsets,n_chans], dtype=INT)
    S_val  =  numpy.zeros([n_jsets,n_chans], dtype=DBLE)
    p_mask =  numpy.zeros([npairs,n_jsets,n_chans], dtype=DBLE)
    seg_val=  numpy.zeros([n_jsets,n_chans], dtype=INT) - 1 
    seg_col=  numpy.zeros([n_jsets], dtype=INT) 

    rksq_val  = numpy.zeros([n_energies,npairs], dtype=DBLE)
    velocity  = numpy.zeros([n_energies,npairs], dtype=DBLE)
    
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
    Spins = [set() for pair in range(npairs)]
    
#  Calculate Coulomb functions    
    for pair in range(npairs):
#         print('Partition',pair,'Q,mu:',QI[pair],rmass[pair])
        if debug:
            foutS = open(base + '+3-S%i' % pair,'w')
            foutP = open(base + '+3-P%i' % pair,'w')
        for ie in range(n_energies):
            E = E_scat[ie]*lab2cm + QI[pair]
            k = cmath.sqrt(fmscal * rmass[pair] * E)
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
            velocity[ie,pair] = k.real/rmass[pair]  # ignoring factor of hbar
            
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
                L = 3 # for printing
                EE = E  # or E_scat[ie]
                print(EE,CF2_val[ie,pair,L].real, file=foutS)
                print(EE,CF2_val[ie,pair,L].imag, file=foutP)
        if debug:
            foutS.close()
            foutP.close()
        
    #  SECOND: fill in arrays:
    jset = 0
    for Jpi in RMatrix.spinGroups:
        J_set[jset] = Jpi.spin
        pi_set[jset] = Jpi.parity
        # print('J,pi =',J_set[jset],pi_set[jset])
        R = Jpi.resonanceParameters.table
        rows = R.nRows
        cols = R.nColumns - 1  # ignore energy col
        seg_col[jset] = cols

        E_poles[jset,:rows] = numpy.asarray( R.getColumn('energy','MeV') , dtype=DBLE)   # lab MeV
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
        
        n = 0
        All_spins = set()
        for ch in Jpi.channels:
            rr = ch.resonanceReaction
            pair = partitions.get(rr,None)
            if pair is None: continue
            m = ch.columnIndex - 1
            g_poles[jset,:rows,n] = numpy.asarray(widths[m][:],  dtype=DBLE) * w_factor
            L_val[jset,n] = ch.L
            S = float(ch.channelSpin)
            S_val[jset,n] = S
            
            seg_val[jset,n] = pair
            p_mask[pair,jset,n] = 1.0
            Spins[pair].add(S)
            All_spins.add(S)

        # Find S and P:
            for ie in range(n_energies):

   
#               print('ie,jset,n: BC=' , ie,jset,n,BC )
                if BC == resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
                    B = -ch.L
                elif BC == resolvedResonanceModule.BoundaryCondition.Given:              # btype='B'
                    B = BV
                elif BC == resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction:              # btype='B'
                    B = None
                if ch.boundaryConditionValue is not None:
                    B = float(ch.boundaryConditionValue)
         
#               print('ie,jset,n: BC,B=' , ie,jset,n,BC,B )

                DL = CF2_val[ie,pair,ch.L]
                S = DL.real
                P = DL.imag
                F = CF1_val[ie,pair,ch.L]
                Psr = math.sqrt(abs(P))
                phi = - math.atan2(P, F - S)
                Omega = cmath.exp(complex(0,phi))
                if B is None:
                    L_diag[ie,jset,n]       = complex(0.,P)
                else:
                    L_diag[ie,jset,n]       = DL - B

                POm_diag[ie,jset,n]      = Psr * Omega
                Om2_mat[ie,jset,n,n]     = Omega**2
                CS_diag[ie,jset,n]       = Csig_exp[ie,pair,ch.L]
            n += 1
        if debug:
            print('J set %i: E_poles \n' % jset,E_poles[jset,:])
            print('g_poles \n',g_poles[jset,:,:])
            print('g_poles \n',g_poles[jset,:,:])
        jset += 1        

#    print('All spins:',All_spins)
#    print('All channel spins',Spins)

    if not TF: return

    E_cpoles = tf.complex(E_poles,tf.constant(0., dtype=DBLE)) 
    g_cpoles = tf.complex(g_poles,tf.constant(0., dtype=DBLE))
    E_cscat = tf.complex(E_scat,tf.constant(0., dtype=DBLE)) 
    
    RMATC,T_mat,TC_mat = R2T_transformsTF(g_cpoles,E_cpoles,E_cscat,L_diag, Om2_mat,POm_diag,CS_diag, n_jsets,n_poles,n_chans ) 

    if debug:
        for ie in range(n_energies):
            for jset in range(n_jsets):
                print('Energy',E_scat[ie],'  J=',J_set[jset],pi_set[jset],'\n R-matrix is size',seg_col[jset])
                for a in range(n_chans):
                    print('   ',a,'row: ',',  '.join(['{:.5f}'.format(RMATC[ie,jset,a,b].numpy()) for b in range(n_chans)]) )
    
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

    print('angles:',angles)
    if angles is not None:

        from numericalFunctions import angularMomentumCoupling
        from xData.series1d  import Legendre

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

        print('# angles=',na,' to L=',NL)
    
        NS = len(All_spins)
        ZZbar = numpy.zeros([NL,NS,n_jsets,n_chans,n_jsets,n_chans])

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

                            for L in range(NL):                    
                                ZZbar[L,iS,jset2,c2,jset1,c1] = angularMomentumCoupling.zbar_coefficient(i2(L1),n2(J1),i2(L2),n2(J2),n2(S),i2(L))

    #     trace = open('trace','w')
    
        # calculate angular distributiones here. Later move to TF kernel.
        sigdd = {}    
        pair = 0
        rr = None
        T_mat_n = T_mat.numpy()
        BB = numpy.zeros([n_energies,NL])
        AA = numpy.zeros([np, n_jsets,n_chans,n_chans, NL, n_jsets,n_chans,n_chans  ], dtype=CMPLX)
        sig_ni = numpy.zeros(n_energies)
        pair = 0
        for rr_out in RMatrix.resonanceReactions:
            if not rr_out.eliminated:
                elastic = rr_out.label == elasticChannel
                distFile = open('fort.%4i' % (1000+pair),'w')
                xsFile = open('fort.%4i' % (2001+pair),'w')
                AA[pair,:,:,:, :, :,:,:] = 0.0
                sigdd[rr_out.label] = []
                for S_out in Spins[pair]:
                    for S_in in Spins[ipair]:
    #                     print('>> S_in:',S_in)
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

    #                                     print('Plain: J,c =',J1,c1)

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

                                                        for L in range(NL):
                                                            ZZ = ZZbar[L,iS,jset2,c2,jset1,c1] * ZZbar[L,iSo,jset2,c2_out,jset1,c1_out] 
                                                            AA[pair, jset2,c2_out,c2, L, jset1,c1_out,c1] += phase * ZZ 
                
                BB_t = T2B_transformsTF(TC_mat,AA[pair, :,:,:, :, :,:,:], n_jsets,n_chans)
            
                A_t = B2A_transformsTF(BB_t, Pleg)

                BB = BB_t.numpy()
                Angular= A_t.numpy()
                TC_mat_n= TC_mat.numpy()
            
                                        
                xsFile2 =   open('A0-xs.%i' % pair,'w')   
                for ie in range(n_energies):
                    E = E_scat[ie]   # lab incident energy
                    print('# Elab =%16.8e %4i' % (E,pair+1), file=distFile)
                    print('# Csig =', csigma_v[ie,pair,:6]-csigma_v[ie,pair,0] , file=distFile)
                    denom = (2.*jp[ipair]+1.) * (2.*jt[ipair]+1)
                    gfacc =  pi * rksq_val[ie,ipair] / denom  * 10.
                    xs = BB[ie,0]/pi*gfacc * 4*pi
                
#                   print('For',pair,'<-',ipair,'BB[:] at E=%7.4f' % E,':',BB[ie,0:6]/(1e-20+BB[ie,0]),'    sig =',xs,'mb')

                    dist = []
                    scale = 0.5 / BB[ie,0] if BB[ie,0] !=0 else 1.0
                    if pair==ipair and charged: sig_ni[ie] = 0.0
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
                    
                            print('%10.5f %13.5e' % (mu,ds), file=distFile)
                            dist.insert(0,ds)
                            dist.insert(0,mu)
                            theta = thmin + ia*thinc
                            print('%10.5f %13.5e' % (theta,ds*xs), file=xsFile)
                        mulast = mu
                    if pair==ipair and charged:  # find normalized difference
                        print(E,sig_ni[ie], file=xsFile2)
                        for ia in range(na):
                            mu = mu_vals[ia]
                            ds = 2*pi*( xsc[ia] - Rutherford[ia] ) / sig_ni[ie]
                            print('%10.5f %13.5e' % (mu,ds), file=distFile)
                            dist.insert(0,ds)
                            dist.insert(0,mu)
                            theta = thmin + ia*thinc
                            thrad = theta*pi/180.
                            mu = math.cos(thrad)
                            print('%10.5f %13.5e %13.5e ' % (theta,xsc[ia],Rutherford[ia]), file=xsFile)
                    else:
                        print(E,xs, file=xsFile2)
                    print('&', file=distFile)
                    print('&', file=xsFile)
                    sigdd[rr_out.label].append([E,dist])
                                
                pair += 1
    
    ## # PROCESS CROSS-SECTIONS
    
                    
    egrid = E_scat[:]    # lab MeV
    totalxs = XSp_tot_n[:,ipair] * 0.01   # barns
    if charged and angles is not None:
        elasticxs = sig_ni[:] * 1e-3 # barns not mb
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
    elastic = crossSectionModule.XYs1d( axes = crossSectionAxes, data=(egrid, elasticxs), dataForm="XsAndYs" )
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
            RRx = RRxsec.thin( accuracy or .001 )
        else:
            RRx = RRxsec
        RRx.label = rStyle

        reaction.crossSection.add( RRx )
       
        # print("Channels ",reaction.label,iselastic,":\n",RRxsec.toString(),"\n&\n",RRx.toString())
        if iselastic:
            effXsc = RRxsec
            
    gnd.styles.add( reconstyle )

    if angles is None: return
    
        ## # PROCESS DISTRiBUTIONS
        
    angularAxes = distributionsModule.angular.defaultAxes( 'MeV' )  # for Elab outerDomainValue
    for rreac in gnd.resonances.resolved.evaluated.resonanceReactions:
        if not rreac.eliminated:
            productName = rreac.ejectile
            residName   = rreac.residual
            elastic = productName == gnd.projectile and residName == gnd.target
            print("Add angular distribution for",productName," in",rreac.label,"channel (elastic=",elastic,")")

            reaction = rreac.reactionLink.link
            firstProduct = reaction.outputChannel.getProductWithName(productName)

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

            newForm = distributionsModule.angular.twoBodyForm( label = reconstyle.label,
                productFrame = firstProduct.distribution.evaluated.productFrame, angularSubform = effDist )
            firstProduct.distribution.add( newForm )

            if elastic and charged:   #    dCrossSection_dOmega for charged-particle elastics:
   
                NCPI = nuclearPlusInterferenceModule.nuclearPlusInterference( muCutoff=muCutoff,
                        crossSection=nuclearPlusInterferenceModule.crossSection( effXsc),
                        distribution=nuclearPlusInterferenceModule.distribution( effDist)
                        )
#                Rutherford = RutherfordScatteringModule.RutherfordScattering()

                CoulombElastic = CoulombPlusNuclearElasticModule.form( gnd.projectile, rStyle, nuclearPlusInterference = NCPI, identicalParticles=identicalParticles )
                reaction.doubleDifferentialCrossSection.add( CoulombElastic )
    
                reaction.crossSection.remove( rStyle )
                reaction.crossSection.add( crossSectionModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )
                firstProduct.distribution.remove( rStyle )
                firstProduct.distribution.add( referenceModule.CoulombPlusNuclearElastic( link = reaction.doubleDifferentialCrossSection[rStyle],
                    label = rStyle, relative = True ) )

            secondProduct = reaction.outputChannel[1]
            # secondProduct.distribution[rStyle].angularSubform.link = firstProduct.distribution[rStyle]    ## Fails
            # give 'recoil' distribution!
    return 


if __name__=="__main__":
    import argparse
    from reconstructTensorFlow import reconstructTensorFlow
    from fudge import reactionSuite as reactionSuiteModule

    parser = argparse.ArgumentParser(description='Translate R-matrix Evaluations')
    parser.add_argument('inFile', type=str, help='The input file you want to pointwise expand.' )
    parser.add_argument("dE", type=float, default='0', help="Reconstruct angle-integrated cross sections using TensorFlow for given E step (in eV)")

    parser.add_argument("-A", "--Angles", metavar='Ang', type=float, nargs=2, help="Reconstruct also angle-dependent cross sections, given thmin, thinc (in deg)")
    parser.add_argument("-t", "--thin", action="store_true", help="Thin distributions")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debugging output (more than verbose)")

    args = parser.parse_args()
    debug = args.debug
    verbose = args.verbose or debug

    gnd=reactionSuiteModule.readXML(args.inFile)
    base = args.inFile.replace('.xml','')

    print("Reconstruct pointwise cross sections using TensorFlow")
    thin = args.thin
    finalStyleName = 'recon'
    reconstructedStyle = stylesModule.crossSectionReconstructed( finalStyleName,
            derivedFrom=gnd.styles.getEvaluatedStyle().label )
    reconstructTensorFlow(gnd,base,verbose,debug,args.dE,args.Angles,thin,reconstructedStyle)


    suffix = '+'+str(args.dE).replace('.0','')+'eV'
    if args.Angles: suffix = 'A'+str(args.Angles[0]).replace('.0','')+','+str(args.Angles[1]).replace('.0','')+suffix
    outFile = base + suffix + '.xml'
    
    open( outFile, mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
    print('Written',outFile)
