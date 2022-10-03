#!/usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

from pqu import PQU as PQUModule
import fudge.resonances.resolved as resolvedResonanceModule
import masses
from getCoulomb import *
fmscal = 0.0478450
etacns = 0.1574855
amu    = 931.494013
freeFormat = True

##############################################  write_azure

def write_azure(gnd,outFile,verbose,debug,nozero):    # elastic channel not put in output file!
    file = open(outFile,'w')
    
    rrr = gnd.resonances.resolved
    RMatrix = rrr.evaluated
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    IFG = RMatrix.reducedWidthAmplitudes      #  This is meaning of widths WITHIN the input gnd evaluation
    BC  = RMatrix.boundaryCondition     
    if BC != resolvedResonanceModule.BoundaryCondition.Brune:
        print("WARNING: \n  AZURE files need Brune parametrization as input\n")
    
    proj,targ = gnd.projectile,gnd.target
    elasticChannel = '%s + %s' % (proj,targ)
    if verbose: print('Elastic is <%s>\n' % elasticChannel)
    PoPs = gnd.PoPs
    if debug: open( 'azure-write.PoPs' , mode='w' ).writelines( line+'\n' for line in PoPs.toXMLList( ) )

    eSepE = -2e6; lab2cm=0
    pars = {}
    nn = 0
    for pair in RMatrix.resonanceReactions:
        kp = pair.label
        if not pair.eliminated:
            nn += 1    # index for ir_index in output
            reaction = pair.reactionLink.link
            p,t = pair.ejectile,pair.residual
            projectile = PoPs[p];
            target     = PoPs[t];     
            pMass = projectile.getMass('amu');   tMass =     target.getMass('amu');   
            if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
            if hasattr(target, 'nucleus'):     target = target.nucleus

            pZ    = projectile.charge[0].value;  tZ    =     target.charge[0].value

            cZA = int((pZ+tZ)*1000 + pMass+tMass + 0.5)  # compound nucleus
            if pair.Q is not None:
                QI = pair.Q.getConstantAs('MeV')
            else:
                QI = reaction.getQ('MeV')

            if verbose: print(pMass, tMass, cZA,masses.getMassFromZA( cZA ))
            cnDefect = (pMass + tMass - masses.getMassFromZA( cZA ))*amu # MeV
            #print pMass, tMass, cZA,masses.getMassFromZA( cZA ),cnDefect

            if pair.label == elasticChannel: lab2cm = tMass / (pMass + tMass)

            if eSepE < -1e6:
                QM = QI
                eSepE = QM + cnDefect  # elastic separation energy (eSepE) set chosen from FIRST p+t capture  (baseline to t in gs by taking QM=QI there)
                print(" Setting CN zero energy from eSepE =",eSepE," as   QM=",QM," + cnDefect=",cnDefect)
            QM = eSepE - cnDefect      # other separation energies use first eSepE
            sepE = eSepE -  QI
            et = QM - QI
            if verbose: print(" Channel ",p,t," has cnDefect=%.5f QI=%.5f, QM=%.5f, et=%.5f" % (cnDefect,QI,QM,et))
    
            jp,pt = projectile.spin[0].value, projectile.parity[0].value
            jt,tt = target.spin[0].value,     target.parity[0].value

            prmax = 1.25 * tMass**(1.0/3.0)   # some kind of default
            if pair.scatteringRadius is not None:
                prmax =  pair.scatteringRadius.getValueAs('fm')   
            else:
                prmax = Rm_global

            pars[kp] = (nn,pZ,tZ,jp,jt,pt,tt,et,sepE,QM,QI,prmax,pMass,tMass)
#
# We need a table of all masses, charges, etc because we need dS/dE for all channels before we can output ANY 'observed' width, and dS/dE depends on them.
    #if debug: print "Pars:", pars #p for p in pars
    if lab2cm <1e-5:
        print("Elastic channel not found")
        raise SystemExit

    config = ['true', 'output/', 'checks/', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']  # default output
    file.write('\n'.join(['<config>']+config+['</config>'])+'\n')
    file.write('<levels>\n')
    term = 0
    vars = 0
    ito = -1
    # if debug: print 'pars:',pars
    for Jpi in RMatrix.spinGroups:
        jtot = Jpi.spin
        parity = Jpi.parity # -1 or +1
        ito += 1
        R = Jpi.resonanceParameters.table
        poleEnergies = R.getColumn('energy','MeV') # Azure wants pole energies in MeV
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
        for i in range(0,len(R)):
            term += 1
            e = poleEnergies[i] * lab2cm
            if term>1: file.write('\n')

            shifty_denom = 1.0
            for ch in Jpi.channels:
                m = ch.columnIndex
                kp = str(ch.resonanceReaction)
                lch = ch.L
                width = widths[m-1][i]
                width *= lab2cm**0.5 if IFG else lab2cm
                if not RMatrix.resonanceReactions[kp].eliminated:
                    nn,pZ,tZ,jp,jt,pt,tt,et,sepE,QM,QI,prmax,pMass,tMass = pars[kp]
                    e_ch = e + QI
                    penetrability,shift,dSdE,W = getCoulomb_PSdSW(e_ch,lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, True)
                    if IFG: 
                        rwa_sq = width**2    # GND file was rwa already, so just square it.
                    else:
                        Gammaf = width
                        rwa_sq = abs(Gammaf) /(2. * penetrability) 
                    s = rwa_sq * dSdE
                    shifty_denom += s
                    if debug: print("i,lch,e_ch,QI:",m,lch,e_ch,QI," PSdS =",penetrability,shift,dSdE,W,', SD contrib:',s)
            if  debug: print("Pole",i,"at",e,", SD:",shifty_denom,' so SS:',2*(1.0-1.0/shifty_denom))

            for ch in Jpi.channels:
                n = ch.columnIndex
                kp = str(ch.resonanceReaction)
                width = widths[n-1][i]
                width *= lab2cm**0.5 if IFG else lab2cm
                lch = ch.L
                sch = ch.channelSpin
                include =  not (nozero and abs(width)<1e-20)

                if include and not RMatrix.resonanceReactions[kp].eliminated:
                    nn,pZ,tZ,jp,jt,pt,tt,et,sepE,QM,QI,prmax,pMass,tMass = pars[kp]

                    e_ch = e + QI
                    penetrability,shift,dSdE,W = getCoulomb_PSdSW(e_ch,lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, False)
                    #if debug: print 'pp ',kp,' P(',lch,'e=',e_ch,') = ',penetrability

#   write OBSERVED WIDTHS to Azure: Gamma_obs = 2*gamma**2 * P/(some shifty expression)
                    if e_ch>0. :
                        if IFG:   # stored width = rwa
                            rwa = width
                            width = 2 * rwa**2 * penetrability
                            if rwa<0: width = -width
                        #else:     # stored width = 2 * rwa**2 * P  already
                        width = width / shifty_denom
                        width *= 1e6  # Give widths in azure files in eV.
    #
                    else:   # bound states
                        if debug: print('pp ',kp,' P(',lch,'e=',e_ch,') = ',penetrability,dSdE,W)
                        if IFG:   # stored width = rwa
                            rwa = width   
                        else:     # stored width = 2 * rwa**2 * P                          
                            rwa = (abs(width) /(2. * penetrability))**0.5
                            if width<0.: rwa = -rwa
                        redmass = pMass*tMass/(pMass+tMass)
                        anc = rwa * (fmscal*redmass*prmax/shifty_denom)**0.5 / float(W)
                        width = anc   # Now ANC
    #
# write:   lJ,lPi,E,lFix,aa,ir,s,l,ID,Active,cFix,gamma,j1,pi1,j2,pi2,e2,m1,m2,z1,z2,eSepE,sepE,j3,pi3,e3,pType,chRad,g1,g2,ecMult
                    sepE = eSepE - QM
                    E = e + eSepE              # R-matrix pole energy converted to continuum energy in elastic channel

                    #if debug: print jtot,parity,E,nn,int(2*sch),2*lch,term, width,jp,pt,jt,tt,et,pMass,tMass,pZ,tZ,eSepE,sepE,prmax
                    if freeFormat:
                        string = "%5.1f %4i %s   0   1 %4i %4i %4i %4i   1   0 %s %5.1f %4i %5.1f %4i %s %s %s %4i %5i %s %s 0 0  0.0  0 %s 0 0 0\n" % (jtot,parity,E,nn,int(2*sch),2*lch,term, width,jp,pt,jt,tt,et,pMass,tMass,pZ,tZ,eSepE,sepE,prmax)
                    else:
                        if abs(width)>10.0 or abs(width)<1e-20: 
                            string = "%5.1f %4i %12.5f   0   1 %4i %4i %4i %4i   1   0 %20.2f %5.1f %4i %5.1f %4i %10.5f %9.6f %10.6f %4i %5i %12.5f %12.5f 0 0  0.0  0 %10.6f 0 0 0\n" % (jtot,parity,E,nn,int(2*sch),2*lch,term, width,jp,pt,jt,tt,et,pMass,tMass,pZ,tZ,eSepE,sepE,prmax)
                        else:
                            string = "%5.1f %4i %12.5f   0   1 %4i %4i %4i %4i   1   0 %20.5f %5.1f %4i %5.1f %4i %10.5f %9.6f %10.6f %4i %5i %12.5f %12.5f 0 0  0.0  0 %10.6f 0 0 0\n" % (jtot,parity,E,nn,int(2*sch),2*lch,term, width,jp,pt,jt,tt,et,pMass,tMass,pZ,tZ,eSepE,sepE,prmax)
                    file.write(string)
                    vars += 1
                    #if debug: print "Pole at ",e," @ ",n,'   width',width, ' from SD =',shifty_denom
    file.write('</levels>\n')

    print("\nAzure2 file '%s' written with  %i level lines" % (outFile,vars))
