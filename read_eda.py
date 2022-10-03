#

##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,eda,hyrma         #
#                                            #
##############################################

from eda_parfile import getEDA

from fudge import reactionSuite as reactionSuiteModule
from fudge import styles as stylesModule
from fudge import physicalQuantity as physicalQuantityModule
from fudge.reactions import reaction as reactionModule
from fudge.reactionData import crossSection as crossSectionModule
from fudge.processing.resonances.getCoulombWavefunctions import *
from xData.Documentation import computerCode as computerCodeModule

import masses
import fudge.resonances.resonances as resonancesModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
import fudge.resonances.resolved as resolvedResonanceModule
import fudge.resonances.common as commonResonanceModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
import fudge.covariances.modelParameters as covarianceModelParametersModule
import fudge.covariances.covarianceSection as covarianceSectionModule
import fudge.covariances.covarianceSuite as covarianceSuiteModule

from PoPs import database as databaseModule
from PoPs import misc as miscModule
from PoPs.families import gaugeBoson as gaugeBosonModule
from PoPs.families import baryon as baryonModule
from PoPs.families import nucleus as nucleusModule
from PoPs.families import nuclide as nuclideModule
from PoPs.quantities import spin as spinModule
from PoPs.groups.misc import *

from pqu import PQU as PQUModule
from xData import table as tableModule
import xData.constant as constantModule
import xData.link as linkModule
import xData.xDataArray as arrayModule

from zeroReaction import *
from getCoulomb import *
import os,pwd,time
import fractions
import numpy
fmscal = 0.0478450
etacns = 0.1574855
amu    = 931.494013
oneHalf = fractions.Fraction( '1/2' )
one = fractions.Fraction( '1' )
zero = fractions.Fraction( '0' )
spinUnit = spinModule.baseUnit

##############################################  read_eda

def read_eda(inFile,covFile,elastic, amplitudes,noCov, emin,emax, verbose,debug):
    
    lines = open(inFile).readlines()
    covlines = None
    if covFile is not None and not noCov: covlines = open(covFile).readlines()
    comment, partitions, boundaries, rmatr, channelList, normalizations, nuclei,covariances  = getEDA(lines,covlines,debug and False)
    if elastic is None:
        elastic = partitions[0][0]
        print("### Elastic channel defaulted to",elastic)
    print('# sets =',len(rmatr))
    domain = stylesModule.projectileEnergyDomain(emin,emax,'MeV')
    style = stylesModule.evaluated( 'eval', '', physicalQuantityModule.temperature( 300, 'K' ), domain, 'from '+inFile , '0.1.0' )
    PoPs_data = databaseModule.database( 'eda', '1.0.0' )
    resonanceReactions = commonResonanceModule.resonanceReactions()
    computePenetrability = True
    MTchannels = []

    approximation = 'Full R-Matrix'
    
    KRL = False  # not relativistic by default (so far)
    LRP = 2  # do not reconstruct resonances pointwise (for now)
    
    BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum  # default
    eunit = 'MeV'
    is_rwa = True      # default in EDA inputs

    ZAdict = {}
    rrList = []
    cm2lab = 0
    Rm_global = None
    for part in partitions:
        np = part[0]
        nt = part[1]
        Lmax = part[2]
        p_q = nuclei[np]
        t_q = nuclei[nt]
        if debug: print('proj',p_q)
        if debug: print('targ',t_q)
        ia,ep,et = 0, 0.0, 0.0  # Feature of EDA: these options are not used
        
        pMass,pZ = p_q[3],int(p_q[2]+0.5)
        tMass,tZ = t_q[3],int(t_q[2]+0.5)
        #  Use standard GND names:
        pA = int(pMass+0.5)
        tA = int(tMass+0.5)

        jp,ptyp = p_q[1],p_q[0]
        jt,ptyt = t_q[1],t_q[0]
        pp = ptyp
        pt = ptyt
        Jp = int(jp) if abs(jp - int(jp))<0.1 else '%i/2' % int(2*jp)
        Jt = int(jt) if abs(jt - int(jt))<0.1 else '%i/2' % int(2*jt)
        
        if pA==0: 
            p = 'photon' if int(jp)==0 else 'photonM'
        else:
            p = idFromZAndA(pZ,pA)
        t = idFromZAndA(tZ,tA)
        tex = nuclideIDFromIsotopeSymbolAndIndex(t,ia)
        rr = '%s + %s' % (p,tex)  # label for resonanceReaction
        #print "\nprojectile: p,pZ,pA,pMass=",p,pZ,pA,pMass
        rrList.append(rr)
    
        prmax = part[4]
        Qvalue = part[5] # -(pMass + tMass)*amu
        QI = Qvalue - ep - et
        if Rm_global is None: Rm_global = prmax
        
        channelName = '%s + %s' % (p,tex)
        #ZAdict[rr] = (float(pMass),float(tMass),float(pZ),float(tZ),float(QI),float(prmax),Lmax)
        ZAdict[rr] = (pMass,tMass,pZ,tZ,QI,prmax)
        if debug: print('ZAdist[',rr,'] =',ZAdict[rr])
        #JTmax = max(JTmax,Lmax+Jp+Jt)

        MT = 5
        if p[:6]=='photon':            MT = 102
        elif p=='n' :              MT = 50+ia
        elif p=='H1' :             MT = 600+ia
        elif p=='H2' :             MT = 650+ia
        elif p=='H3' :             MT = 700+ia
        elif p=='He3' :            MT = 750+ia
        elif p=='He4' :            MT = 800+ia

        if debug: print(' projectile=',p,pMass,pZ,Jp,ptyp, ', target=',tex,tMass,tZ,Jt,ptyt, ' Q=',Qvalue,' MT=',MT,' prmax =',prmax)
        if pZ==0 and pMass == 0 :   # g
            projectile = miscModule.buildParticleFromRawData( gaugeBosonModule.particle, p, mass = ( 0, 'amu' ), spin = (Jp,spinUnit ),  parity = (ptyp,'' ), charge = (0,'e') )
        elif pZ<1 and pMass > 0.5 and pMass < 1.5 and p != 'H1' :  # n or p
            projectile = miscModule.buildParticleFromRawData( baryonModule.particle, p, mass = (pMass,'amu' ), spin = (Jp,spinUnit ),  parity = (ptyp,'' ), charge = (pZ,'e') )
        else: # nucleus in its gs
            nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, p, index = 0, energy = ( 0.0, 'MeV' ) , spin=(Jp,spinUnit), parity=(ptyp,''), charge=(pZ,'e'))
            projectile = miscModule.buildParticleFromRawData( nuclideModule.particle, p, nucleus = nucleus,  mass=(pMass,'amu'))
        PoPs_data.add( projectile )

        # Some state of target at energy 'et':
        if debug: print("Build PoPs for target ",tex,Jt,ptyt,tZ,tMass,ia,et)
        nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, tex, index = ia, energy = (et,'MeV' ) , spin=(Jt,spinUnit), parity=(ptyt,''), charge=(tZ,'e') )
        target = miscModule.buildParticleFromRawData( nuclideModule.particle, tex, nucleus = nucleus, mass=(tMass,'amu'))
        PoPs_data.add( target )


        if np == elastic:                                        # that was the incident channel of',p,tex
            elastics = (p,tex)
            MT = 2
            cm2lab = (tMass + pMass)/tMass

        # Create zero background cross section
        MTchannels.append((rr,zeroReaction(rr,MT, QI, [projectile,target], None, emin,emax,eunit, debug), channelName,prmax,p))
        compoundA = pA + tA
        compoundZ = pZ + tZ
            
    damped = False

#  After making all the channels, and gnd is generated for the elastic channel, now add them to gnd
    p,tex = elastics   
    gnd = reactionSuiteModule.reactionSuite( p, tex, 'EDA R-matrix fit', PoPs =  PoPs_data, style = style, interaction='nuclear')

    for rr,reaction,channelName,prmax,p in MTchannels:
#  Get zero background cross section and link to it
        #reaction,channelName,prmax = MTchannels[rr]
        gnd.reactions.add(reaction)
        eliminated = False
        reactionLink = linkModule.link(reaction)
        rreac = commonResonanceModule.resonanceReaction ( label=rr, reactionLink=reactionLink, ejectile=p, computePenetrability=True,
                                                     computeShiftFactor=True, Q=None, eliminated=eliminated  )
        if prmax is not None and prmax != Rm_global:
            rreac.scatteringRadius = scatteringRadiusModule.scatteringRadius(      
                constantModule.constant1d(prmax, domainMin=emin, domainMax=emax,
                    axes=axesModule.axes(labelsUnits={1: ('energy_in', eunit), 0: ('radius', 'fm')})) )
        resonanceReactions.add(rreac)
        if debug: print("RR <"+rr+"> is "+channelName)

    if cm2lab<1e-5:
        print("Missed elastic channel for cm2lab factor!")
        raise SystemExit

#  Now read and collate the reduced channel partial waves and their reduced width amplitudes
# next we have NJS spin groups, each containing channels and resonances

    NJS = len(channelList)
    npars = 0
    # partialWave = index of LANL channel order
    spinGroups = resolvedResonanceModule.spinGroups()
    for spinGroupIndex in range(NJS):
        chans = channelList[ spinGroupIndex ]
        J,pi = chans[0]
        JJ = resolvedResonanceModule.spin( J )
        pi= resolvedResonanceModule.spin( pi)
        if verbose  or debug: print('\n ##### Spinset #',spinGroupIndex,': J,pi =',JJ,pi,'\n',chans)
    

        columnHeaders = [ tableModule.columnHeader(0, name="energy", unit="MeV") ]
        width_units = 'MeV'   ##   'MeV**{1/2}' if amplitudes else 'MeV'  # wrong units given to GND: correct later if needed
        channelNames = []
        channels = resolvedResonanceModule.channels()
        NCH = len(chans)-1
        for chidx in range(NCH):
            part,p,t,lch,sch,BC,partialWave = chans[chidx+1]
            rr = rrList[part] 
            #if debug: print "From p,t =",p,t," find channel ",rr
            thisChannel = resonanceReactions[rr]
            channelName = "%s width" % thisChannel.label

            jdx = 2
            while True:
                if channelName not in channelNames:
                    channelNames.append( channelName ); break
                channelName = '%s width_%d' % (thisChannel.label, jdx)
                jdx += 1

            columnHeaders.append( tableModule.columnHeader(chidx+1, name=channelName, unit= width_units) )

            Sch = resolvedResonanceModule.spin( sch )
            channels.add( resolvedResonanceModule.channel(str(chidx+1), rr, columnIndex=chidx+1, 
                    L=lch, channelSpin=Sch, boundaryConditionValue = BC ))
                            
            if debug: print(str(chidx), str(chidx),'LS:', int(lch), float(sch), chidx+1, 'B=',BC)
        
        resonances = []
        rmatset = rmatr[spinGroupIndex]
        energies = rmatset[2][0]
        nle = len(energies)
        if debug: print(" Energies: ",energies)
        if debug: print(" rmatset: ",rmatset)

        for level in range(nle):
            energy = energies[level]
            row = [energy*cm2lab]
            npars += 1 + NCH  # energy and widths
            for ich in range(NCH):
                #print " W for",ich+1," for all levels:",rmatset[ich+3][0]
                part,p,t,lch,sch,BC,partialWave = chans[ich+1]
                rr = rrList[part] 
                w = rmatset[ich+3][0][level]
                if debug: print("level",level+1," ch",ich+1," rwa=",w)
                if is_rwa != amplitudes:   # fix to give correct output: rwa or formal width
                    pMass,tMass,pZ,tZ,QI,prmax = ZAdict[ rr ]
                    e_ch = energy + QI
                    penetrability,shift,dSdE,W = getCoulomb_PSdSW(
                              e_ch,lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, False)   # CWF at abs(e_ch)
                    #if debug: print 'p,t =',p,tex,': call coulombPenetrationFactor(L=',lch,'rho=',rho,'eta=',eta,') =',penetrability,dSdE,W
                    #   find gamma or Gamma_formal from the G_obs in the AZR input
                    #    Gamma_formal = G_obs * shifty_denom
                    #    gamma =  sqrt(Gamma_formal/(2*P))
                    if amplitudes:         # GND to have rwa from Gammaf
                        width = ( abs(w) /(2. * penetrability) ) **0.5
                        if w < 0: width = -width
                        if debug: print("              Converting Gammaf",w," to rwa",width)
                    else:           # GND to have Gammaf from rwa
                        width = 2.0 * w*w * penetrability
                        if w < 0: width = -width
                        if debug: print("              Converting rwa",w," to Gammaf",width)
                else:
                    width = w
                width *= cm2lab**0.5 if amplitudes else cm2lab
                row.append(width)
            resonances.append(row)

        table = tableModule.table( columns=columnHeaders, data=resonances )
        spinGroups.add(    resolvedResonanceModule.spinGroup(str(spinGroupIndex), JJ, pi, channels,
                           resolvedResonanceModule.resonanceParameters(table)) ) 
        #if verbose: print " J,pi =",JJ,pi,": partial waves",pw1,"to",partialWave,"\n"

    if verbose: print(" Read in ",partialWave," EDA partial waves")
    if partialWave != len(boundaries):
        print('\n\n ERROR:  %5i channels enumerated, but boundaries given for %5i  !!\n\n' % (partialWave,len(boundaries)))
        
    BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum    
    RMatrix = resolvedResonanceModule.RMatrix( 'eval', approximation, resonanceReactions, spinGroups, boundaryCondition=BC,
                relativisticKinematics=KRL, reducedWidthAmplitudes=bool(amplitudes), 
                supportsAngularReconstruction=True, calculateChannelRadius=False )

    resolved = resolvedResonanceModule.resolved( emin,emax,'MeV' )
    resolved.add( RMatrix )

#   scatteringRadius = Rm_global
    scatteringRadius = scatteringRadiusModule.scatteringRadius(
        constantModule.constant1d(Rm_global, domainMin=emin, domainMax=emax,
            axes=axesModule.axes(labelsUnits={1: ('energy_in', 'MeV'), 0: ('radius', 'fm')})) )
    unresolved = None
    resonances = resonancesModule.resonances( scatteringRadius, resolved, unresolved )
    gnd.resonances = resonances

    docnew = RMatrix.documentation
    docLines = [' ','Converted from EDA parameter file','   '+inFile,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']    
    computerCode = computerCodeModule.ComputerCode( label = 'R-matrix output', name = 'EDA', version = '') #, date = time.ctime() )
    computerCode.note.body = '\n'.join( docLines )     
    
    dataLines = ['Fixed variables']
    for jpi in rmatr:
        for chan in jpi[1:]:
            fix = chan[1]
            fc = ' '
            #print fix
            for f in fix:  fc += '%s ' % f
            dataLines += [fc]
    #print '\n'.join( dataLines )
    docFV = computerCodeModule.InputDeck( 'Fixed_variables in EDA', inFile, '\n'.join( dataLines ) )
    computerCode.inputDecks.add( docFV )
            
    dataLines = ['%5i    Fitted data normalizations  (and reciprocals)' % len(normalizations)]
    for n in normalizations:
        s1 = ' %-8s' % n[0]
        fc = 'f' if n[2] else ' '
        s2 = '%15.8e%c    %15.8e  %10.3f %12.8f' % (n[1],fc,n[3],n[4],1./n[1])
        s = s1 + s2.replace('e','E')
        dataLines += [s]
    docNorms = computerCodeModule.InputDeck( 'Fitted data normalizations in EDA', inFile, '\n'.join( dataLines ) )
    computerCode.inputDecks.add( docNorms )
    
    docnew.computerCodes.add( computerCode ) 

    if covlines is not None:
        pVarying,pMatrix = covariances
        nv = len(pVarying)
        print("Covariance matrix for",nv,"varied out of ",npars)
        R2icor = [pVarying[iv][0] for iv in range(nv)]
          
        matrix = numpy.zeros([npars,npars])
        for i in range(nv):
            for j in range(nv):
                matrix[R2icor[i],R2icor[j]] = pMatrix[i,j]

     # store into GNDS (need links to each spinGroup)
        parameters = covarianceModelParametersModule.parameters()
        startIndex = 0
        for spinGroup in resonances.resolved.evaluated:
            nParams = spinGroup.resonanceParameters.table.nColumns * spinGroup.resonanceParameters.table.nRows
            if nParams == 0: continue
            parameters.add( covarianceModelParametersModule.parameterLink(
                label = spinGroup.label, link = spinGroup.resonanceParameters.table, root="$reactions",
                matrixStartIndex=startIndex, nParameters=nParams
            ))
            startIndex += nParams
        if debug: 
            print(parameters.toXML(),'\n')
            print(type(matrix))
            print('matrix:\n',matrix)
        if True:
            correlation = numpy.zeros([npars,npars])
            if debug: print("Cov shape",matrix.shape,", Corr shape",correlation.shape)
            # print "\nCov diagonals:",[matrix[i,i] for i in range(npars)]
            # print "\nCov diagonals:\n",numpy.array_repr(numpy.diagonal(matrix),max_line_width=100,precision=3)
            print("Diagonal uncertainties:\n",numpy.array_repr(numpy.sqrt(numpy.diagonal(matrix)),max_line_width=100,precision=4))
            for i in range(npars):
                for j in range(npars): 
                    t = matrix[i,i]*matrix[j,j]
                    if t !=0: correlation[i,j] = matrix[i,j] / t**0.5

            from scipy.linalg import eigh
            eigval,evec = eigh(matrix)
            if debug:
                print("  Covariance eigenvalue     Vector")
                for kk in range(npars):
                    k = npars-kk - 1
                    print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print("Covariance eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))


            if False: print('correlation matrix:\n',correlation)
            eigval,evec = eigh(correlation)
            if debug:
                print("  Correlation eigenvalue     Vector")
                for kk in range(npars):
                    k = npars-kk - 1
                    print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print("Correlation eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))

        GNDSmatrix = arrayModule.flattened.fromNumpyArray(matrix, symmetry=arrayModule.symmetryLowerToken)
        # print GNDSmatrix.toXML()
        Type="absoluteCovariance"
        covmatrix = covarianceModelParametersModule.parameterCovarianceMatrix('eval', GNDSmatrix,
            parameters, type=Type )
        if verbose: print(covmatrix.toXML())
        rowData = covarianceSectionModule.rowData(gnd.resonances.resolved.evaluated,
                root='')
        parameterSection = covarianceModelParametersModule.parameterCovariance("resolved resonances", rowData)
        parameterSection.add(covmatrix)

        p,tex = elastics
        covarianceSuite = covarianceSuiteModule.covarianceSuite( p, tex, 'EDA R-matrix covariances' )
        covarianceSuite.parameterCovariances.add(parameterSection)

        if debug: print(covarianceSuite.toXMLList())
        if verbose: covarianceSuite.saveToFile('CovariancesSuite.xml')

    else:
        if noCov: print("     Covariance data ignored")
        else:     print("     No covariance data found")
        covarianceSuite = None

    return gnd,covarianceSuite

