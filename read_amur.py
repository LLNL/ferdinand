#

##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,eda,amur,hyrma    #
#                                            #
##############################################

from amur_file import getAMUR

from fudge import reactionSuite as reactionSuiteModule
from fudge import styles as stylesModule
from fudge import physicalQuantity as physicalQuantityModule
from fudge.reactions import reaction as reactionModule
from fudge.reactionData import crossSection as crossSectionModule
from fudge.processing.resonances.getCoulombWavefunctions import *
from fudge import documentation as documentationModule
import masses
import fudge.resonances.resonances as resonancesModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
import fudge.resonances.resolved as resolvedResonanceModule
import fudge.resonances.common as commonResonanceModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule

from PoPs import database as databaseModule
from PoPs import misc as miscModule
from PoPs.families import gaugeBoson as gaugeBosonModule
from PoPs.families import baryon as baryonModule
from PoPs.families import nucleus as nucleusModule
from PoPs.families import nuclide as nuclideModule
from PoPs.quantities import spin as spinModule
from PoPs.chemicalElements.misc import *

from pqu import PQU as PQUModule
from xData import table as tableModule
import xData.constant as constantModule
import xData.link as linkModule
from zeroReaction import *
from getCoulomb import *
import os,pwd,time
import fractions
fmscal = 0.0478450
etacns = 0.1574855
amu    = 931.494013
oneHalf = fractions.Fraction( '1/2' )
one = fractions.Fraction( '1' )
zero = fractions.Fraction( '0' )
spinUnit = spinModule.baseUnit

##############################################  read_amur

def read_amur(inFile,elastic, amplitudes, emin,emax, verbose,debug):
    
    lines = open(inFile).readlines()
    #comment, partitions, boundaries, rmatr, channelList, normalizations, nuclei  = getAMUR(lines,debug and False)
    title, partitions, boundaries, widthList, channelList, dataSets, nuclei = getAMUR(lines,debug) # and False)

    domain = stylesModule.ProjectileEnergyDomain(emin,emax,'MeV')
    style = stylesModule.Evaluated( 'eval', '', physicalQuantityModule.Temperature( 300, 'K' ), domain, 'from '+inFile , '0.1.0' )
    PoPs_data = databaseModule.Database( 'amur', '1.0.0' )
    resonanceReactions = commonResonanceModule.ResonanceReactions()
    MTchannels = []

    approximation = 'Full R-Matrix'
    
    KRL = False  # not relativistic
    LRP = 2  # do not reconstruct resonances pointwise (for now)
    
    BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum  # default
    eunit = 'MeV'
    is_rwa = True      # default in AMUR inputs
#   if elastic is None:
#       elastic = partitions[0][0]
#       print "### Elastic channel defaulted to",elastic
    Rm_global = 0.0

    ZAdict = {}
    rrList = []
    cm2lab = 0
    for part in partitions:
        zap = part[0]
        zat = part[2]
        p_q = nuclei[zap]
        t_q = nuclei[zat]
        if debug: print('proj',p_q)
        if debug: print('targ',t_q)
        ia,ep,et = 0, 0.0, 0.0  # Feature of AMUR: these options are not used
        
        pMass,tMass = p_q[2],t_q[2]
        pZ,tZ = zap/1000, zat/1000
        #  Use standard GND names:
        pA,tA = zap % 1000, zat % 1000

        p = idFromZAndA(pZ,pA)
        if pA==0: p = 'photon'
        t = idFromZAndA(tZ,tA)
        tex = nuclideIDFromIsotopeSymbolAndIndex(t,ia)
        rr = '%s + %s' % (p,tex)  # label for resonanceReaction
        #print "\nprojectile: p,pZ,pA,pMass=",p,pZ,pA,pMass
        rrList.append(rr)
    
        jp,ptyp = p_q[0],p_q[1]
        jt,ptyt = t_q[0],t_q[1]
        pp = ptyp
        pt = ptyt
        Jp = int(jp) if abs(jp - int(jp))<0.1 else '%i/2' % int(2*jp)
        Jt = int(jt) if abs(jt - int(jt))<0.1 else '%i/2' % int(2*jt)
        
        prmax = part[4]
        Qvalue = part[5] 
        QI = Qvalue - ep - et
        
        channelName = '%s + %s' % (p,tex)
        #ZAdict[rr] = (float(pMass),float(tMass),float(pZ),float(tZ),float(QI),float(prmax),Lmax)
        ZAdict[rr] = (pMass,tMass,pZ,tZ,QI,prmax)
        if debug: print('ZAdist[',rr,'] =',ZAdict[rr])
        #JTmax = max(JTmax,Lmax+Jp+Jt)

        MT = 5
        if p=='photon':            MT = 102
        elif p=='n' :              MT = 50+ia
        elif p=='H1' :             MT = 600+ia
        elif p=='H2' :             MT = 650+ia
        elif p=='H3' :             MT = 700+ia
        elif p=='He3' :            MT = 750+ia
        elif p=='He4' :            MT = 800+ia

        if debug: print(' projectile=',p,pMass,pZ,Jp,ptyp, ', target=',tex,tMass,tZ,Jt,ptyt, ' Q=',Qvalue,' MT=',MT,' prmax =',prmax)
        if pZ==0 and pMass == 0 :   # g
            projectile = miscModule.buildParticleFromRawData( gaugeBosonModule.Particle, p, mass = ( 0, 'amu' ), spin = (Jp,spinUnit ),  parity = (ptyp,'' ), charge = (0,'e') )
        elif pZ<1 and pMass > 0.5 and pMass < 1.5 and p != 'H1' :  # n or p
            projectile = miscModule.buildParticleFromRawData( baryonModule.Particle, p, mass = (pMass,'amu' ), spin = (Jp,spinUnit ),  parity = (ptyp,'' ), charge = (pZ,'e') )
        else: # nucleus in its gs
            nucleus = miscModule.buildParticleFromRawData( nucleusModule.Particle, p, index = 0, energy = ( 0.0, 'MeV' ) , spin=(Jp,spinUnit), parity=(ptyp,''), charge=(pZ,'e'))
            projectile = miscModule.buildParticleFromRawData( nuclideModule.Particle, p, nucleus = nucleus,  mass=(pMass,'amu'))
        PoPs_data.add( projectile )

        # Some state of target at energy 'et':
        if debug: print("Build PoPs for target ",tex,Jt,ptyt,tZ,tMass,ia,et)
        nucleus = miscModule.buildParticleFromRawData( nucleusModule.Particle, tex, index = ia, energy = (et,'MeV' ) , spin=(Jt,spinUnit), parity=(ptyt,''), charge=(tZ,'e') )
        target = miscModule.buildParticleFromRawData( nuclideModule.Particle, tex, nucleus = nucleus, mass=(tMass,'amu'))
        PoPs_data.add( target )

        #if int(elastic) == zap:
        if elastic == rr:
            print("### Elastic channel set from projectile",elastic)
            elastics = p,tex
            MT = 2
            cm2lab = (tMass + pMass)/tMass

        # Create zero background cross section
        MTchannels.append((rr,zeroReaction(rr,MT, QI, [projectile,target], None, emin,emax,eunit, debug), channelName,prmax,p))
        compoundA = pA + tA
        compoundZ = pZ + tZ
            
    if cm2lab<1e-5:
        print("Missed elastic channel for cm2lab factor!")
        raise SystemExit

    damped = False

#  After making all the channels, and gnd is generated for the elastic channel, now add them to gnd
    p,tex = elastics   
    gnd = reactionSuiteModule.ReactionSuite( p, tex, 'AMUR R-matrix fit', PoPs =  PoPs_data, style = style, interaction='nuclear')
    Rm_global = prmax  #  choose a radius. Here: the last radius

    for rr,reaction,channelName,prmax,p in MTchannels:
#  Get zero background cross section and link to it
        #reaction,channelName,prmax = MTchannels[rr]
        gnd.reactions.add(reaction)
        eliminated = False
        link = linkModule.Link(reaction)
        if prmax is not None and prmax != Rm_global:
            scatRadius = scatteringRadiusModule.ScatteringRadius( 
                constantModule.Constant1d(prmax, domainMin=emin, domainMax=emax,
                    axes=axesModule.Axes(labelsUnits={1: ('energy_in', eunit), 0: ('radius', 'fm')})) )
        else:
            scatRadius = None
        rreac = commonResonanceModule.ResonanceReaction ( label=rr, link=link, ejectile=p, Q=None, eliminated=eliminated, scatteringRadius = scatRadius  )
        reaction.updateLabel( )
        resonanceReactions.add(rreac)
        if debug: print("RR <"+rr+"> is "+channelName)

#  Now read and collate the reduced channel partial waves and their reduced width amplitudes
# next we have NJS spin groups, each containing channels and resonances

    NJS = len(channelList)
    # partialWave = index of LANL channel order
    spinGroups = resolvedResonanceModule.SpinGroups()
    for spinGroupIndex in range(NJS):
        chans = channelList[ spinGroupIndex ]
        J,piv = chans[0]
        JJ = resolvedResonanceModule.Spin( J )
        pi= resolvedResonanceModule.Parity( piv )
        if verbose  or debug: print('\n ##### Spinset #',spinGroupIndex,': J,pi =',JJ,piv,'\n',chans)
    

        columnHeaders = [ tableModule.ColumnHeader(0, name="energy", unit="MeV") ]
        width_units = 'MeV'   ##   'MeV**{0.5}' if amplitudes else 'MeV'  # wrong units given to GND: correct later if needed
        channelNames = []
        channels = resolvedResonanceModule.Channels()
        NCH = len(chans)-1

        for chidx in range(NCH):
            part,lch,sch = chans[chidx+1]
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

            columnHeaders.append( tableModule.ColumnHeader(chidx+1, name=channelName, unit= width_units) )

            Sch = resolvedResonanceModule.Spin( sch )
            channels.add( resolvedResonanceModule.Channel(str(chidx+1), rr, columnIndex=chidx+1, 
                    L=lch, channelSpin=Sch, boundaryConditionValue = None ))
                            
            if debug: print(str(chidx), str(chidx), int(lch), float(sch), chidx+1, 'B=',BC)
        
        resonances = []
        rmatset = widthList[spinGroupIndex]
        energies = rmatset[1:]
        nle = len(energies)
        if debug: print(" rmatset: ",rmatset)
        if debug: print(" Energies: ",energies)

        for level in range(nle):
            energy = energies[level]
            if energy is None: continue
            row = [energy[0]*cm2lab]
            if debug: print(" Energy: ",energy,row,energy[1:])
            for ich in range(NCH):
                #print " W for",ich+1," for all levels:",rmatset[ich+2][0]
                part,lch,sch = chans[ich+1]
                rr = rrList[part] 
                w = 0.0
                for win in energy[1:]:
                    if win[0] == ich+1:
                        w = win[1]
                        break
                     
                if debug: print("level",level+1," ch",ich+1," rwa=",w)
                if is_rwa != amplitudes:   # fix to give correct output: rwa or formal width
                    pMass,tMass,pZ,tZ,QI,prmax = ZAdict[ rr ]
                    e_ch = energy[0] + QI
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

        table = tableModule.Table( columns=columnHeaders, data=resonances )
        spinGroups.add(    resolvedResonanceModule.SpinGroup(str(spinGroupIndex), JJ, pi, channels,
                           resolvedResonanceModule.ResonanceParameters(table))  ) 
        #if verbose: print " J,pi =",JJ,piv,": partial waves",pw1,"to",partialWave,"\n"

    if verbose: print(" Read in AMUR R-matrix parameters")

    BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum    
    RMatrix = resolvedResonanceModule.RMatrix( 'eval', approximation, resonanceReactions, spinGroups, boundaryCondition=BC,
                relativisticKinematics=KRL, reducedWidthAmplitudes=bool(amplitudes), 
                supportsAngularReconstruction=True, calculateChannelRadius=False )

    resolved = resolvedResonanceModule.Resolved( emin,emax,'MeV' )
    resolved.add( RMatrix )

    scatteringRadius = scatteringRadiusModule.ScatteringRadius(
        constantModule.Constant1d(Rm_global, domainMin=emin, domainMax=emax,
            axes=axesModule.Axes(labelsUnits={1: ('energy_in', 'MeV'), 0: ('radius', 'fm')})) )
    unresolved = None
    resonances = resonancesModule.Resonances( scatteringRadius, None, resolved, unresolved )
    gnd.resonances = resonances

    docLines = [' ','Converted from AMUR parameter file','   '+inFile,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']
    doc = documentationModule.documentation( 'ENDL', '\n'.join( docLines ) )
    gnd.styles[0].documentation['ENDL'] = doc
    dataLines = []

    dataLines = ['Fixed variables']

    doc = documentationModule.documentation( 'Fixed_variables', '\n'.join( dataLines ) )
    gnd.styles[0].documentation['Fixed_variables'] = doc
            
    dataLines = ['%5i    Fitted data normalizations' % len(dataSets)]
    for n in dataSets:
        try:
            #s = '%20s %5i points, %10.3f chisq, norm =%s' % (n[0],n[1],n[2],str(n[3]))
            s = '%20s %5i points, %10.3f chisq, norm =%s' % (n[0],n[1]['npoints'],n[1]['Chisq'],n[1].get('normalization','N/A'))
        except:
            print("Not 4 values:",n)
            s = ' '
        dataLines += [s]
    doc = documentationModule.documentation( 'Fitted_data', '\n'.join( dataLines ) )
    gnd.styles[0].documentation['Fitted_data'] = doc

    return gnd

