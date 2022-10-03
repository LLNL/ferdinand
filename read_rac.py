#

##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,eda,amur,rac,hyrma    #
#                                            #
##############################################

from rac_file import getRAC

from fudge import reactionSuite as reactionSuiteModule
from fudge import styles as stylesModule
from fudge import physicalQuantity as physicalQuantityModule
from fudge.reactions import reaction as reactionModule
from fudge.reactionData import crossSection as crossSectionModule
from fudge.processing.resonances.getCoulombWavefunctions import *
from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode as computerCodeModule

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
from PoPs.groups.misc import *

from pqu import PQU as PQUModule
from xData import table as tableModule
import xData.constant as constantModule
import xData.link as linkModule
from zeroReaction import *
from getCoulomb import *
import os,pwd,time,sys
import fractions
fmscal = 0.0478450
etacns = 0.1574855
amu    = 931.494013
oneHalf = fractions.Fraction( '1/2' )
one = fractions.Fraction( '1' )
zero = fractions.Fraction( '0' )
spinUnit = spinModule.baseUnit

##############################################  read_rac

def read_rac(inFile,elastic, amplitudes, emin,emax, Lvals,wzero, verbose,debug):
    
    lines = open(inFile).readlines()
    #comment, partitions, boundaries, rmatr, channelList, normalizations, nuclei  = getRAC(lines,debug and False)
    #title, partitions, boundaries, widthList, channelList, dataSets, nuclei = getRAC(lines,debug and False)
    partitions, widthList, channelList, dataSets, nuclei = getRAC(lines,wzero,debug and False)


    domain = stylesModule.projectileEnergyDomain(emin,emax,'MeV')
    style = stylesModule.evaluated( 'eval', '', physicalQuantityModule.temperature( 300, 'K' ), domain, 'from '+inFile , '0.1.0' )
    PoPs_data = databaseModule.database( 'rac', '1.0.0' )
    resonanceReactions = commonResonanceModule.resonanceReactions()
    computePenetrability = True
    MTchannels = []

    approximation = 'Full R-Matrix'
    
    KRL = False  # not relativistic
    LRP = 2  # do not reconstruct resonances pointwise (for now)
    
    BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum  # default
    eunit = 'MeV'
    is_rwa = True      # default in RAC inputs
    Qbase = None
    if elastic is None:
        elastic = partitions[0][0]
        print("### Elastic projectile defaulted to",elastic)
        Qbase = 0.0
    else:
        for part in partitions:
            print('Compare',elastic,'to',part[0],elastic == part[0])
            if str(elastic) == str(part[0]): 
                Qbase = part[3]   # make sure new elastic is Q=0
                break
    if Qbase is not None:
        print('Q-value to new elastic=',Qbase)
    else:
        print('Elastic',elastic,'not found in list',[part[0] for part in partitions])
        sys.exit()

    ZAdict = {}
    rrList = []
    cm2lab = 0
    Rm_global = None
    last_excited = {}; Q4gs = {}
    reactionLabelList = []
    for part in partitions:
        zap = part[0]
        zat = part[1]
        prmax = part[2]
        Qvalue = part[3]

        p_q = nuclei[zap]
        t_q = nuclei[zat]
        if debug: print('proj',p_q)
        if debug: print('targ',t_q)
        ep = 0.0
        if zat not in last_excited.keys(): 
            last_excited[zat] = -1
            Q4gs[zat] = Qvalue
        ia = last_excited[zat] + 1
        et = Q4gs[zat] - Qvalue 
        last_excited[zat] = ia
        QI = Qvalue - Qbase
        
        pMass,tMass = p_q[2],t_q[2]
        
        pZ,tZ = zap//1000, zat//1000
        pA,tA = zap % 1000, zat % 1000
        print(' Partition:',part,': projectile=',pZ,pA,'on target',tZ,tA," ia,et=",ia,et,'QI:',QI)

        if pA==0: 
            p = 'photon'
        else:
            p = idFromZAndA(pZ,pA)
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
        

        if Rm_global is None: Rm_global = prmax
        
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
#       if pZ==0 and pMass == 0 :   # g
        if p == 'photon':
            pMass = 0.0
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

        if int(elastic) == zap and ia==0:
            print("### Elastic channel set from channel",rr)
            elastics = p,tex
            MT = 2
            cm2lab = (tMass + pMass)/tMass

        # Create zero background cross section
        reactionLabelList.append(rr)
        MTchannels.append((rr,zeroReaction(rr,MT, QI, [projectile,target], None, emin,emax,eunit, debug), channelName,prmax,p,False))
        compoundA = pA + tA
        compoundZ = pZ + tZ
            
    if cm2lab<1e-5:
        print("Missed elastic channel for cm2lab factor!")
        raise SystemExit

# Check if any damping and hence need for Reich-Moore channel
    damped = False
    for spinGroupIndex in range(len(channelList)):
        rmatset = widthList[spinGroupIndex]
        energies = rmatset[1:]    
        for energy in energies:
            damp = energy[0][1]*cm2lab
            damped = damped or damp > 2e-6

    print('Damped:',damped)
    if damped:
        print('Reactions so far:',reactionLabelList)
        approximation = 'Reich_Moore'
        level = 0
        cMass = masses.getMassFromZA( compoundZ*1000 + compoundA )
        compoundName = idFromZAndA(compoundZ,compoundA)
        compoundNameIndex = nuclideIDFromIsotopeSymbolAndIndex(compoundName,level)

        gchannelName = 'photon + %s' % compoundName 
        Q = (pMass + tMass - cMass)*amu
        rrcap = 'photon + '+compoundName
        rrcapr= compoundName + ' + photon'
        print(" Create Reich-Moore channel '%s' from damping" % rrcap)

        print("Reich-Moore particle pair: ",gchannelName,' with CN mass %.5f so Q=%.3f, label=%s' % (cMass,Q,rrcap))

#       gData = { '0' : [ 0.0,       .0,           1, None,    1,     +1 ] }
        gammaParticle =  miscModule.buildParticleFromRawData( gaugeBosonModule.particle, 'photon',
            mass = ( 0, 'amu' ), spin = ( zero, spinUnit ),  parity = ( 1, '' ), charge = ( 0, 'e' ))
        PoPs_data.add(gammaParticle)

        nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, compoundNameIndex, index = level, energy = ( 0.0, 'MeV' ) ,
                                                       spin=(zero,spinUnit), parity=(1,''), charge=(compoundZ,'e') )
        compoundParticle = miscModule.buildParticleFromRawData( nuclideModule.particle, compoundNameIndex, nucleus = nucleus, mass=(cMass,'amu') )
        #print PoPs_data.toXML()
        productList = [gammaParticle,compoundParticle]
        if rrcap in reactionLabelList or rrcapr in reactionLabelList: 
            print("'%s' already in reaction list" % rrcap)
            rrcap = 'damping'
        PoPs_data.add(compoundParticle)
# Create background ReichMoore cross section (zero to start with)
        MT_capture = 102
        label = 'capture'
        capture = zeroReaction(label,MT_capture, Q, productList, 'damping', emin,emax,'MeV', debug)
        print("RM label:",capture.label)
        MTchannels.insert(0,(rrcap, capture, gchannelName,None,'photon',True))        
        
#  After making all the channels, and gnd is generated for the elastic channel, now add them to gnd
    p,tex = elastics   
    gnd = reactionSuiteModule.reactionSuite( p, tex, 'RAC R-matrix fit', PoPs =  PoPs_data, style = style, interaction='nuclear')

    for rr,reaction,channelName,prmax,p,eliminated in MTchannels:
#  Get zero background cross section and link to it
        #reaction,channelName,prmax = MTchannels[rr]
        print('    Add %s reaction "%s"' % (rr,reaction.label),' (eliminated)' if eliminated else '')
        gnd.reactions.add(reaction)
            
        reactionLink = linkModule.link(reaction)
        computeShiftFactor =  not eliminated
        rreac = commonResonanceModule.resonanceReaction ( label=rr, reactionLink=reactionLink, ejectile=p, computePenetrability=True,
                                                     computeShiftFactor=computeShiftFactor, Q=None, eliminated=eliminated  )
        if prmax is not None and prmax != Rm_global:
            rreac.scatteringRadius = scatteringRadiusModule.scatteringRadius(      
                constantModule.constant1d(prmax, domainMin=emin, domainMax=emax,
                    axes=axesModule.axes(labelsUnits={1: ('energy_in', eunit), 0: ('radius', 'fm')})) )
        resonanceReactions.add(rreac)
        reaction.updateLabel( )
        if debug: print("RR <"+rr+"> is "+channelName)

#  Now read and collate the reduced channel partial waves and their reduced width amplitudes
# next we have NJS spin groups, each containing channels and resonances

    NJS = len(channelList)
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
        firstp =1
        if damped:
            columnHeaders.append( tableModule.columnHeader(1, name=gchannelName, unit= width_units) )
            Sch = resolvedResonanceModule.spin( 0.0 )
            channels.add( resolvedResonanceModule.channel('1', rrcap, columnIndex=1, L=0, channelSpin=Sch) )
            firstp = 2
                    
        NCH = len(chans)-1
        for chidx in range(NCH):
            part,sch,lch = chans[chidx+1]
            rr = rrList[part-1] 
            #if debug: print "From p,t =",p,t," find channel ",rr
            thisChannel = resonanceReactions[rr]
            channelName = "%s width" % thisChannel.label

            jdx = 2
            while True:
                if channelName not in channelNames:
                    channelNames.append( channelName ); break
                channelName = '%s width_%d' % (thisChannel.label, jdx)
                jdx += 1

            columnHeaders.append( tableModule.columnHeader(chidx+firstp, name=channelName, unit= width_units) )

            Sch = resolvedResonanceModule.spin( sch )
            channels.add( resolvedResonanceModule.channel(str(chidx+firstp), rr, columnIndex=chidx+firstp, 
                    L=lch, channelSpin=Sch, boundaryConditionValue = None ))
                            
            if debug: print(str(chidx), str(chidx), int(lch), float(sch), chidx+firstp, 'B=',BC)
        
        resonances = []
        rmatset = widthList[spinGroupIndex]
        energies = rmatset[1:]
        nle = len(energies)
        if debug: print(" rmatset: ",rmatset)
        if debug: print(" Energies: ",energies)

        for level in range(nle):
            energy = energies[level]
            damp = energy[0][1]*cm2lab
            row = [energy[0][0]*cm2lab]
            if damped:
                row.append(damp)
            if debug: print(" Energy: ",energy,damp) #row,energy[1:]
            
            for ich in range(NCH):
                #print " W for",ich+1," for all levels:",rmatset[ich+2][0]
                part,sch,lch = chans[ich+1]
                rr = rrList[part-1] 
                w = 0.0
                for win in energy[1:]:
                    if win[0] == ich+1:
                        w = win[1]
                        break
                     
                if debug: print("level",level+1," ch",ich+1," rwa=",w)
                if is_rwa != amplitudes:   # fix to give correct output: rwa or formal width
                    pMass,tMass,pZ,tZ,QI,prmax = ZAdict[ rr ]
                    e_ch = energy[0][0] + QI
#                   if abs(e_ch) < 1e-10:
#                       print('level',level,'in Jpi',JJ,pi,'at',energy[0][0],'+',QI,'so', e_ch,'for w=',w)
                    penetrability,shift,dSdE,W = getCoulomb_PSdSW(
                              e_ch,lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, False)   # CWF at abs(e_ch)
                    #if debug: print 'p,t =',p,tex,': call coulombPenetrationFactor(L=',lch,'R=',prmax,'e_ch=',e_ch,') =',penetrability,dSdE,W
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
        spinGroups.add( resolvedResonanceModule.spinGroup(str(spinGroupIndex), JJ, pi, channels,
                        resolvedResonanceModule.resonanceParameters(table)) )
        #if verbose: print " J,pi =",JJ,pi,": partial waves",pw1,"to",partialWave,"\n"

    if verbose: print(" Read in RAC R-matrix parameters")

    BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum    
    RMatrix = resolvedResonanceModule.RMatrix( 'eval', approximation, resonanceReactions, spinGroups, boundaryCondition=BC,
                relativisticKinematics=KRL, reducedWidthAmplitudes=bool(amplitudes), 
                supportsAngularReconstruction=True, calculateChannelRadius=False )

    resolved = resolvedResonanceModule.resolved( emin,emax,'MeV' )
    resolved.add( RMatrix )

    scatteringRadius = scatteringRadiusModule.scatteringRadius(      
                constantModule.constant1d(Rm_global, domainMin=emin, domainMax=emax,
                    axes=axesModule.axes(labelsUnits={1: ('energy_in', eunit), 0: ('radius', 'fm')})) )
    unresolved = None
    resonances = resonancesModule.resonances( scatteringRadius, resolved, unresolved )
    gnd.resonances = resonances

    docnew = RMatrix.documentation
    docLines = [' ','Converted from RAC parameter file','   '+inFile,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']
    computerCode = computerCodeModule.ComputerCode( label = 'R-matrix fit', name = 'RAC', version = '', date = time.ctime() )
    computerCode.note.body = '\n'.join( docLines )

#     dataLines = ['Fixed variables']
# 
#     doc = documentationModule.documentation( 'Fixed_variables', '\n'.join( dataLines ) )
#     gnd.styles[0].documentation['Fixed_variables'] = doc
            
    dataLines = ['%5i    Fitted data normalizations' % len(dataSets)]
    for n in dataSets:
        s = '%20s %5i points, %10.3f chisq, norm =%s' % (n[0],n[1],n[2],str(n[3]))
        dataLines += [s]
    inputDataSpecs= computerCodeModule.InputDeck( 'Fitted_data 1', ('\n  %s\n' % time.ctime() )  + ('\n'.join( dataLines ))+'\n' )
    computerCode.inputDecks.add( inputDataSpecs )
    docnew.computerCodes.add( computerCode )
    
    print (gnd.toXML(), file = open('rac.xml','w') )
    return gnd

