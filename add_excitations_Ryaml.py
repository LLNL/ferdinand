#! /usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.60, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma,tex,yaml    #
#                                            #
##############################################

import math,numpy,time,pwd,os
import fractions
from pqu import PQU as PQUModule
from fudge import reactionSuite as reactionSuiteModule

# from fudge.processing.resonances.getCoulombWavefunctions import *
import fudge.resonances.resonances as resonancesModule
import fudge.resonances.resolved as resolvedResonanceModule
import fudge.resonances.common as commonResonanceModule
from fudge import styles as stylesModule
from fudge import physicalQuantity as physicalQuantityModule
from fudge.reactions import reaction as reactionModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
import fudge.covariances.modelParameters as covarianceModelParametersModule
from fudge.covariances import enums as covarianceEnumsModule
import fudge.covariances.covarianceSection as covarianceSectionModule
import fudge.covariances.covarianceSuite as covarianceSuiteModule

from fudge import documentation as documentationModule
from xData.Documentation import computerCode as computerCodeModule
from xData.Documentation import exforDataSet as ExforDataSetModule

import masses
from zeroReaction import *
from PoPs import database as databaseModule
from PoPs import misc as miscModule
from PoPs.families import gaugeBoson as gaugeBosonModule
from PoPs.families import baryon as baryonModule
from PoPs.families import nucleus as nucleusModule
from PoPs.families import nuclide as nuclideModule
from PoPs.quantities import spin as spinModule
from PoPs.chemicalElements.misc import *
import xData.link as linkModule
import xData.constant as constantModule
from xData import table as tableModule
import xData.xDataArray as arrayModule
from xData import date

from yaml import load
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

oneHalf = fractions.Fraction( '1/2' )
one = fractions.Fraction( '1' )
zero = fractions.Fraction( '0' )
spinUnit = spinModule.baseUnit

def nuclIDs (nucl):
    datas = chemicalElementALevelIDsAndAnti(nucl)
    if datas[1] is not None:
        return datas[1]+str(datas[2]),datas[3]
    else:
        return datas[0],0

##############################################  write_Ryaml / Rjson

def add_excitations_Ryaml(gnds_file, newExcitations, defaultWidth, NLmax, verbose,debug):
  
    print("Read gnds file",gnds_file,'and add excited states from file',newExcitations,'\n')
    
    gnds = reactionSuiteModule.ReactionSuite.readXML_file( gnds_file )
    proj,targ = gnds.projectile,gnds.target
    elasticChannel = '%s + %s' % (proj,targ)
    PoPs = gnds.PoPs

    
    ifile = open(newExcitations,'r')
    data = load(ifile, Loader=Loader)       
    
    Particles = data['Particles']
    Reactions = data['Reactions']
    
    rrr = gnds.resonances.resolved
    Rm_Radius = gnds.resonances.getScatteringRadius()
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    
    domain = gnds.styles.getEvaluatedStyle().projectileEnergyDomain
    energyUnit = domain.unit
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs(energyUnit)
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs(energyUnit)    

    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes  
    approximation = RMatrix.approximation
    widthUnit = energyUnit+'**(1/2)' if IFG else energyUnit
    
    # FIXED DEFAULTS FOR NOW:
    RelativisticKinematics = False

    
    style = stylesModule.Evaluated( 'eval', '', 
        physicalQuantityModule.Temperature( 300, 'K' ), domain, 'from '+gnds_file+' + ',newExcitations , '0.1.0' )
    
    if debug: 
        print('New particles:\n',Particles)
        print('New reactions:\n',Reactions)

# PARTICLES
    newTargets = []
    for id in Particles.keys():
        p = Particles[id]
        gndsName = p['gndsName']
        if gndsName in PoPs.keys(): continue
        m = p['gsMass']
        Z  = p['charge']
        s = p.get('spin',None)
        pt = p.get('parity',None)
        ex = p.get('excitation',0.0)
        if s is not None:
            if int(2*s) % 2 == 0:
                s = int(s)
            else:
                s = '%i/2' % int(2*s)
        if True: print(id,'is',gndsName,m,Z,s,pt,ex)
#         level = 0 if ex == 0.0 else 1  # FIXME
        level = 0 if '_e' not in gndsName else float(gndsName.split('_e')[1])

        if Z==0 and m == 0 :   # g
            particle = miscModule.buildParticleFromRawData( gaugeBosonModule.Particle, gndsName, mass = ( 0, 'amu' ), spin = (s,spinUnit ),  parity = (pt,'' ), charge = (0,'e') )
        elif Z<1 and m > 0.5 and m < 1.5 and gndsName != 'H1' :  # n or p
            particle = miscModule.buildParticleFromRawData( baryonModule.Particle, gndsName, mass = (m,'amu' ), spin = (s,spinUnit ),  parity = (pt,'' ), charge = (Z,'e') )
        else: # nucleus in its gs
            if s is not None and pt is not None:
                nucleus = miscModule.buildParticleFromRawData( nucleusModule.Particle, gndsName, index = level, energy = ( ex, energyUnit) , spin=(s,spinUnit), parity=(pt,''), charge=(Z,'e'))
            else:
                nucleus = miscModule.buildParticleFromRawData( nucleusModule.Particle, gndsName, index = level, energy = ( ex, energyUnit) , charge=(Z,'e'))
            particle = miscModule.buildParticleFromRawData( nuclideModule.Particle, gndsName, nucleus = nucleus,  mass=(m,'amu'))
            newTargets.append(gndsName)

        PoPs.add( particle )
 
    print('New targets:',newTargets)

# REACTIONS

    reactionOrder = Reactions.get('order', list(Reactions.keys()) )
    new_resonanceReactions = commonResonanceModule.ResonanceReactions()
    new_Reactions = []
    for id in reactionOrder:
        partition = Reactions[id]
        label = partition['label']
        print('partition',id,':',label)
        if label in gnds.reactions: 
            rreac = RMatrix.resonanceReactions[label]
            new_resonanceReactions.add(rreac)
            continue
        
        ejectile = partition['ejectile']
        residual = partition['residual']
        Q = partition['Q']
        prmax = partition.get('scatteringRadius',Rm_global)
        eject,i = nuclIDs(ejectile)
        resid,level = nuclIDs(residual)
        B = partition.get('B',None)

        MT = 5
        if   ejectile=='n' :              MT = 50+level
        elif ejectile=='H1' :             MT = 600+level
        elif ejectile=='H2' :             MT = 650+level
        elif ejectile=='H3' :             MT = 700+level
        elif ejectile=='He3' :            MT = 750+level
        elif ejectile=='He4' :            MT = 800+level
        elif ejectile[:6]=='photon':      MT = 900+level
        if label == elasticChannel:       MT = 2
        
        reaction  = zeroReaction(label,MT, Q, [PoPs[ejectile],PoPs[residual]], None, emin,emax,energyUnit, debug)
        gnds.reactions.add(reaction)
        eliminated = False
        link = linkModule.Link(reaction)
        if prmax is not None and prmax != Rm_global:
            scatRadius = scatteringRadiusModule.ScatteringRadius(      
                constantModule.Constant1d(prmax, domainMin=emin, domainMax=emax,
                    axes=axesModule.Axes(labelsUnits={1: ('energy_in', energyUnit), 0: ('radius', 'fm')})) )
        else:
            scatRadius = None
        rreac = commonResonanceModule.ResonanceReaction ( label=label, link=link, ejectile=ejectile, Q=None, eliminated=eliminated, scatteringRadius = scatRadius  )
        reaction.updateLabel( )
        new_resonanceReactions.add(rreac)
        new_Reactions.append(label)
    print('New Reactions:',new_Reactions)

# SPINGROUPS
    spinGroups = resolvedResonanceModule.SpinGroups()

    spinGroupIndex = 0
    for Jpi in RMatrix.spinGroups:
        J = str(Jpi.spin)
        parity = int(Jpi.parity)
        pi = '+' if parity>0 else '-'
        if True: print("\nSpin group:",J,pi)

        JJ = resolvedResonanceModule.Spin( J )
        pi= resolvedResonanceModule.Parity( parity )
        if verbose  or debug: print('\n ##### Spinset #',spinGroupIndex,': J,pi =',J,pi)
    
        columnHeaders = [ tableModule.ColumnHeader(0, name="energy", unit=energyUnit) ]
        channelNames = []
        new_channels = resolvedResonanceModule.Channels()
        
        R = Jpi.resonanceParameters.table
        poleEnergies = R.getColumn('energy',energyUnit)
        widths = [R.getColumn( col.name, widthUnit ) for col in R.columns if col.name != 'energy']
        rows = len(poleEnergies)
        columns = len(widths)
        if True: print('r*c =',rows,columns)
        
        chidx = 0
        for ch in Jpi.channels:
            n = ch.columnIndex
            rr = ch.resonanceReaction
            rreac = RMatrix.resonanceReactions[rr]
            label = rreac.label
            lch = ch.L
            sch = float(ch.channelSpin)
            B = ch.boundaryConditionValue            
            if B is not None: channelBCOverrides += 1
            
#             if debug: print("From p,t =",p,t," find channel ",rr)
            thisChannel = RMatrix.resonanceReactions[rr]
            channelName = "%s width" % thisChannel.label

            jdx = 2
            while True:
                if channelName not in channelNames:
                    channelNames.append( channelName ); break
                channelName = '%s width_%d' % (thisChannel.label, jdx)
                jdx += 1
                
            
            

            columnHeaders.append( tableModule.ColumnHeader(chidx+1, name=channelName, unit= widthUnit ) )

            Sch = resolvedResonanceModule.Spin( sch )
            new_channels.add( resolvedResonanceModule.Channel(str(chidx+1), rr, columnIndex=chidx+1, 
                    L=lch, channelSpin=Sch, boundaryConditionValue = B ))
                            
            if debug: print(channelName, str(chidx),'LS:', int(lch), float(sch), chidx+1, 'B=',B)
            chidx += 1    

        # ADD NEW CHANNELS for NEW PARTITIONS with NEW TARGETS:
        for new_Reaction in new_Reactions:
            key = new_Reaction.replace(' ','')
            p,t = key.split('+')
            projectile = PoPs[p];
            target     = PoPs[t];
            pMass = projectile.getMass('amu');   tMass =     target.getMass('amu');
            if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
            if hasattr(target, 'nucleus'):     target = target.nucleus
            pZ    = projectile.charge[0].value;  tZ  = target.charge[0].value        
            jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0 # projectile.energy[0].float('MeV')
            jt,tt,et = target.spin[0].float('hbar'),     target.parity[0].value,     target.energy[0].float('MeV')      
            B     = Reactions[key].get('boundaryCondition',None)
            prmax = Reactions[key].get('scatteringRadius',None)
            iSmin2 = int(abs(jp-jt)*2+0.5)
            iSmax2 = int(abs(jp+jt)*2+0.5)+2
            if verbose: print('\nNew channel: <'+new_Reaction+'>', 'B=',B, 'R=',prmax,' and p,t,tZ,jt,tt,et:',p,t,tZ,jt,tt,et)
#             print('S range',iSmin2/2.,'to',iSmax2/2.-1,'from J',jp,jt,' pi',pt,tt)
            for iS2 in range(iSmin2,iSmax2,2):
                sch = iS2/2.0
                iLmin = int(abs(JJ-sch)+0.5)
                iLmax = int(abs(JJ+sch)+0.5)+1
#                 print('S=',sch)
                NL = 0
                for lch in range(iLmin,iLmax):
#                     print('L=',lch,' parity',parity != pt * tt * (-1)**lch)
                    if parity != pt * tt * (-1)**lch: break
                    NL += 1
#                     print('NL:',NLmax is not None and NL > NLmax)
                    if NLmax is not None and NL > NLmax: break
#                     print('go')
                    channelName = "%s width" % new_Reaction
        
                    jdx = 2
                    while True:
                        if channelName not in channelNames:
                            channelNames.append( channelName ); break
                        channelName = '%s width_%d' % (new_Reaction, jdx)
                        jdx += 1
                        
                    columnHeaders.append( tableModule.ColumnHeader(chidx+1, name=channelName, unit= widthUnit ) )
        
                    Sch = resolvedResonanceModule.Spin( sch )
                    new_channels.add( resolvedResonanceModule.Channel(str(chidx+1), new_Reaction, columnIndex=chidx+1, 
                            L=lch, channelSpin=Sch, boundaryConditionValue = B ))
                                    
                    if verbose: print('Found w: <'+channelName+'>', str(chidx),'LS:', int(lch), float(sch), chidx+1, 'B=',B, 'R=',prmax)
                    for row in range(rows):
                        R[row].append(defaultWidth)
                    chidx += 1    

        rmatr = R
        print("R matrix now",chidx,'columns')
        nle = len(poleEnergies)
        if debug: print(" Energies: ",energies)
#         if debug: print(" rmatr: ",R)


        table = tableModule.Table( columns=columnHeaders, data=rmatr )
        spinGroups.add(    resolvedResonanceModule.SpinGroup(str(spinGroupIndex), JJ, pi, new_channels,
                           resolvedResonanceModule.ResonanceParameters(table)) ) 
        #if verbose: print " J,pi =",J,piv,": partial waves",pw1,"to",partialWave,"\n"
        spinGroupIndex += 1
    if verbose: print(nVarPar," covIndices in new order with",nFixedPars,"fixed")
    if verbose: print(" covIndices in new order: ",covIndices)
        
    RMatrix = resolvedResonanceModule.RMatrix( 'eval', approximation, new_resonanceReactions, spinGroups, boundaryCondition=BC,
                relativisticKinematics=RelativisticKinematics, reducedWidthAmplitudes=bool(IFG), 
                supportsAngularReconstruction=True, calculateChannelRadius=False )

    resolved = resolvedResonanceModule.Resolved( emin,emax,energyUnit )
    resolved.add( RMatrix )

#   scatteringRadius = Rm_global
    scatteringRadius = scatteringRadiusModule.ScatteringRadius(
        constantModule.Constant1d(Rm_global, domainMin=emin, domainMax=emax,
            axes=axesModule.Axes(labelsUnits={1: ('energy_in', energyUnit), 0: ('radius', 'fm')})) )
    unresolved = None
    resonances = resonancesModule.Resonances( scatteringRadius, None, resolved, unresolved )
    gnds.resonances = resonances

    docnew = RMatrix.documentation
    docLines = [' ','Added excitations from Ryaml parameter file','   '+newExcitations,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']    
    computerCode = computerCodeModule.ComputerCode( label = 'R-matrix fit', name = 'Ryaml', version = '') #, date = time.ctime() )
    computerCode.note.body = '\n'.join( docLines )     
    
    docExcitations = computerCodeModule.InputDeck( 'Added excitationsfrom Ryaml', newExcitations )
    computerCode.inputDecks.add( docExcitations )
    docnew.computerCodes.add( computerCode ) 
    
#     dataSets = ExforDataSetModule.ExforDataSet( name,  "2021-01-21")
# 
#     
#     docnew.experimentalDataSets.add( dataSets )

          

    return gnds
        
################################# MAIN PROGRAM
if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Translate R-matrix evaluations from Ryaml to GNDS')

    parser.add_argument('gnds_file', type=str, help='The input file you want to add to.' )
    parser.add_argument('newExcitations', type=str, help='The input Ryaml file with new excited states.' )
    parser.add_argument('defaultWidth', type=float, default="9.1",  help="default width for new channels")
    parser.add_argument("-N", "--NLmax", type=int, help="Max number of partial waves in one reaction pair")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug output (more than verbose)")
# Process command line options
    args = parser.parse_args()
        
    gnds = add_excitations_Ryaml(args.gnds_file, args.newExcitations, args.defaultWidth, args.NLmax, args.verbose,args.debug)
    NLmax = '-N'+str(args.NLmax) if args.NLmax is not None else ''
    output = args.gnds_file+'-with-'+args.newExcitations+NLmax+'.xml'
    print('Write',output)
    gnds.saveAllToFile( output , covarianceDir = '.' )