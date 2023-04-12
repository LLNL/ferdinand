#! /usr/bin/env python3

# <<BEGIN-copyright>>
# <<END-copyright>>

import os,numpy,sys
import argparse

from fudge import GNDS_formatVersion as formatVersionModule
from fudge import GNDS_file as GNDSTypeModule
from zeroReaction import *

from fudge import reactionSuite as reactionSuiteModule
from fudge import styles as stylesModule
from fudge import physicalQuantity as physicalQuantityModule
from fudge.reactions import reaction as reactionModule
from fudge.reactionData import crossSection as crossSectionModule

import fudge.resonances.resonances as resonancesModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
import fudge.resonances.resolved as resolvedResonanceModule
import fudge.resonances.common as commonResonanceModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
# from fudge.covariances import enums as covarianceEnumsModule
# import fudge.covariances.modelParameters as covarianceModelParametersModule
# import fudge.covariances.covarianceSection as covarianceSectionModule
# import fudge.covariances.covarianceSuite as covarianceSuiteModule

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
import xData.axes as axesModule
import xData.constant as constantModule
import xData.link as linkModule
import xData.xDataArray as arrayModule
from xData.Documentation import computerCode as computerCodeModule

extensionDefault = '-x.xml'
debug = False
styleName = 'eval'

description1 = """Read a GNDS file into Fudge, then write back to the GNDS/xml format with projectile and target exchanged in one partition.  
"""

# Still to do:
#    Copy complete reaction from old form, not just make a new zeroReaction
#
#
__doc__ = description1

parser = argparse.ArgumentParser( description1 )
parser.add_argument( 'input',                                                           help = 'GNDS and/or PoPs file to translate.' )
parser.add_argument( 'output', nargs = '?', default = None,                             help = 'The name of the output file.' )
parser.add_argument( '--energyUnit', type = str, default = 'MeV',                        help = 'Convert all energies in the gnds file to this unit.' )
parser.add_argument( '-e', '--extension', default = extensionDefault,                   help = 'The file extension to add to the output file. Default = "%s"' % extensionDefault )
parser.add_argument( '-x', '--exchange', type = str, default = '',                    help = 'channel in which to exchange nuclei. Spaces optional.' )
parser.add_argument( '--formatVersion', default = formatVersionModule.default, choices = formatVersionModule.allowed,
                                                                                        help = 'Specifies the GNDS format for the outputted file.  Default = "%s".' % formatVersionModule.default )

args = parser.parse_args( )

fileName = args.input
gnds = GNDSTypeModule.read( fileName )

output = args.output
extension = args.extension

PoPs = gnds.PoPs
p = gnds.projectile
t = gnds.target
projectile = gnds.PoPs[p]
target     = gnds.PoPs[t]
elasticChannel = '%s + %s' % (gnds.projectile,gnds.target)
if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
if hasattr(target, 'nucleus'):     target = target.nucleus
pMass = projectile.getMass('amu');   
tMass =     target.getMass('amu')
cm2lab = (pMass + tMass) / tMass 
 
rrr = gnds.resonances.resolved

RMatrix = rrr.evaluated
Emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs(args.energyUnit)
Emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs(args.energyUnit)
IFG = RMatrix.reducedWidthAmplitudes
BC = RMatrix.boundaryCondition
BV = RMatrix.boundaryConditionValue
KRL = RMatrix.relativisticKinematics

exchanged = None
info = {}
energies_o2n = 1.0
widths_o2n = 1.0
renameProtare = False
for partition in RMatrix.resonanceReactions:
    label = partition.label
    if label.replace(' ','') == args.exchange.replace(' ',''):
        exchanged = partition
        rreac_old = label
        resid_new = partition.ejectile 
        eject_new = partition.residual
        rreac_new = "%s + %s" %(eject_new,resid_new)
        ejectile = gnds.PoPs[eject_new]
        residual = gnds.PoPs[resid_new]
        info[label] = (ejectile,residual,rreac_new)
        if hasattr(ejectile, 'nucleus'): ejectile = ejectile.nucleus
        if hasattr(residual, 'nucleus'): residual = residual.nucleus
        Isum = ejectile.spin[0].float('hbar') + residual.spin[0].value
        
        if rreac_old == elasticChannel:
            print('Flip elastic channel. Change definition of the protare "%s" to "%s"' % (rreac_old,rreac_new))
            renameProtare = True
            energies_o2n = tMass/pMass
            cm2lab *= energies_o2n
            widths_o2n = energies_o2n
            if IFG: widths_o2n = widths_o2n**0.5
            t,p = p,t
        info[label] = (renameProtare,) + info[label] + (Isum,)
    else:
        eject = partition.ejectile
        resid = partition.residual
        ejectile = gnds.PoPs[eject]
        residual = gnds.PoPs[resid]
        info[label] = (renameProtare,ejectile,residual,label)
        if hasattr(ejectile, 'nucleus'): ejectile = ejectile.nucleus
        if hasattr(residual, 'nucleus'): residual = residual.nucleus     
        Isum = ejectile.spin[0].float('hbar') + residual.spin[0].value
        info[label] += (Isum,)
        print('  Reaction',label,'unchanged')

if exchanged is None: 
    print('No channel found to match',args.exchange)
elif not renameProtare:
    print('Flip non-elastic channel',exchanged.label)

domain = stylesModule.ProjectileEnergyDomain(Emin*energies_o2n,Emax*energies_o2n,args.energyUnit)
style = stylesModule.Evaluated(styleName, '', physicalQuantityModule.Temperature( 0, 'K' ), domain, 
                               'flipped '+args.exchange+' in '+fileName , '0.1.0' )
    
gnds_new = reactionSuiteModule.ReactionSuite( p, t, gnds.evaluation, PoPs = PoPs, style=style, interaction=gnds.interaction)
resonanceReactions = commonResonanceModule.ResonanceReactions()

print('\nChannels:')
channels = {}
pair = 0
damping = 0
for partition in RMatrix.resonanceReactions:
    kp = partition.label
    if partition.eliminated: damping = 1 

    channels[pair] = kp
    reaction_old = partition.link.link
    renameProtare,ejectile,residual,rreac_new,Isum = info[kp]
    
    pair += 1
    print('  reaction "%s"' % kp,' (eliminated)' if partition.eliminated else '')
    if partition.Q is not None:
        Q = partition.Q.getConstantAs(args.energyUnit)
    else:
        Q = reaction_old.getQ(args.energyUnit)
    if Q < 0:  
        emi = -Q*cm2lab
    else:
        emi = Emin*energies_o2n 
    reaction = zeroReaction(rreac_new,reaction_old.ENDF_MT, Q, [ejectile,residual], None,
                            emi,Emax*energies_o2n,args.energyUnit, debug)
    gnds_new.reactions.add(reaction); print('  Add',reaction.label)
    link = linkModule.Link(reaction)
    rreac = commonResonanceModule.ResonanceReaction ( label=rreac_new, link=link, ejectile=ejectile.id, 
                    Q=None, eliminated=partition.eliminated, scatteringRadius = partition.scatteringRadius )
    reaction.updateLabel( )
    resonanceReactions.add(rreac)
    
print('\nPoles:')
spinGroups = resolvedResonanceModule.SpinGroups()

jset = 0
for Jpi in RMatrix.spinGroups:
    R = Jpi.resonanceParameters.table
    rows = R.nRows
    cols = R.nColumns
    parity = '+' if int(Jpi.parity) > 0 else '-'
    print('  J,pi =%5.1f%s, channels %3i, poles %3i : #%i' % (Jpi.spin,parity,cols,rows,jset) )
    E_poles = R.data   # lab MeV
    
    columnHeaders = [ tableModule.ColumnHeader(0, name="energy", unit=args.energyUnit) ]
    channels_new = resolvedResonanceModule.Channels()

    
    Lvals = []; Svals = []; flip = []
    channelNames = []; chidx = 0
    for ch in Jpi.channels:
        rreac = RMatrix.resonanceReactions[ch.resonanceReaction]
        renameProtare,ejectile,residual,rreac_new,Isum = info[rreac.label]

        if not rreac.eliminated :
            Lvals.append(ch.L)
            Svals.append(ch.channelSpin)
            if rreac.label == rreac_old:
                flip.append(True)
            else:
                flip.append(False)
        
        channelName = "%s width" % rreac_new
        jdx = 2
        while True:
            if channelName not in channelNames:
                channelNames.append( channelName ); break
            channelName = '%s width_%d' % (rreac_new, jdx)
            jdx += 1

        columnHeaders.append( tableModule.ColumnHeader(chidx+1, name=channelName, unit= args.energyUnit) )
        channels_new.add( resolvedResonanceModule.Channel(str(chidx+1), rreac_new, columnIndex=chidx+1, 
                L=ch.L, channelSpin=ch.channelSpin ))
        chidx += 1
                            
    for n in range(rows):
        E = R.data[n][0] * energies_o2n
        D = (R.data[n][damping] if damping==1 else 0) * energies_o2n

        for c in range(cols-1-damping):
           if flip[c]:
               phase =  (-1)**(Lvals[c] + Svals[c] - Isum) #; phase=1
           else:
               phase = 1.0
           R.data[n][1+damping+c] *= widths_o2n * phase
        R.data[n][0] = E
        if damping==1: R.data[n][damping] = D
        
    table = tableModule.Table( columns=columnHeaders, data=R )
    spinGroups.add(    resolvedResonanceModule.SpinGroup(str(jset), Jpi.spin, Jpi.parity, channels_new,
                       resolvedResonanceModule.ResonanceParameters(table)) )

    jset += 1

RMatrix_new = resolvedResonanceModule.RMatrix( styleName, RMatrix.approximation, resonanceReactions, spinGroups, 
              boundaryCondition=BC, relativisticKinematics=KRL, reducedWidthAmplitudes=bool(IFG), 
              supportsAngularReconstruction=True, calculateChannelRadius=False )

resolved_new = resolvedResonanceModule.Resolved( Emin,Emax,args.energyUnit)
resolved_new.add( RMatrix_new )

unresolved = None
resonances_new = resonancesModule.Resonances( gnds.resonances.scatteringRadius, None, resolved_new, unresolved )
gnds_new.resonances = resonances_new

docs = RMatrix.documentation
docnew = RMatrix_new.documentation
for computerCode in docs.computerCodes:
    docnew.computerCodes.add( computerCode ) 
    
if( output is None ) :
    output = fileName + '=' + args.exchange + extension

gnds_new.saveToFile( output, formatVersion = args.formatVersion )

