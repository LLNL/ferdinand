#! /usr/bin/env python3


##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

# <<BEGIN-copyright>>
# <<END-copyright>>

# This script takes a gnd/XML file, reads it in and rewrites it to a file with the same name 
# (only in the currently directory) with the extension '.g2g' added.
# Various channel filters are applied, and data can be transformed to/from amplitudes
# This code is included in Ferdinand.

# TO DO
# Use ch.getScatteringRadius and ch.getHardSphereRadius everywhere, including making gndnew.
#

import numpy
from numpy.matlib import zeros
import fractions
import time,sys

from fudge import reactionSuite as reactionSuiteModule
import fudge.styles as stylesModule
from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode as computerCodeModule
from fudge import outputChannel      as channelsModule
import masses

import fudge.resonances.resonances as resonancesModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule
import fudge.resonances.resolved as resolvedResonanceModule
import fudge.resonances.common as commonResonanceModule
import fudge.resonances.scatteringRadius as scatteringRadiusModule

from PoPs import database as databaseModule
from PoPs import alias as aliasModule
from PoPs import misc as miscModule
from PoPs import IDs as IDsModule
from PoPs.quantities import quantity as quantityModule
from PoPs.quantities import mass as massModule
from PoPs.quantities import spin as spinModule

from PoPs.families import gaugeBoson as gaugeBosonModule
from PoPs.families import lepton as leptonModule
from PoPs.families import baryon as baryonModule
from PoPs.families import nucleus as nucleusModule
from PoPs.families import nuclide as nuclideModule

from PoPs.chemicalElements import isotope as isotopeModule
from PoPs.chemicalElements import chemicalElement as chemicalElementModule
from PoPs.chemicalElements.misc import *
from xData import text as textModule


from pqu import PQU as PQUModule
from xData import table as tableModule
import xData.constant as constantModule
import xData.link as linkModule
#from fudge.processing.resonances.getCoulombWavefunctions import *
from getCoulomb import *
from zeroReaction import *
from BruneTransformation import *
from BarkerTransformation import *

fmscal = 0.0478450
etacns = 0.1574855
amu    = 931.494013
oneHalf = fractions.Fraction( '1/2' )
one = fractions.Fraction( '1' )
zero = fractions.Fraction( '0' )
spinUnit = spinModule.baseUnit

        
Negative = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum
Given    = resolvedResonanceModule.BoundaryCondition.Given
Eliminate= resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction
Brune    = resolvedResonanceModule.BoundaryCondition.Brune
Standards= [Negative,Given]
Outputs  = [Negative,Given,Brune]
    

def gndTransform (gnd,nocm, Elastic,nogamma,noreac,filter,amplitudes,Gammas, Adjust,File,ReichMoore,noReichMoore,Qel_nzero, bndnew,p6, verbose,debug):

    # print('amplitudes:',amplitudes,Gammas,'Qel_nzero:',Qel_nzero)
    if Gammas: amplitudes=False
    #if not (nogamma or noreac or Gammas or amplitudes or filter or bndnew!=None or ReichMoore): return gnd
    if debug: print("\ngndTransform:" ,nocm, Elastic,nogamma,noreac,filter,amplitudes,Gammas,ReichMoore,noReichMoore)

    recons = gnd.styles.findInstancesOfClassInChildren(stylesModule.CrossSectionReconstructed)
    if len(recons) > 0:
        if len(recons) > 1: raise Exception('ERROR: protare with more than one reconstructed cross section style not supported.')
        gnd.removeStyle(recons[0].label)
        print('Removing %s style data' % recons[0].label)

    evaluation = gnd.evaluation
    evaluation += ' mod'
    resonances = gnd.resonances
    rrr = resonances.resolved
    Rm_Radius = resonances.getScatteringRadius()
    Rm_global = Rm_Radius.getValueAs('fm')
 
    RMatrix = rrr.evaluated
    IFG = RMatrix.reducedWidthAmplitudes      #  This is meaning of widths WITHIN given gnd evaluation
    KRL = RMatrix.relativisticKinematics
    approximation_new = RMatrix.approximation
    if ReichMoore is not None: approximation_new = 'Reich_Moore'
    RWA_OUT = not Gammas and (amplitudes or IFG)   # Want results in rwa
    if nogamma or noReichMoore:
        approximation_new = 'Full R-Matrix'
        if ReichMoore and nogamma: print("***** -g overrides -R")
        ReichMoore = None
        
    haveOverrides = False
    for Jpi in RMatrix.spinGroups:
        for ch in Jpi.channels:
            if ch.boundaryConditionValue is not None: haveOverrides = True
 
    PoPs_in = gnd.PoPs

    proj,targ = gnd.projectile,gnd.target  # initial elastic channel
    pold,told = (proj,targ)
    elasticChannel = '%s + %s' % (pold,told)
    if verbose: print("Elastic channel: ",elasticChannel) 
    elasticOld = elasticChannel
    for pair in RMatrix.resonanceReactions: 
        if debug: print("channel ",pair.label,pair.label.split('_e'))
        if elasticOld is None:
            if pair.label == elasticChannel:                elasticOld = pair.label
            if pair.label.split('_e')[0] == elasticChannel: elasticOld = pair.label
            if pair.label                == elasticChannel: elasticOld = pair.label
    if verbose: print("Elastic channel: ",elasticOld) 
    evalStyle = gnd.styles['eval']

# start newer evaluation

    BC_old = RMatrix.boundaryCondition
    BV_old = RMatrix.boundaryConditionValue   
    if BC_old == 'S': BC_old = Eliminate
    if BC_old == '-L':BC_old = Negative

    BC_req = BC_old if bndnew is None else Negative if 'L' in bndnew else Brune if 'Brune' in bndnew else bndnew  # parsing the -b input option (bndnew)
    try:  
        BV_new = float(BC_req)   # input a float? # parsing the -b input option
        BC_new = Given  # only if float() succeeds!
    except:
        BV_new = BV_old  # (could be None if BC_new is not Given)
        BC_new = BC_req  # otherwise BC_in (from -b) was a string
    if BC_new not in Outputs and BC_new != Eliminate:
        print('ERROR: Output boundary condition',BC_new,'is not in',Outputs) # no transformations to Eliminate !
        sys.exit()

    standard_in   = BC_old in Standards   # or haveOverrides  # input  is standard (Lane and Thomas) R-matrix (-L or float: not 'Brune' or 'S')
    standard_out  = BC_new in Standards                       # output is standard (Lane and Thomas) R-matrix (-L or float)
    
    if BC_old == BC_new and BV_old == BV_new:  # and not haveOverrides: 
         standard_in   = False # no need for intermediate Brune basis at all, if in=out boundary conditions exactly!!
         standard_out  = False
    transformed = standard_in or standard_out  # either Barker or Brune transformations needed, so calculate S, P infrastructure, etc.
    changeText = "Changing boundary condition from %s,%s to %s,%s \n   by std-in,std-out: %s %s ( Overrides=%s)" % (BC_old,BV_old,BC_new,BV_new,standard_in,standard_out,haveOverrides)
    print(changeText)

    if Elastic != None:
        pnew = RMatrix.resonanceReactions[Elastic].ejectile
        tnew = RMatrix.resonanceReactions[Elastic].residual
        if verbose: print('ELASTIC:',rrr.evaluated.resonanceReactions[Elastic].label,pnew,tnew,' from ',Elastic)
        proj,targ = pnew,tnew
        elasticChannel = '%s + %s' % (proj,targ)  # revised
        elasticNew = Elastic
    else:
        pnew,tnew = (proj,targ)
        elasticNew = elasticOld
    if verbose: print('Elastic: old & new = ',elasticOld,' & ',elasticNew)
    if Qel_nzero:
        Q_offset = 0.0
    else:
        for pair in RMatrix.resonanceReactions:
            if pair.label==elasticChannel:
                if pair.Q is not None:
                    Q_offset = pair.Q.getConstantAs('MeV')
                else:
                    reaction = pair.link.link
                    Q_offset = reaction.getQ('MeV')
    if verbose: print('Q values shifted by Q_offset =',Q_offset)

    if File!=None: 
        dataFile = open (File,'r')    # reading
        print("dataFile ",dataFile," opened for reading")

    if verbose or Adjust is not None or File is not None:
        traceFileName = 'R-matrix-parameters.out'
        traceFile = open (traceFileName,'w')
    else:
        traceFileName = None
# initialise
    resonanceReactionsNew = commonResonanceModule.ResonanceReactions()

# Add inclusive Gamma channel as first particle pair, but zero widths
    if ReichMoore is not None:
        level = 0
        projectile = PoPs_in[pnew];
        target     = PoPs_in[tnew];
        pMass = projectile.getMass('amu');
        tMass = target.getMass('amu')
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus
        pZ = projectile.charge[0].value; tZ =  target.charge[0].value
        
        compoundZ = pZ + tZ
        compoundA = int(pMass + tMass + 0.5)
        cZA = compoundZ*1000 + compoundA
        
        cMass = masses.getMassFromZA( cZA )
        compoundName = idFromZAndA(compoundZ,compoundA)
        cLevelName = nuclideIDFromIsotopeSymbolAndIndex(compoundName,level)

        gchannelName = '%s + photon' % cLevelName   # compoundName
        Q = (pMass + tMass - cMass)*amu - Q_offset
        rr = gchannelName
        print("Reich-Moore particle pair: ",gchannelName,' with CN mass %.5f so Q=%.3f, label=%s' % (cMass,Q,rr))

#       gData = { '0' : [ 0.0,       .0,           1, None,    1,     +1 ] }
        gammaParticle =  miscModule.buildParticleFromRawData( gaugeBosonModule.Particle, 'photon',
            mass = ( 0, 'amu' ), spin = ( one, spinUnit ),  parity = ( 1, '' ), charge = ( 0, 'e' ))
        PoPs_in.add(gammaParticle)

        nucleus = miscModule.buildParticleFromRawData( nucleusModule.Particle, cLevelName, index = level, energy = ( 0.0, 'eV' ) ,
                                                       spin=(zero,spinUnit), parity=(1,''), charge=(compoundZ,'e') )
        compoundParticle = miscModule.buildParticleFromRawData( nuclideModule.Particle, cLevelName, nucleus = nucleus, mass=(cMass,'amu') )
        PoPs_in.add(compoundParticle)
        if verbose and debug: print(PoPs_in.toXML())
        
 # Create background ReichMoore cross section (zero to start with)
        MT_capture = 102
        label = 'ReichMoore capture'
        capture = zeroReaction(label,MT_capture, Q, [gammaParticle,compoundParticle], 'damping', rrr.domainMin,rrr.domainMax,rrr.domainUnit, debug)
        link = linkModule.Link(capture)
        
        resonanceReactionsNew.add(commonResonanceModule.ResonanceReaction( label=rr, link=link, ejectile='photon', Q=None, eliminated=True) )

# filters
    neither = nogamma or noreac
    g_here = False; changed = False
    pairnew = { }
    redmass = {};  lab2cm = {}; prmax = {}; ZZ = {}; Qval={}
    Rmax = 0.0
##### reaction channels
    for pair in RMatrix.resonanceReactions:
        rr = pair.label
        eliminated = pair.eliminated
        reaction = pair.link.link
        channelName = pair.label
        if verbose:  print("rr,channelName,Q =",rr,channelName,pair.Q,'for reaction',reaction.label)
        if  not pair.isFission():
            p,t = pair.ejectile,pair.residual
        else:
            p,t = ('fission','fission_products')
        if verbose: print('p,t =',p,t)
        projectile = PoPs_in[p];
        target     = PoPs_in[t];
        pMass = projectile.getMass('amu');
        if not eliminated: tMass = target.getMass('amu')
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus
        #if verbose: print 'For nuclides:',projectile,target
        
        pZ = projectile.charge[0].value; tZ =  target.charge[0].value
# dyanamical parameters:
        if pair.getScatteringRadius() is not None:
            RM =  pair.getScatteringRadius().getValueAs('fm')
        else:
            RM = Rm_global
        Rmax = max(Rmax,RM)

        if not eliminated:
            redmass[rr] = pMass*tMass/(pMass+tMass)
            lab2cm[rr]  = tMass/(pMass+tMass)
        else:
            redmass[rr] = 0.0
            lab2cm[rr] = 1.0
        prmax[rr] = RM
        ZZ[rr] = pZ*tZ
        pairQ = pair.Q
        if pairQ is not None:
            Qval[rr] = pairQ.getConstantAs('MeV')
        else:
            Qval[rr] = reaction.getQ('MeV')
        if Q_offset:
            Qval[rr] -= Q_offset
            newQ = channelsModule.QModule.Constant1d( Qval[rr], domainMin=rrr.domainMin,
                    domainMax=rrr.domainMax, axes=axesModule.Axes(labelsUnits={1: ('energy_in', rrr.domainUnit), 0: ('energy', 'MeV')}) ,
                    label = 'eval' )
            pairQ = channelsModule.QModule.component()
            pairQ.add( newQ )
        g = 'photon' in p or 'gam' in p
        g_here = g_here or g

        # include = channelName==elasticChannel or not (g or noreac) or (g and not neither)
        include = True
        if nogamma and g: include = False
        if noreac and not g and not channelName==elasticChannel: include = False
        if noReichMoore and 'damping' in rr: include = False

        if filter != None: include = rr in filter
        if noReichMoore or debug: 
            if include:
                print(" Include pair ",rr,":",include,' (eliminated)' if eliminated else '')
            else:
                print(" Exclude pair ",rr,":",include,' (eliminated)' if eliminated else '')
        if include:
            pout = '%s + %s' % (p,t)
            if 'damping' in rr: pout += ' [damping]'
            pairnew[rr] = pout  # index from old channel forward to new channel
            MT = reaction.ENDF_MT
            link = pair.link

            resonanceReactionsNew.add(
                 commonResonanceModule.ResonanceReaction( label=pout, link=link, ejectile=p, Q=pairQ, eliminated=eliminated  ) )

            if pair.getScatteringRadius() is not None and pair.getScatteringRadius().getValueAs('fm') != Rm_global:
                resonanceReactionsNew[pout].scatteringRadius = commonResonanceModule.ScatteringRadius( 
                    constantModule.Constant1d(RM, domainMin=rrr.domainMin, domainMax=rrr.domainMax,
                        axes=axesModule.Axes(labelsUnits={1: ('energy_in', rrr.domainUnit), 0: ('radius', 'fm')})) )
        else:
            changed = True
            if True: print ('Channel',p,'excluded')
                
    if debug: print("redmass:",redmass,"\nprmax:",prmax,"\nZZ:",ZZ,"\nQval:",Qval,"\nlab2cm:",lab2cm)
    if changed and debug:
        print("Mapping to new pairs:",pairnew)
    if debug: open( 'PoPs.in' , mode='w' ).writelines( line+'\n' for line in PoPs_in.toXML_strList( ) )

    gndnew = reactionSuiteModule.ReactionSuite( proj,targ, evaluation, style=evalStyle, PoPs = PoPs_in, interaction='nuclear')
    
    #if debug: open( 'PoPs.out' , mode='w' ).writelines( line+'\n' for line in gndnew.PoPs.toXML_strList( ) )

    if ReichMoore is not None and not noReichMoore: gndnew.reactions.add(capture)
    energy_unitsf =  'MeV'
    width_unitsf =  energy_unitsf     # this is for reading initial GNDS only: change e.g eV to MeV
    width_unitsfo =  energy_unitsf + ( '**(1/2)' if RWA_OUT else '')    # this is for final GNDS, after any rwa conversions
    print('width_unitsf:',width_unitsf,'width_unitsfo:',width_unitsfo)

    lab2cm_in  = 1.0 if nocm else lab2cm[elasticOld]
    lab2cm_new = 1.0 if nocm else lab2cm[elasticNew]

    spinGroupsNew = resolvedResonanceModule.SpinGroups()
    spinGroupIndex = 0
    for Jpi in RMatrix.spinGroups:
        jtot = Jpi.spin
        parity = Jpi.parity
        if verbose: print("\nT: For J,pi =",jtot,parity)
        channelsNew = resolvedResonanceModule.Channels()

        R = Jpi.resonanceParameters.table
        #widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
        poleEnergies = R.getColumn('energy',energy_unitsf)  # in LAB frame !
        rows = len(poleEnergies)
        #cols = len(widths) + 1  # +1 for energy
        cols = len(Jpi.channels) + 1  # +1 for energy
        if debug: print(" R matrix",jtot,parity," width table with rc =",rows,cols)
        idx = 0

        columnHeaders = [ tableModule.ColumnHeader(idx, name="energy", unit=energy_unitsf) ]
        channelNames = []
        energy_unitsi =  R.columns[0].unit
        energy_scale = PQUModule.PQU( 1.0 ,energy_unitsi).unit.conversionFactorTo(energy_unitsf)
        width_scale = {}

        colsto  = {}; colsfrom = {}; pair_of_col = {}
        colsnew = 1  # include energy always first, so this 1 is the first channel
        colsto[0] = 0 # energy  is col 0, channels=1,2,3,4..

        if ReichMoore is not None:
            columnHeaders.append( tableModule.ColumnHeader(1, name=gchannelName + ' width', unit= width_unitsf) )
            Sch = resolvedResonanceModule.Spin( 0.0 )
            channelsNew.add( resolvedResonanceModule.Channel('1', gchannelName, columnIndex=1, L=0, channelSpin=Sch) )
            idx += 1
            colsnew += 1
            if debug: print('Channel ','RM', 'is new',gchannelName,' -- ',''+str(idx), '.   Units',width_unitsf)


        BVi_old= []
        for ch in Jpi.channels:
            found = False
            n = ch.columnIndex
            rr = ch.resonanceReaction
            pair = RMatrix.resonanceReactions[rr]
            lch = ch.L
            sch = ch.channelSpin
            BV = ch.boundaryConditionValue
            #if standard_in: BV = None   # Make Overrides disappear after transform to Brune basis
            #            channelName = pair.label
            pair_of_col[n-1] = rr
            ppnew = pairnew.get(rr)
            if ppnew != None:  # included!
                idx += 1
                found = True
                colsto[n] = colsnew
                colsfrom[colsnew] = n  # Exclude e column here
                #if debug: print " Mapping ",n," to ",colsnew
                BVi_old.append(BV) # to be indexed by colsnew
                # BV = None  # boundaryConditionValue only used to get out of old basis, NOT for any new basis

                channelName = "%s width" % pair.label
                jdx = 2
                while True:
                    if channelName not in channelNames:
                        channelNames.append( channelName ); break
                    channelName = '%s width_%d' % (pair.label, jdx)
                    jdx += 1

                width_unitsi =  R.columns[n].unit.replace('**(1/2)','')   # just get energy unit for first conversion
                width_scale[idx]  = PQUModule.PQU( 1.0 ,width_unitsi).unit.conversionFactorTo(width_unitsf) * lab2cm_in / lab2cm_new # no **(1/2) here!
                if IFG: width_scale[idx] = width_scale[idx] ** 0.5
                
                #if debug: print " col ",channelName," has BND =",bndnew_col,' from ',transform,bnd,' with initial ',bndi
                if debug: print('Channel ',rr, 'is new',ppnew,' -- ',''+str(idx), '.   Units',width_unitsi,' to ',width_unitsf, '(',width_scale[idx],energy_scale,')')
                
                columnHeaders.append( tableModule.ColumnHeader(idx, name=channelName, unit= width_unitsfo) )
                channelsNew.add( resolvedResonanceModule.Channel(''+str(idx), ppnew, columnIndex=idx, L=lch, channelSpin=sch, boundaryConditionValue=BV) )
            else:
                if debug: print('Channel ',rr,' excluded')
            if found: colsnew += 1

        if verbose:
            print(" for J,pi =",jtot,parity,' there are now ',rows,' rows and ',colsnew,' columns (before:',cols,')')
            print("  Mapping to new cols:",colsto)
            print("  Mapping to old cols:",colsfrom)
            print("  Mapping to from old col to old particle pair:",pair_of_col)
            print("  width_scale:", width_scale)
            print("  lab2cm, in & new:",lab2cm_in,lab2cm_new)
            print("  BVi_old:", BVi_old)
            print("  channelsNew:")
            for ch in channelsNew:
                print(ch.columnIndex,': ',ch.resonanceReaction)

        if colsnew < 2:   # no channels!
            print('Spin group',jtot,parity,'empty of channels: exclude')
            continue

        if transformed:  # Prepare for Brune transformation of rwa
            NCH = colsnew-1 # first col is for pole energies
            gami = numpy.zeros([rows,NCH])
            if standard_in:   BV_in  = numpy.zeros(NCH)
            if standard_out:  BV_out = numpy.zeros(NCH)
            Shift = numpy.zeros([rows,NCH])
            lchv = numpy.zeros(cols)
        #if transformed or Adjust!=None:  Ei = numpy.zeros(rows);
        ncols_new = cols + 1 if ReichMoore is not None else cols
        P = numpy.zeros([rows,ncols_new])
       
        if File!=None:   # read in R-matrix data from 'dataFile'
             for i in range(rows):
                 line = dataFile.readline()
                 data = line.split(',')
                 if verbose: print("Data read:",data)
                 lr = int(data[0])
                 for n in range(lr):  R[i][n] = float(data[n+1])  #widths[n-1][i] =

# First thing: transform to new projectile energy basis if needed
        Rnew = [[0. for c in range(colsnew)] for i in range(rows)]
        if debug: print("New R array of R*C =",rows,colsnew,'from',colsfrom,'with width_scale=',width_scale)
        for i in range(rows):
            Rnew[i][0] = (poleEnergies[i] * lab2cm_in + Q_offset + Qval[elasticNew]-Qval[elasticOld]) / lab2cm_new
            for new in range(1,colsnew):   # 0 is first column of widths
                if new in colsfrom.keys():  # check not new RM channel
                    Rnew[i][new] = R[i][colsfrom[new]] * width_scale[new]   # convert to MeV or MeV**0.5. Including lab2cm factor
                else:
                    Rnew[i][new] = ReichMoore

        if not IFG or transformed or amplitudes:
            if debug: print("Calculate P,S and dSdE")
            poleCME = [0. for i in range(rows)]
            poleLABE = [0. for i in range(rows)]
            ES = [0. for i in range(rows)]
            if verbose: print("J,pi =",jtot,parity,'poleEnergies: ',poleEnergies,' lab')
            for i in range(rows):
                poleCME[i] = poleEnergies[i] * lab2cm_in   + Q_offset #  MeV
                ES[i] = poleCME[i]/lab2cm_new
                ecm = poleCME[i]

                for ch in channelsNew:
                    n = ch.columnIndex 
                    RM = n not in colsfrom.keys()

                    if RM and n==1:   # photon
                        lch = 0
                        pen = 1.0
                        shift = 0.
                    else:
                        lch = ch.L
                        no = colsfrom[n]-1 # start from no=0
                        ppo = pair_of_col[no]
                        BV = BVi_old[no-1]   # Override (or None) in old basis
                        e_ch = ecm + Qval[ppo] 
                        if debug and i==0: print(n,' channel ',ch.resonanceReaction, "n,no,ppo =",n,no,ppo)
                    
                        pen,shift,derS = getCoulombp_PSdS(e_ch,lch, prmax[ppo], redmass[ppo],ZZ[ppo], False)
                    #if debug: print n,no,ppo," P (",ecm,e_ch,lch, prmax[ppo], redmass[ppo],ZZ[ppo],') =',pen,shift
                    P[i,n] = float(pen)

                    w = Rnew[i][n]
                    #if debug: print  IFG , amplitudes , transformed,':',not IFG and (amplitudes or transformed)
                    if not IFG and (transformed or amplitudes):              # translate Gamma(f) to rwa: stored width = 2 * rwa**2 * P
                        #if R.columns[no+1].unit == 'MeV': w=w*1e6
                        if abs(w)>1e-20: 
                            #print  "w,P =",w,P[i,n]
                            rwa = abs(w/(2 * P[i,n]))**0.5
                            if w<0: rwa = -rwa
                            if debug: print('row',i,'col',n,' Transform to rwa =',rwa,' via P=',P[i,n],'from ',w)
                        else:
                            rwa = 0.0
                        Rnew[i][n] = float(rwa) 
                    if transformed:
                        lchv[n-1] = lch
                        Shift[i,n-1] = shift
                        gami[i,n-1] = Rnew[i][n]  # convert to MeV**{0.5} for Brune
                        if ecm < 0 and verbose: print('Bound state ecm=',e_ch,'L=',lch,' Shift [',i,n-1,'] =',shift)

                    if standard_in :
                        if   BC_old == Negative:
                            BV_in[n-1] = -lch
                        elif BC_old == Given:
                            BV_in[n-1] =  BV_old
                        elif BC_old == Eliminate or BC_old == Brune:
                            print(" Unexpected input b.c.",BC_old,BV_old,BV)
                            raise SystemExit
                        else:
                            print(" Boundary C,V:",BC_old,BV_old," not recognized in gndtransform.")
                            raise SystemExit
                        #if verbose: print 'For transform to Brune param: BV_in [',n-1,'] =',BV_in[n-1]
                        
        RWA = transformed or amplitudes or IFG   # when Rnew now has rwa

        if debug: print('standard_in,Adjust,standard_out =',standard_in,Adjust,standard_out)
        if standard_in:

            gamf,Ef = BarkerTransformation ( gami, ES, BV_in, lchv, colsfrom,pair_of_col, Qval,prmax,redmass,ZZ,lab2cm_new, jtot,parity, debug)

            for i in range(rows):
                Rnew[i][0] = Ef[i] 
                for n in range(1,len(Rnew[i])): Rnew[i][n] = gamf[i,n-1] 
            if verbose: print("J,pi =",jtot,parity,'poleEnergies: ',[Rnew[i][0] for i in range(rows)],' lab after Barker')

        if Adjust!=None:  # adjust pole energies in LAB frame
            for i in range(rows):
                 E = R[i][0]
                 Rnew[i][0] = eval(Adjust)
                 if verbose and abs(E-Rnew[i][0])>1e-10: print(" Pole energy %f adjusted to %f" % (E,Rnew[i][0]))
    
        if traceFileName is not None:
            #print >>traceFileName, "%f %f :   Previous pole energy, increment " % (E, Rnew[i][0]-E)
            for i in range(rows):
                lr = len(R[i])
                print(str(lr)+',', ', '.join(map(repr,R[i])), file=traceFile)

        if standard_out:
            for i in range(rows):
                poleLABE[i] = Rnew[i][0] # LAB frame
                poleCME[i] = Rnew[i][0] * lab2cm_new  # LAB-to-CM
                if transformed:   # recalculate S and P for the new pole energies from BarkerTransformation, and set gami
                    ecm = poleCME[i]
                    for ch in channelsNew:
                        n = ch.columnIndex
                        RM = n not in colsfrom.keys()

                        if RM and n==1:   # photon
                            lch = 0
                            pen = 1.0
                            shift = 0.
                        else:
                            lch = ch.L
                            no = colsfrom[n]-1 # start from no=0
                            ppo = pair_of_col[no]
                            e_ch = ecm + Qval[ppo] 
                            pen,shift,derS = getCoulombp_PSdS(e_ch,lch, prmax[ppo], redmass[ppo],ZZ[ppo], False)
                        #if debug: print n,no,ppo," P (",ecm,e_ch,lch, prmax[ppo], redmass[ppo],ZZ[ppo],') =',pen,shift
                        Shift[i,n-1] = shift
                        gami[i,n-1] = Rnew[i][n]  # convert to MeV**{0.5} for Brune

                for ch in channelsNew:
                    n = ch.columnIndex
                    lch = ch.L
                    if BC_new == Negative:
                        BV_out[n-1] = -lch
                    elif BC_new == Given:
                        BV_out[n-1] =  BV_new
                    elif BC_new == Eliminate or BC_new == Brune:
                        print(" Unexpected output b.c.",BC_new,BV_new)
                        raise SystemExit
                    if debug: print('For transform to standard param: BV_out [',n-1,'] =',BV_out[n-1])

            gamf,Ef = BruneTransformation ( gami, poleLABE, BV_out, Shift, debug, jtot,parity, lab2cm_new) 

            for i in range(rows):
                Rnew[i][0] = Ef[i] 
                for n in range(1,len(Rnew[i])): Rnew[i][n] = gamf[i,n-1] 
            if verbose: print("J,pi =",jtot,parity,'poleEnergies: ',[Rnew[i][0] for i in range(rows)],' lab after Brune')
    
#   Rnew up to now has rwa if RWA = transformed or IFG or amplitudes 

#       GAMMA = Gammas or not (amplitudes or IFG)      # Want results in formal 
#       RWA_OUT = not Gammas and (amplitudes or IFG)   # Want results in rwa
#  
        if RWA != RWA_OUT : # translate rwa to Gamma(f)   # do this after any Brune  / Barker transforms
            for i in range(rows):
                for ch in channelsNew:
                    n = ch.columnIndex
                    RM = n not in colsfrom.keys()
                                    # have to recalculate P now that energies have changed
                    if RM and n==1:   # photon
                        P[i,n] = 1
                    else:
                        ecm = Rnew[i][0] * lab2cm_new
                        no = colsfrom[n]-1
                        ppo = pair_of_col[no]
                        e_ch = ecm + Qval[ppo]

                        lch = ch.L
                        pen,shift,derS = getCoulombp_PSdS(e_ch,lch, prmax[ppo], redmass[ppo],ZZ[ppo], False)
                        P[i,n] = float(pen)

                    Gammaf = 2.0 * Rnew[i][n]**2 * P[i,n]
                    if Rnew[i][n]<0.: Gammaf = -Gammaf
                    if debug: print('row',i,'col',n,' Transform rwa =',Rnew[i][n],' via P=',P[i,n],' to ',Gammaf)
                    Rnew[i][n] = Gammaf 
                    
#   Tidy up table numbers and width signs:
        for i in range(rows):
            if p6:
                Rnew[i][0] = float( "%.9e" % Rnew[i][0])
                for ch in channelsNew: 
                    Rnew[i][ch.columnIndex] = float( "%.6e" % Rnew[i][ch.columnIndex])
                    if abs(Rnew[i][ch.columnIndex]) < 1e-20: Rnew[i][ch.columnIndex] = 0
                
            change = False
            for ch in channelsNew:
                eliminated = resonanceReactionsNew[ch.resonanceReaction].eliminated
                if eliminated and Rnew[i][ch.columnIndex]<0:  change=True
            if change:
                for ch in channelsNew:
                    Rnew[i][ch.columnIndex] *= -1

        table = tableModule.Table( columns=columnHeaders, data=Rnew )
        spinGroupsNew.add( resolvedResonanceModule.SpinGroup(str(spinGroupIndex), Jpi.spin, parity, channelsNew,
                           commonResonanceModule.ResonanceParameters(table) ) ) 
        spinGroupIndex += 1
        # end Jpi loop
    print('BC_new,BV_new,RWA_OUT:',BC_new,BV_new,RWA_OUT)
    RMatrixnew = resolvedResonanceModule.RMatrix( 'eval', approximation_new, resonanceReactionsNew, spinGroupsNew, 
                                            boundaryCondition=BC_new,  boundaryConditionValue=BV_new, 
                                            relativisticKinematics=KRL,     reducedWidthAmplitudes=RWA_OUT,
                                            supportsAngularReconstruction=True, calculateChannelRadius=False )
                                            
    resolved = resolvedResonanceModule.Resolved( rrr.domainMin,rrr.domainMax,rrr.domainUnit )
    resolved.add( RMatrixnew )
    
    scatteringRadius = Rm_Radius
    unresolved = None
    resonancesnew = resonancesModule.Resonances( scatteringRadius, None, resolved, unresolved )
    gndnew.resonances = resonancesnew

    if (pold,told)==(pnew,tnew):   # Copy the documentation and other reactions if the elastic channel is the same

        docnew = RMatrixnew.documentation
        
        transformLabel = 'transform'
        labels = []
        for computerCode in RMatrix.documentation.computerCodes:
            docnew.computerCodes.add( computerCode ) 
            labels.append( computerCode.label )
#         labels = docnew.computerCodes.labels
        if transformLabel in labels: print('gndstransform: existing labels:',labels)
        
        for i in range(2,100): 
            if transformLabel+str(i) not in labels: break
            
        computerCodeTransform = computerCodeModule.ComputerCode( label = transformLabel+str(i) , name = 'ferdinand', version = '') # , date = time.ctime() )
        docLines = computerCodeTransform.note.body

        modtext = changeText + '\n'
        if Elastic!= None:  modtext += '\nelastic=%s, ' % elastic
        if nogamma:  modtext += '\n  nogamma=%r, ' % nogamma
        if ReichMoore is not None or noReichMoore:  modtext += '\n  ReichMoore=%s, ' % (ReichMoore and  not noReichMoore)
        if noreac:  modtext += '\n  noreac=%r, ' % noreac
        if filter!= None:  modtext += '\n  filter=%s, ' % filter
        if amplitudes:  modtext += '\n  amplitudes=%r, ' % amplitudes
        if Gammas:  modtext += '\n  Gammas=%r, ' % Gammas
        if standard_in:   modtext += '\n  transform from stanard in BC,BV=%s,%s' % (BC_old,BC_old)
        if standard_out:  modtext += '\n  transform to standard out BC,BV=%s,%s' % (BC_new,BV_new)
        if Adjust!= None: modtext += '\n  Pole energies adjusted according to "%s", ' % Adjust
        modtext = "Processed by Ferdinand:\n"+modtext + '\n'
        #print("Modification record:\n",modtext)
        docLines += '\n\n'+ modtext + '\n' + time.ctime()+'\n\n'
        computerCodeTransform.note.body =  docLines
        
        docnew.computerCodes.add( computerCodeTransform ) 
        
        endfDoc = gnd.styles.getEvaluatedStyle().documentation.endfCompatible.body
        if len(endfDoc)>0: gndnew.styles.getEvaluatedStyle().documentation.endfCompatible.body =  endfDoc 

        for reaction in gnd.reactions:
            if Q_offset: 
                Q = reaction.getQ('MeV') - Q_offset
                print("New Q for",reaction.label,"should be",Q)
            if reaction.label not in [r.label for r in gndnew.reactions]:
                if noReichMoore and 'damping' in reaction.label: continue 
                if debug: print('Add',reaction.label,'as not yet')
                gndnew.reactions.add ( reaction )
        for reaction in gnd.orphanProducts :
            gndnew.orphanProducts.add ( reaction )
        for sum in gnd.sums.crossSectionSums:
            gndnew.sums.crossSectionSums.add ( sum )
        for sum in gnd.sums.multiplicitySums:
            gndnew.sums.multiplicitySums.add ( sum )

    else:                   # Just put in zero background cross sections, or something, or nothing
        print("\nConstruction of new non-elastic pointwise cross sections not implemented,")
        print(" as old ",(pold,told),"!= new ",(pnew,tnew))
      

    if traceFileName!=None:  # adjust pole energies
        print(" Energy adjustments saved in file %s " % traceFileName)

    return gndnew

################################# MAIN PROGRAM
if __name__=="__main__":

    import sys, os,argparse
    # Process command line options
    parser = argparse.ArgumentParser(description='Translate R-matrix Evaluations')

    parser.add_argument('inFile', type=str, help='The input file you want to translate.' )
    parser.add_argument("-g", "--nogamma", action="store_true", help="Omit gamma channels")
    parser.add_argument(      "--noreac", action="store_true", help="Omit all nonelastic (reaction) channels")
    parser.add_argument("-f", "--filter", type=str,  help="Filter of csv list of particle-pair-labels to include. Overrides -g,-r options")
    parser.add_argument("-E", "--Elastic", type=str,  help="ResonanceReaction label of elastic particle-pair in output")
    parser.add_argument("-R", "--ReichMoore", type=float, help="Add a Reich-Moore gamma channel at this value")

    parser.add_argument("-a", "--amplitudes", action="store_true", help="Convert intermediate gnd file stores to reduced width amplitudes, not widths. Otherwise do nothing")
    parser.add_argument("-G", "--Gammas", action="store_true", help="Convert intermediate gnd file stores to formal widths, not reduced width amplitudes. Overrides -a. Otherwise do nothing")
    parser.add_argument("-b", "--boundary", type=str, help="Boundary condition in output: 'Brune'; 'S'; 'L' for B=-L; or 'X' for B=float(X).")
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug output (more than verbose)")
    parser.add_argument("-c", "--nocm", action="store_true", help="No incoming transformations of cm-lab pole energies: for old mistaken files")
    parser.add_argument("-A", "--Adjust", type=str, help="Adjust pole energies: give arithmetic function of E, such as 'E+5000 if E>2e6 else E'. Applied before any Brune and after any Barker transformations")
    parser.add_argument("-F", "--File", type=str, help="Data file for reading R-matrix data")

    args = parser.parse_args()
    # Read in
    gnd = reactionSuiteModule.ReactionSuite.readXML_file( args.inFile )

    ## CHANGE:
    gndout = gndTransform(gnd,args.nocm, args.Elastic,args.nogamma,args.noreac,args.filter,args.amplitudes,args.Gammas,
                           args.Energy,args.File,args.ReichMoore,  args.Q, args.boundary, args.verbose,args.debug)
#             gndTransform (gnd,nocm,    elastic,     nogamma,     noreac,     filter,     amplitudes,     Gammas, 
#                          Energy,     File,     ReichMoore,       Qel_nzero, bndnew, verbose,debug):

        
    outFile = os.path.basename( args.inFile ) + '.g2g'

    gndout.saveToFile(outFile)
    print("Written file",outFile)
