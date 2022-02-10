

##############################################
#                                            #
#    Ferdinand 0.40, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

from fudge import reactionSuite as reactionSuiteModule
from fudge import styles as stylesModule
from fudge import physicalQuantity as physicalQuantityModule
from fudge.reactions import reaction as reactionModule
from fudge.reactionData import crossSection as crossSectionModule
from fudge.processing.resonances.getCoulombWavefunctions import *

from xData.Documentation import documentation as documentationModule
from xData.Documentation import computerCode as computerCodeModule

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
from numpy import zeros

from pqu import PQU as PQUModule
from xData import table as tableModule
import xData.constant as constantModule
import xData.link as linkModule
import xData.xDataArray as arrayModule
import xData.axes as axesModule

#from f90nml import read
import f90nml.parser 
from zeroReaction import *
from getCoulomb import *
# import masses
import masses 

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

##############################################  read_sfresco

def read_fresco(inFile, amplitudes,Lvals,CNspec,nonzero,noCov, verbose,debug):

#   print('amplitudes:',amplitudes,' and nonzero:',nonzero)
    srch = open(inFile,'r')
    els = srch.readline().split("'")
    #print 'els=',els
    name_frin = els[1]
    name_frout= els[3]
    if verbose: print('in,out =',name_frin,',',name_frout)
    np,nd = [int(x) for x in srch.readline().split()][0:2]
    print(np,'parameters and',nd,'fitted data sets')
    srch.readline()  # blank line	
    header = srch.readline()  
    srch.readline()  # NAMELIST line

    srch.seek(0)
    namelists = f90nml.read(inFile)
    
    fresco = namelists['fresco']
    partitions = namelists['partition']
    states = namelists['states']
    pots = namelists['pot']
    variables = namelists['variable'][0:np] if np>0 else []
    try:    Covariances = namelists['Cov']
    except: Covariances = None
  
    if nd>0: 
        try: 
            datanml = namelists['data'] #[0:nd]
        except:
            nd = 0
            datanml = None
    rmatch = fresco['rmatch']
    elab = fresco['elab']
    emin,emax = elab[0],elab[1]
    Rm_global = rmatch
    Rm_radius = scatteringRadiusModule.scatteringRadius(
        constantModule.constant1d(Rm_global, domainMin=emin, domainMax=emax,
            axes=axesModule.axes(labelsUnits={1: ('energy_in', 'MeV'), 0: ('radius', 'fm')})) )
    if verbose: print('np,nd,emin,emax=',np,nd,emin,emax)
    
    try: s = states[0]         # fail if only one 'states' namelist
    except: 
        #print 'Rewrite states namelist from ',states
        states = [states]  # so put it in list of 1
    if debug:
        print('\nfresco =',fresco)
        for i in range(0,len(partitions)-1): print('\npartitions ',i,' =',partitions[i])
        try:
            for i in range(0,len(states)): print('\nstates ',i,' =',states[i])
        except:
            print('\nstates only =',states)
            states = [states]
            for i in range(0,len(states)): print('\nstates ',i,' =',states[i])
        #for i in range(0,len(pots)-1): print '\npots ',i,' =',pots[i]
        #for i in range(0,len(variables)): print '\nvariables =',variables[i]

    try: pel = fresco['pel']
    except: pel=1
    try: exl = fresco['exl']
    except:exl=1
    if debug: print('pel,exl =',pel,exl)

    domain = stylesModule.projectileEnergyDomain(emin,emax,'MeV')
    style = stylesModule.evaluated( 'eval', '', physicalQuantityModule.temperature( 300, 'K' ), domain, 'from '+inFile , '0.1.0' )
    PoPs_data = databaseModule.database( 'fresco', '1.0.0' )
    resonanceReactions = commonResonanceModule.resonanceReactions()
    computePenetrability = True
    MTchannels = []

    approximation = 'Full R-Matrix'
    #print ' Btype =',fresco['btype'],' bndx = ',fresco['bndx'][0]
    BV = None
    if fresco['btype']=='S':
        BC = resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction
    elif fresco['btype']=='L':
        BC = resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum
    elif fresco['btype']=='A':
        BC = resolvedResonanceModule.BoundaryCondition.Brune
    elif fresco['btype']=='B': # and isinstance(fresco['bndx'][0],float):
        BC = resolvedResonanceModule.BoundaryCondition.Given
        BV = float(fresco['bndx'][0])
    else:
        print('Unrecognized btype =',fresco['btype'],'with bndx =',fresco['bndx'])

    if debug: print("btype,bndx so BC=",fresco['btype'],fresco.get('bndx',None),' so BC=',BC,'BV=',BV)
    rela = fresco.get('rela','')
    relref = fresco.get('relref',1)
    KRL = relref if len(rela)>0 else 0
    
    #amplitudes = amplitudes  # for RWA in Mev**(1/2) for gnd 
    LRP = 2  # do not reconstruct resonances pointwise (for now)
    jtmax = fresco['jtmax']

    itcm = len(states)-1+1   #  = ITCM   (the -1 is for the final empty namelist)
    rrc=[[None for j in range(itcm)] for i in range(itcm)]  # initialize array
    ZAdict = {};  p1p2 = {}
    ic=0
    inc = partitions[ic]
    itstart = 0
    itfin = itstart + inc['nex']-1
    cm2lab = 0
    spinUnit = spinModule.baseUnit
    gchannelName = None
    CNspin, CNparity = CNspec
    CNparity = 1 if CNparity in ['P','p'] else -1
    Q_offset = 0.0
    #if debug: print 'itsart,itfin=',itstart,itfin
    for it in range(len(states)):  # ignore final empty namelist)
        if it>itfin:
            ic+=1   # next partition
            inc = partitions[ic]
            itstart = itfin+1
            if debug: print(ic,' inc[]=',inc)
            itfin = itstart + inc['nex']-1
            #if debug: print 'itsart,itfin=',itstart,itfin
       
        p,pMass,pZ = inc['namep'],inc['massp'],inc['zp']
        t,tMass,tZ = inc['namet'],inc['masst'],inc['zt']
#  Use standard GND names:
        pA = int(pMass+0.5)
        tA = int(tMass+0.5)
        p = idFromZAndA(pZ,pA) if p not in ['photon','gamma'] else 'photon'
        t = idFromZAndA(tZ,tA)

        Qvalue = inc['qval']
        try: prmax = partitions[ic]['prmax']
        except: prmax = rmatch
        ia=it-itstart
        tex = nuclideIDFromIsotopeSymbolAndIndex(t,ia)
        
        rr = '%s + %s' % (p,tex)  # label for resonanceReaction
        rrc[ic][ia] = rr
        if debug: print("rrc [",ic,',',ia,'] =',rrc[ic][ia])
        state = states[it]
        
        try:    jp,ptyp,ep = state['jp'],state['ptyp'],state['ep']
        except: pass
        try:
            jt,ptyt,et = state['jt'],state['ptyt'],state['et']
        except:
            print('Cannot find target info in ',state,'\n Stop')
            raise SystemExit
        pp = ptyp
        pt = ptyt
        Jp = int(jp) if abs(jp - int(jp))<0.1 else '%i/2' % int(2*jp)
        Jt = int(jt) if abs(jt - int(jt))<0.1 else '%i/2' % int(2*jt)
	# convert parities in namelist from fresco (-1,1) to gnd (-+) strings:
        state['ptyp'] = pp
        state['ptyt'] = pt
        QI = Qvalue - ep - et
        channelName = '%s + %s' % (p,tex)
        ZAdict[rr] = (float(pMass),float(tMass),float(pZ),float(tZ),float(QI),float(prmax))
        p1 = (1-pp)//2; p2 = (1-pt)//2  #  0 or 1 for + or - parities
        p1p2[(ic+1,ia+1)] = (p1,p2)       # save for later access from ic, and ia
        if debug: print('p1p2[(',ic+1,ia+1,')] = (',p1,p2,')')

        MT = 5
        if p=='photon':            MT = 102
        elif p=='n' :              MT = 50+ia
        elif p=='H1' :             MT = 600+ia
        elif p=='H2' :             MT = 650+ia
        elif p=='H3' :             MT = 700+ia
        elif p=='He3' :            MT = 750+ia
        elif p=='He4' :            MT = 800+ia

        if verbose: print(ic,ia,' projectile=',p,pMass,pZ,Jp,ptyp, ', target=',tex,tMass,tZ,Jt,ptyt, ' Q=',Qvalue,' MT=',MT,' prmax =',prmax)
        if pZ==0 and pMass == 0 :   # g
            projectile = miscModule.buildParticleFromRawData( gaugeBosonModule.particle, p, mass = ( 0, 'amu' ), spin = (Jp,spinUnit ),  parity = (ptyp,'' ), charge = (0,'e') )
            CNspin = Jt
            CNparity = ptyt
        elif pZ<1 and pMass < 1.5 and p != 'H1' :  # n or p
            projectile = miscModule.buildParticleFromRawData( baryonModule.particle, p, mass = (pMass,'amu' ), spin = (Jp,spinUnit ),  parity = (ptyp,'' ), charge = (pZ,'e') )
        else: # nucleus in its gs
            nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, p, index = 0, energy = ( 0.0, 'MeV' ) , spin=(Jp,spinUnit), parity=(ptyp,''), charge=(pZ,'e'))
            projectile = miscModule.buildParticleFromRawData( nuclideModule.particle, p, nucleus = nucleus,  mass=(pMass,'amu'))
        PoPs_data.add( projectile )

        # Some state of target at energy 'et':
        if verbose: print("Build PoPs for target ",tex,Jt,ptyt,tZ,tMass,ia,et)
        nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, tex, index = ia, energy = (et,'MeV' ) , spin=(Jt,spinUnit), parity=(ptyt,''), charge=(tZ,'e') )
        target = miscModule.buildParticleFromRawData( nuclideModule.particle, tex, nucleus = nucleus, mass=(tMass,'amu'))
        #print target.toXML()
        PoPs_data.add( target )


        if ic==pel-1 and ia==exl-1:
            if verbose: print('                                           -- that was the incident channel of',p,tex)
            elastics = (p,tex)
            Q_offset = QI
            MT = 2
            cm2lab = (tMass + pMass)/tMass
                
        try: prmax = partitions[ic]['prmax']
        except: prmax = rmatch
        # Create zero background cross section
        # MTchannels.append((rr,zeroReaction(rr,MT, QI, [projectile,target], None, emin,emax,'MeV', debug), channelName,prmax,p))
        MTchannels.append((rr,(rr,MT, QI, projectile,target, emin,emax), channelName,prmax,p))
        compoundA = pA + tA
        compoundZ = pZ + tZ

    cMass = masses.getMassFromZA( compoundZ*1000 + compoundA )
    compoundName = idFromZAndA(compoundZ,compoundA)
    if CNspin is None or CNparity is None:
         ptyt = 1; Jt=0
         print('\n WARNING: %s spin and parity set to %s,%s by default. If neeed, please fix PoPs data in final file by hand!!!\n' % (compoundName,Jt,ptyt))
    else:
         Jt = CNspin
         ptyt = CNparity
         print('\n%s spin and parity set to %s,%s. If needed, please fix PoPs data in final file by hand!!!\n' % (compoundName,Jt,ptyt))
    CNnucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, compoundName, index = 0, energy = (0,'MeV' ),  spin=(Jt,spinUnit), parity=(ptyt,''))
    CNtarget = miscModule.buildParticleFromRawData( nuclideModule.particle, compoundName, nucleus = CNnucleus, mass=(cMass,'amu'))
    PoPs_data.add( CNtarget )
            
# Check if any damping and hence need for Reich-Moore channel
    damped = False
    changeAmp = False; changed=0
    for v in variables:
        if v['kind']==3:
            damped = damped or v.get('damp',0.0) > 1e-20
        if v['kind']==4:
            is_rwa = v.get('rwa',True)   # same default as frescox
            changeAmp = changeAmp or (is_rwa != amplitudes)
            if is_rwa != amplitudes: changed += 1
        if v['kind']==7:
            damped = damped or v.get('damp',0.0) > 1e-20

    if changeAmp: print(" Need to change ",changed," reduced width amplitudes with formal widths ")

    if damped:
        approximation = 'Reich_Moore'
        print(" Create Reich-Moore channel 'photon' from damping")
        level = 0
        compoundNameIndex = nuclideIDFromIsotopeSymbolAndIndex(compoundName,level)

        gchannelName = '%s + photon' % compoundName
        Q = (pMass + tMass - cMass)*amu
        rrcap = gchannelName
        print("Reich-Moore particle pair: ",gchannelName,' with CN mass %.5f so Q=%.3f, label=%s' % (cMass,Q,rrcap))

#       gData = { '0' : [ 0.0,       .0,           1, None,    1,     +1 ] }
        gammaParticle =  miscModule.buildParticleFromRawData( gaugeBosonModule.particle, 'photon',
            mass = ( 0, 'amu' ), spin = ( zero, spinUnit ),  parity = ( 1, '' ), charge = ( 0, 'e' ))
        PoPs_data.add(gammaParticle)

        nucleus = miscModule.buildParticleFromRawData( nucleusModule.particle, compoundNameIndex, index = level, energy = ( 0.0, 'MeV' ) ,
                                                       spin=(zero,spinUnit), parity=(1,''), charge=(compoundZ,'e') )
        compoundParticle = miscModule.buildParticleFromRawData( nuclideModule.particle, compoundNameIndex, nucleus = nucleus, mass=(cMass,'amu') )
        #print PoPs_data.toXML()
        PoPs_data.add(compoundParticle)
        # if verbose: print PoPs_data.toXML()
        
# Create background ReichMoore cross section (zero to start with)
        MT_capture = 102
        # label = 'capture'
        label = rrcap
        capture = zeroReaction(label,MT_capture, Q, [gammaParticle,compoundParticle], 'damping', emin,emax,'MeV', debug)
        # MTchannels.append((rrcap, capture, gchannelName,None,'photon'))
        MTchannels.append((rrcap, (label,MT_capture, Q, gammaParticle,compoundParticle, emin,emax), gchannelName,None,'photon'))

#  After making all the channels, and gnd is generated for the elastic channel, now add them to gnd
    p,tex = elastics   
    gnd = reactionSuiteModule.reactionSuite( p, tex, 'fresco R-matrix fit', PoPs =  PoPs_data, style = style, interaction='nuclear') 

    for rr,reacInfo,channelName,prmax,p in MTchannels:
#  Get zero background cross section and link to it
        #reaction,channelName,prmax = MTchannels[rr]
        rr,MT, QI, pn,tn, emi,ema = reacInfo
        reaction = zeroReaction(rr, MT, QI - Q_offset, [pn,tn], None, emi,ema,'MeV', debug)
        gnd.reactions.add(reaction)
        eliminated = channelName == gchannelName
        reactionLink = linkModule.link(reaction)
        computeShiftFactor = BC != resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction and not eliminated
        computePenetrability = not eliminated  # Should be False also for fission channels (but they are not specified yet)  TEMPORARY

        rreac = commonResonanceModule.resonanceReaction ( label=rr, reactionLink=reactionLink, ejectile=p, 
                                                     computePenetrability=computePenetrability,
                                                     computeShiftFactor=computeShiftFactor, Q=None, eliminated=eliminated  )
        if prmax is not None and prmax != Rm_global:
            rreac.scatteringRadius = scatteringRadiusModule.scatteringRadius(      
                constantModule.constant1d(prmax, domainMin=emin, domainMax=emax,
                    axes=axesModule.axes(labelsUnits={1: ('energy_in', 'MeV'), 0: ('radius', 'fm')})) )
        resonanceReactions.add(rreac)
        if debug: print("RR <"+rr+"> is "+channelName)

    if cm2lab<1e-5:
        print("Missed elastic channel for cm2lab factor!")
        raise SystemExit

    if Lvals is not None:
        print("Remake channels in each pair for L values up to ",Lvals)

#  Now read and collate the reduced channel partial waves and their reduced width amplitudes
# next we have NJS spin groups, each containing channels and resonances
    spinGroups = resolvedResonanceModule.spinGroups()
    JpiList = []
    for i in range(0,len(variables)): 
        v = variables[i]
        if v['kind']==3: 
            pi = v['par']
            J = v['jtot']
            Jpi = J,pi
            if Jpi not in JpiList: JpiList.append(Jpi)
    if debug: print("   List of Jpi",JpiList)
    NJS = len(JpiList)
    JpiMissing = []
    frac = J-int(J)  # to fix whether integer or half-integer spins!
    NJ = int(jtmax-frac+0.1)+1
    for i in range(NJ):
        J = frac + i
        for pi in [-1,1]:
            if (J,pi) not in JpiList:  JpiMissing.append( (J,pi) )
    NJM = len(JpiMissing)
    if NJM>0: print("Spin-parity groups with no poles:",JpiMissing)
	 
    kvar = 0 # ; ivar2G = {}; G2ivar = [];
    kvarData = []
    # if debug: print(resonanceReactions.toXML())
    for spinGroupIndex in range(NJS+NJM):
        J,piv = JpiList [ spinGroupIndex ] if spinGroupIndex < NJS else JpiMissing[spinGroupIndex-NJS]
        JJ = resolvedResonanceModule.spin( J )
        pi= resolvedResonanceModule.spin( piv)
        x = (1-pi)//2
        if verbose: print('\n',spinGroupIndex,': J,pi,x =',JJ,pi,x)

# Previously we got channel quantum numbers from looking at which combinations have poles.
# But this is not good from physics, as we have to be careful to cater for channels even without poles.
# So now (Oct 9, 2017) I re-organize how to make list of channels.
#
        chans = set()

        itc = 0
        for rreac in resonanceReactions:
            if not rreac.eliminated:
                icch=0; iach=0
                for ic in range(len(rrc)):    # find icch,iach for this reaction pair
                    for ia in range(len(rrc[ic])):
                        if rreac.label==rrc[ic][ia]:
                            icch=ic+1; iach=ia+1
                if debug: print("    pair:",rreac.label," at ic,ia",icch,iach)
                p = rreac.ejectile
                t = rreac.residual
                projectile = PoPs_data[p];
                target     = PoPs_data[t];
                if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
                if hasattr(target, 'nucleus'):     target = target.nucleus
                jp,pt = projectile.spin[0].float('hbar'), projectile.parity[0].value
                jt,tt =     target.spin[0].float('hbar'), target.parity[0].value
                smin = abs(jt-jp)
                smax = jt+jp
                s2min = int(2*smin+0.5)
                s2max = int(2*smax+0.5)
                for s2 in range(s2min,s2max+1,2):
                    sch = s2*0.5
                    lmin = int(abs(sch-JJ) +0.5)
                    lmax = int(sch+JJ +0.5)
                    if Lvals is not None: lmax = min(lmax,Lvals[itc])
                    for lch in range(lmin,lmax+1):
                        if pi != pt*tt*(-1)**lch: continue
                        chans.add((icch,iach,lch,sch))
                        if debug: print(' Partial wave channels IC,IA,L,S:',icch,iach,lch,sch)
                itc += 1
                    
        channelList = sorted(chans)
        NCH = len(channelList)
        if debug: print('    channels =',chans,' (',NCH,')')
        if debug: print('    channelList =',channelList,' (',NCH,')')

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

        for chidx in range(NCH):
            icch,iach,lch,sch = channelList[chidx]

            rr = rrc[icch-1][iach-1]
            if debug: print("From ic,ia =",icch,iach," find channel ",rr)
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
            channels.add( resolvedResonanceModule.channel(str(chidx+firstp), rr, columnIndex=chidx+firstp, L=lch, channelSpin=Sch) )
            if debug: print(str(chidx), str(chidx), int(lch), float(sch), chidx+firstp)

        terms = set()   # for this J,pi spin group
        damping = {}
        for i in range(0,len(variables)): 
            v = variables[i]
            ivare = v.get('ivar',None)
            if ivare is not None and ivare!=i+1 and Covariances is not None:
                print("Variable namelists out of order. Expect %i but found %i" % (i+1,ivare))
            if v['kind']==3:
                Jv = v['jtot']
                if Jv==J and v['par']==pi:
                    term = v['term']
                    terms.add((term,v['energy'],ivare))

                    if damping.get(term,None) is None: damping[term] = 0.0
                    try:  d = v['damp']
                    except: d = 0.
                    damping[term] += d
                    if debug: print(i,':',v,'for term',term,' damping',d)
            if v['kind']==7:
                term = v['term']
                if damping.get(term,None) is None: damping[term] = 0.0
                try:  d = v['damp']
                except: d = 0.
                damping[term] += d
                if debug: print(i,':',v,'for term',term,' damping',d)

        terms = sorted(terms)
        if debug: print('    terms =',terms)
        
        resonances = []
        for term,energy,ivare in terms:
            
            # G2ivar.append(ivare)  #  energies come before widths, in GNDS
            kvar += 1 # ; ivar2G[ivare] = kvar
            if debug: print('Jpi',JJ,pi,'kvar=',kvar,'for E=',energy)
            kvarData.append([J,piv,'E',energy])
            energy += Q_offset
            row = [energy*cm2lab]
            if damped: 
                damp = damping.get(term,0.0)*cm2lab
                row.append(damp)
                kvar += 1 # ; ivar2G[ivare] = kvar
                kvarData.append([JJ,piv,'d',damp])
                if debug: print('kvar=',kvar,'for damping=',damp)
            else:
                damp = 0.0
            for ch in channelList:
                found = False
                ic,ia,lch,sch = ch
                for i in range(0,len(variables)):
                    v = variables[i]
                    #print v['kind'],4 , v['term'],term , v['icch'],ch[0] , v['iach'],ch[1] , v['lch'],ch[2] ,  v['sch'],ch[3]
                    if v['kind']==4 and v['term']==term and v['icch']==ic and v['iach']==ia and v['lch']==lch and  v['sch']==sch:
                        #print ' After ch',ch,' widths =',v['width']
                        p1,p2 = p1p2[(ic,ia)]
                        phaz =  p1 + p2 + lch - x
                        if phaz % 2 == 1:    # should be even. If not: stop
                            print('Error: ic,ia,p1,p2,lch,x,phaz=',ic,ia,p1,p2,lch,x,phaz)
                            sys.exit(1)
                        phase = (-1)**(phaz//2)
                        w = v['width'] * phase   # convert to phase from Fresco, which has explicit i^L Huby phases.
                        if debug: print('    E= %.3f MeV, damp=%.2e width=%.4e l=%i S=%.1f p1,p2,phaz,s=%i,%i,%i: %i %i' % (energy, damp, w,lch,sch,p1,p2,x,phaz,phase))
                        try:
                            is_rwa = v['rwa']
                        except:
                            is_rwa = True  # same as Frescox
                        if is_rwa != amplitudes:   # fix to give correct output: rwa or formal width
                            rr = rr = rrc[ch[0]-1][ch[1]-1]
                            pMass,tMass,pZ,tZ,QI,prmax = ZAdict[ rr ]
                            e_ch = energy + QI - Q_offset
                            penetrability,shift,dSdE,W = getCoulomb_PSdSW(
                                      abs(e_ch),lch, prmax, pMass,tMass,pZ,tZ, fmscal,etacns, False)   # CWF at abs(e_ch)
                            if debug: print('p,t =',p,tex,'QI=',QI,': call coulombPenetrationFactor(L=',lch,'e_ch=',e_ch,') =',penetrability,dSdE,W)
                            #   find gamma or Gamma_formal from the G_obs in the AZR input
                            #    Gamma_formal = G_obs * shifty_denom
                            #    gamma =  sqrt(Gamma_formal/(2*P))
                            if amplitudes:         # GND to have rwa from Gammaf
                                width = ( abs(w) /(2. * penetrability) ) **0.5
                                if w < 0: width = -width
                                if debug: print("   Converting Gammaf",w," to rwa",width)
                            else:           # GND to have Gammaf from rwa
                                width = 2.0 * w*w * penetrability
                                if w < 0: width = -width
                                if debug: print("   Converting rwa",w," to Gammaf",width,'ampl:',amplitudes)
                        else:
                            width = w
                        width *= cm2lab**0.5 if amplitudes else cm2lab
                        if nonzero is not None and abs(width)<1e-20: 
                            print('Change',width,'to',nonzero)
                            width = nonzero
#                       else:
#                           print('No change',width,'to',nonzero,'as',abs(width),1e-20,abs(width)<1e-20)
                        row.append(width)
                        found = True

                        # ivar = v.get('ivar',None)
                        # G2ivar.append(ivar)
                        kvar += 1; #  ivar2G[ivar] = kvar
                        if debug: print('kvar=',kvar,'for width=',width,'from',v['width'])
                        kvarData.append([J,piv,energy,damp,width,ic-1,ia-1,lch,sch])
                        
                nfv = 0 if nonzero is None else nonzero
                if not found: row.append(nfv)
            resonances.append(row)

        if debug: 
            print('Col headers:',len(columnHeaders))
            print('Make table for',J,pi,[len(row) for row in resonances],'\n',resonances)
        table = tableModule.table( columns=columnHeaders, data=resonances )
        spinGroups.add( resolvedResonanceModule.spinGroup(str(spinGroupIndex), JJ, pi, channels,
                        resolvedResonanceModule.resonanceParameters(table)) ) 
                        
#     for J,pi in JpiMissing:
#         JJ = resolvedResonanceModule.spin( J )
#         spinGroupIndex += 1
#         if verbose: print spinGroupIndex,': add empty J,pi =',JJ,pi
#         table = tableModule.table( columns=None, data=None )
#         channels = resolvedResonanceModule.channels()
#         spinGroups.add( resolvedResonanceModule.spinGroup(str(spinGroupIndex), JJ, pi, channels,
#                         resolvedResonanceModule.resonanceParameters(table)) )         
        
    
    RMatrix = resolvedResonanceModule.RMatrix( 'eval', approximation, resonanceReactions, spinGroups, 
            boundaryCondition=BC, boundaryConditionValue=BV,
            relativisticKinematics=KRL, reducedWidthAmplitudes=bool(amplitudes), 
            supportsAngularReconstruction=True, calculateChannelRadius=False )

    docnew = RMatrix.documentation
    docLines = [' ','Converted from SFRESCO search file %s' % inFile,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']
    computerCode = computerCodeModule.ComputerCode( label = 'R-matrix fit', name = 'sfrescox', version = '7.1-6-gad5c8e') #, date = time.ctime() )
    computerCode.note.body = '\n'.join( docLines )     

    listallvars = namelists.toString(['variable'])
    dataLines = []
    for i in range(len(variables)):
        if variables[i]['kind'] in [5,6]:
            vnew = listallvars[i].replace(' = ','=')
            vnew = ' '.join(vnew.split()).replace(" ',","',")
            dataLines += [vnew]
    for datanml in namelists.toString(['data']):
        dnew =  datanml.replace(' = ','=').replace('.false.','F').replace('.true.','T')
        dnew = ' '.join(dnew.split()).replace(" ',","',")
        dataLines += [dnew]
        
    fresco_text = '\n'.join(namelists.toString(['fresco']))
    inputDataSpecs= computerCodeModule.InputDeck( 'Fresco input 1', inFile, ('\n  %s\n' % time.ctime() )  + fresco_text +'\n' )
    computerCode.inputDecks.add( inputDataSpecs )
    inputDataSpecs= computerCodeModule.InputDeck( 'Fitted_data 1', inFile, ('\n  %s\n' % time.ctime() )  + ('\n'.join( dataLines ))+'\n' )
    computerCode.inputDecks.add( inputDataSpecs )
#     print('computerCode:\n', computerCode.toXML( ) )

    docnew.computerCodes.add( computerCode ) 

    resolved = resolvedResonanceModule.resolved( emin,emax,'MeV' )
    resolved.add( RMatrix )

    scatteringRadius = Rm_radius
    unresolved = None
    resonances = resonancesModule.resonances( scatteringRadius, resolved, unresolved )
    gnd.resonances = resonances

    if Covariances is not None and not noCov:
        nvar_covs = Covariances[0]['nvariables']
        icor2ivar = Covariances[0]['variables']
        nRvariables = kvar
        if verbose: print('nvar_covs=',nvar_covs,' varying:',icor2ivar,' for nRvars:',nRvariables)
        nvar_Rcovs = 0; R2icor = []; icor2R = []
        for icor0 in range(nvar_covs):
            icor = icor0+1  # from 1..nvar_covs
            varnum = icor2ivar[icor0]
            kind = variables[varnum-1]['kind']
            if debug: print("Variable",varnum," has kind",kind,"at",icor)
            if kind==3 or kind==4 or kind==7: 
                nvar_Rcovs += 1
                R2icor += [icor]
                icor2R += [nvar_Rcovs]
            else:
                icor2R += [None]
        print("\nCovariance matrix varies",nvar_Rcovs,"R-matrix parameters within nRvars:",nRvariables)
        if nvar_Rcovs > nRvariables: sys.exit(1)
        if verbose: 
            print('     icor2ivar:',icor2ivar, len(icor2ivar))
            print('     R to icor:',R2icor , len(R2icor))
            print('     icor to R',icor2R, len(icor2R))
    #       print '     GND to ivar',G2ivar, len(G2ivar)
    #       print '     ivar to GND',ivar2G #, len(ivar2G)
  
        matrix = zeros([nRvariables,nRvariables])
        covar_matrix = zeros([nvar_Rcovs,nvar_Rcovs])
        for covs in Covariances[1:]:
            row = covs['row']-1
            ivar = icor2ivar[row]
            if verbose: print('Row:',covs['row'],'Fresco var#',varnum,'of kind',variables[varnum-1]['kind'])
            if icor2R[row] is not None:
                for i in range(nvar_Rcovs):
                    ivar_c = icor2ivar[R2icor[i]-1]
                    # print "matrix[",ivar,ivar_c,"] = covs[",R2icor[i]-1,"] from R=",R2icor[i],i
                    matrix[ivar-1,ivar_c-1] = covs['emat'][R2icor[i]-1]
                    covar_matrix[row,i] = matrix[ivar-1,ivar_c-1]
                    
# data for computation
        covData = open('fresco-covData.dat','w')
        print('# covariance data from',inFile, file=covData)
        print(nvar_Rcovs, '#, J,pi,<type>,<value>,partition,excitation,L,S, variable, source', file=covData)
        print('# i   Jtot,parity,L,S')
        for i in range(nvar_Rcovs):
            ifv = icor2ivar[R2icor[i]-1]
            v = variables[ifv]
#             print(i,variables[ifv],' '.join([str(v) for v in kvarData[ifv]]), ifv , file=covData)
            name = v['name']
            kind = v['kind']
#             print(ifv, v)
            datas = []
            if kind==3:
                energy = v['energy']
                datas = [energy]
            if kind==4:
                width = v['width']
                ic = v['icch']-1; ia = v['iach']-1; L = v['lch']; S = v['sch']
                datas = [width, ic, ia, L, S]
            print(i,name,','.join([str(d) for d in datas]), ifv , file=covData)
        for i in range(nvar_Rcovs):
            line = ' '.join([str(covar_matrix[i,j]) for j in range(nvar_Rcovs)]) 
            print(i,line, file=covData)

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
            correlation = zeros([nRvariables,nRvariables])
            if debug: print("Cov shape",matrix.shape,", Corr shape",correlation.shape)
            # print "\nCov diagonals:",[matrix[i,i] for i in range(nRvariables)]
            # print "\nCov diagonals:\n",numpy.array_repr(numpy.diagonal(matrix),max_line_width=100,precision=3)
            print("Diagonal uncertainties:\n",numpy.array_repr(numpy.sqrt(numpy.diagonal(matrix)),max_line_width=100,precision=3))
            for i in range(nRvariables):
                for j in range(nRvariables): 
                    t = matrix[i,i]*matrix[j,j]
                    if t !=0: correlation[i,j] = matrix[i,j] / t**0.5

            from scipy.linalg import eigh
            eigval,evec = eigh(matrix)
            if debug:
                print("  Covariance eigenvalue     Vector")
                for kk in range(nRvariables):
                    k = nRvariables-kk - 1
                    print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print("Covariance eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))


            if True: print('correlation matrix:\n',correlation)
            eigval,evec = eigh(correlation)
            if debug:
                print("  Correlation eigenvalue     Vector")
                for kk in range(nRvariables):
                    k = nRvariables-kk - 1
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
        covarianceSuite = covarianceSuiteModule.covarianceSuite( p, tex, 'fresco R-matrix covariances' )
        covarianceSuite.parameterCovariances.add(parameterSection)

        if debug: print(covarianceSuite.toXMLList())
        if verbose: covarianceSuite.saveToFile('CovariancesSuite.xml')

    else:
        if noCov: print("     Covariance data ignored")
        else:     print("     No covariance data found")
        covarianceSuite = None
    
#     print('gnds:\n', gnd.toXML( ), file=open('read_fresco-out.xml','w') )
    return gnd,covarianceSuite

