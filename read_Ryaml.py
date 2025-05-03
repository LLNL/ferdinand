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
from xData.Documentation import exforDataSet as exforDataSetModule

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
from xData import date as dateModule

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

##############################################  read_Ryaml / Rjson

def read_Ryaml(inFile, x4dict, emin_arg,emax_arg, noCov, plot, verbose,debug):
  
    print("Read",inFile,'\n')
    
    ifile = open(inFile,'r')
    data = load(ifile, Loader=Loader)       
    
    Parts = ['Header', 'R_Matrix', 'Particles', 'Reactions', 'Spin Groups', 'Data', 'Covariances'] 

    Header = data['Header']
    R_Matrix = data['R_Matrix']
    Particles = data['Particles']
    Reactions = data['Reactions']
    SpinGroups = data['Spin Groups']
    Data = data['Data']
    Covariances = data['Covariances']

    proj = Header['projectile']
    targ = Header['target']
    evaluation = Header['evaluation']
    projectileFrame = Header['frame']
    energyUnit = Header['energyUnit']
    emin = Header['emin'] if emin_arg is None else emin_arg
    emax = Header['emax'] if emax_arg is None else emax_arg
    Rm_global = Header['scatteringRadius']
    
    approximation = R_Matrix['approximation']
    IFG = R_Matrix['reducedWidthAmplitudes']
    BC = R_Matrix['boundaryCondition']
    BV = R_Matrix.get('boundaryConditionValue',None)
    
    elasticChannel = '%s + %s' % (proj,targ)
    if IFG == 0:
        widthUnit = energyUnit
    else:
        widthUnit = energyUnit + '**(1/2)'
    
    domain = stylesModule.ProjectileEnergyDomain(emin,emax,energyUnit)
    style = stylesModule.Evaluated( 'eval', '', physicalQuantityModule.Temperature( 300, 'K' ), domain, 'from '+inFile , '0.1.0' )
    PoPs_data = databaseModule.Database( 'Ryaml', '1.0.0' )
    resonanceReactions = commonResonanceModule.ResonanceReactions()
    
# FIXED DEFAULTS FOR NOW:
    RelativisticKinematics = False


# PARTICLES
    if debug: print(Particles)
    for id in Particles.keys():
        p = Particles[id]
        gndsName = p['gndsName']
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
        if debug: print(id,'is',gndsName,m,Z,s,pt,ex)
        level = 0 if ex == 0.0 else 1  # FIXME

        if Z==0 and m == 0 :   # g
            particle = miscModule.buildParticleFromRawData( gaugeBosonModule.Particle, gndsName, mass = ( 0, 'amu' ), spin = (s,spinUnit ),  parity = (pt,'' ), charge = (0,'e') )
        elif Z<1 and m > 0.5 and m < 1.5 and gndsName != 'H1' :  # n or p
            particle = miscModule.buildParticleFromRawData( baryonModule.Particle, gndsName, mass = (m,'amu' ), spin = (s,spinUnit ),  parity = (pt,'' ), charge = (Z,'e') )
        else: # nucleus in its gs
            if s is not None and pt is not None:
                nucleus = miscModule.buildParticleFromRawData( nucleusModule.Particle, gndsName, index = level, energy = ( ex, energyUnit) , spin=(s,spinUnit), parity=(pt,''), charge=(Z,'e'))
            else:
                nucleus = miscModule.buildParticleFromRawData( nucleusModule.Particle, gndsName, index = level, energy = ( ex, energyUnit ) , charge=(Z,'e'))
            particle = miscModule.buildParticleFromRawData( nuclideModule.Particle, gndsName, nucleus = nucleus,  mass=(m,'amu'))
        PoPs_data.add( particle )
 
    gnds = reactionSuiteModule.ReactionSuite( proj, targ, 'R-matrix fit via Ryaml', PoPs =  PoPs_data, style = style, interaction='nuclear')

# REACTIONS

    reactionOrder = Reactions.get('order',list(Reactions.keys()))

    for id in reactionOrder:
        partition = Reactions[id]
        label = partition['label']
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
        
        reaction  = zeroReaction(label,MT, Q, [PoPs_data[ejectile],PoPs_data[residual]], None, emin,emax,energyUnit, debug)
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
        resonanceReactions.add(rreac)

# SPINGROUPS

    spinGroups = resolvedResonanceModule.SpinGroups()
    spinGroupIndex = 0
    covIndices = []
    spinGroupOrder = SpinGroups.get('order',list(SpinGroups.keys()))
    nFixedPars = 0
    for id in spinGroupOrder:
        group = SpinGroups[id]
        partialWaves = group['channels']
        parity = 1 if id[-1]=='+' else -1
        J      = id[:-1]

        JJ = resolvedResonanceModule.Spin( J )
        pi= resolvedResonanceModule.Parity( parity )
        if verbose  or debug: print('\n ##### Spinset #',spinGroupIndex,': J,pi =',J,parity,'\n',partialWaves)
    
        columnHeaders = [ tableModule.ColumnHeader(0, name="energy", unit=energyUnit) ]
        channelNames = []
        channels = resolvedResonanceModule.Channels()
        
        chidx = 0
        for chan in partialWaves:
            rr,lch,sch,B = chan
            if debug: print(J,parity,rr,lch,sch,B)
            
#             if debug: print("From p,t =",p,t," find channel ",rr)
            thisChannel = resonanceReactions[rr]
            channelName = "%s width" % thisChannel.label

            jdx = 2
            while True:
                if channelName not in channelNames:
                    channelNames.append( channelName ); break
                channelName = '%s width_%d' % (thisChannel.label, jdx)
                jdx += 1

            columnHeaders.append( tableModule.ColumnHeader(chidx+1, name=channelName, unit= widthUnit ) )

            Sch = resolvedResonanceModule.Spin( sch )
            channels.add( resolvedResonanceModule.Channel(str(chidx+1), rr, columnIndex=chidx+1, 
                    L=lch, channelSpin=Sch, boundaryConditionValue = B ))
                            
            if debug: print(str(chidx), str(chidx),'LS:', int(lch), float(sch), chidx+1, 'B=',BC)
            chidx += 1    

        poleData = group['poles']
        rmatr = []
        energies = []
        for pole in poleData.keys():
            pd = poleData[pole]
#             print('For pole',pole,' poleData is',pd)
            covIndex   = pd[0][0]
            poleEnergy = pd[0][1]
            rowData =        [ poleEnergy ]   # pole energy
            covIndices.append( covIndex )   # pole covIndex
            energies.append( poleEnergy )
            if covIndex is None: nFixedPars += 1
            
            for col in range(len(pd[1])):   # widths
                width = pd[1][col][1]
                covIndex = pd[1][col][0] 
                rowData.append(     width )
                covIndices.append(  covIndex )
                if covIndex is None: nFixedPars += 1
                    
            rmatr.append(rowData )

        nle = len(energies)
        nVarPar = len(covIndices)
        if debug: print(" Energies: ",energies)
        if verbose: print(" rmatr: ",rmatr)


        table = tableModule.Table( columns=columnHeaders, data=rmatr )
        spinGroups.add(    resolvedResonanceModule.SpinGroup(str(spinGroupIndex), JJ, pi, channels,
                           resolvedResonanceModule.ResonanceParameters(table)) ) 
        #if verbose: print " J,pi =",J,piv,": partial waves",pw1,"to",partialWave,"\n"
        spinGroupIndex += 1
    if verbose: print(nVarPar," covIndices in new order with",nFixedPars,"fixed")
    if verbose: print(" covIndices in new order: ",covIndices)
        
    RMatrix = resolvedResonanceModule.RMatrix( 'eval', approximation, resonanceReactions, spinGroups, boundaryCondition=BC,
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
    docLines = [' ','Converted from Ryaml parameter file','   '+inFile,time.ctime(),pwd.getpwuid(os.getuid())[4],' ',' ']    
    computerCode = computerCodeModule.ComputerCode( label = 'R-matrix fit', name = 'Ryaml', version = '') #, date = time.ctime() )
    computerCode.note.body = '\n'.join( docLines )     
    
#  DATA
    normData = Data['Normalizations']

#  FITTED
    dataLines = ['\n']
    nVarData = 0
    DataOrder = normData.get('order',list(normData.keys()))
    fakeSubentry = 0
    fakeEntry = '01000'
    fakes = 0
    subentrys = {}
    for name in DataOrder:
        dataDict = normData[name]
#         file     = dataDict.get('file',None)
        datanorm = dataDict['datanorm']
        reffile = dataDict.get('filename',None)
        shape = dataDict.get('shape',False) 
        expected = dataDict.get('expected',1.0)
        syserror = dataDict.get('syserror',None)
        subentry = dataDict.get('subentry',None)
        
        covIndex = dataDict.get('covIndex',None)
        rName = 'r:' + name
        
        dataLine = "&variable kind=5 name='%s' datanorm=%f " % (rName,datanorm)
        if reffile is not None: dataLine += "reffile='%s'" % reffile

        if subentry is None: 
            subentry = x4dict.get(name,None)
        if subentry is None: 
            try:
                subentry =  name.split('_e')[0].split('-',1)[1]
            except:
                subentry = '0'
            if len(subentry) < 8 or len(subentry) > 10:
                fakeSubentry += 1
                subent = fakeEntry + str(fakeSubentry).zfill(3)
                print('Subentry name',subentry,'for dataset',name,'not valid. Choose',subent)
                subentry = subent
                fakes += 1

        if subentry not in subentrys:
            subentrys[subentry] = exforDataSetModule.ExforDataSet(subentry, dateModule.Date(None))
            subentrys[subentry].note.body = "\n  --Ryaml data begins--\n    --normalization begins--\n"
        subentryNote = subentrys[subentry].note

        subentryNote.body += '      %s:\n' % name
        if covIndex is not None:
            subentryNote.body += '        covIndex: %s\n' % covIndex
        if datanorm is not None:
            subentryNote.body += '        datanorm: %s\n' % datanorm
        if expected is not None:
            subentryNote.body += '        expected: %s\n' % expected
        if reffile is not None: 
            subentryNote.body += '        filename: %s\n' % reffile
        if shape is not None:
            subentryNote.body += '        shape: %s\n' % shape
        if syserror is not None:
            subentryNote.body += '        syserror: %s\n' % syserror

        if subentry is not None: 
            dataLine += " subentry='%s' " % subentry
        else:
            print('## No subentry for dataset',name)

        if covIndex is not None: 
            dataLine += "covIndex='%s'" % covIndex
            covIndices.append(covIndex)
            
        dataLine += '/ '
        dataLines.append(dataLine)
        nVarData += 1       
    
    if fakes>0: print('\nSome fake subentry names generated. Use --x4 option to read file of correct entries\n')

    for subentry in subentrys:
        exforDataSet = subentrys[subentry]
        exforDataSet.note.body += "    --normalization ends--\n  --Ryaml data ends--"
        docnew.experimentalDataSets.exforDataSets.add(exforDataSet)

    docNorms = computerCodeModule.InputDeck( 'Data normalizations from Ryaml', inFile, '\n'.join( dataLines ) )
    computerCode.inputDecks.add( docNorms )
    docnew.computerCodes.add( computerCode ) 
    
# COVARIANCES

    covariances = Covariances.get('square matrix',None)      
    nParams = nVarPar + nVarData  # Data covariances not yet included in GNDS  FIXME


        
    if covariances is not None and not noCov:
        ncovs = len(covariances)
        if nParams > ncovs:
            nParams = nVarPar
            print('Covariance data not given for',nVarData,'data norm uncertainties')
            nVarData = 0
        
        print("Covariance matrix for",nParams,"varied (",nVarPar,'R parameters and',nVarData,'data norms), out of',ncovs)
        matrix = numpy.zeros([nParams,nParams])
        for i in range(nParams):
            for j in range(nParams):
                ic = covIndices[i]
                jc = covIndices[j]
                if ic is not None and jc is not None:
                    matrix[i,j] = covariances[ic][jc]
                else:
                    matrix[i,j] = 0.0     #  one of i or j is not in fit

# store into GNDS (need links to each spinGroup)
        parameters = covarianceModelParametersModule.Parameters()
        startIndex = 0
        for spinGroup in resonances.resolved.evaluated:
            nParams = spinGroup.resonanceParameters.table.nColumns * spinGroup.resonanceParameters.table.nRows
            if nParams == 0: continue
            parameters.add( covarianceModelParametersModule.ParameterLink(
                label = spinGroup.label, link = spinGroup.resonanceParameters.table, root="$reactions",
                matrixStartIndex=startIndex, nParameters=nParams))
            startIndex += nParams
        if debug: 
            print(parameters.toXML(),'\n')
            print(type(matrix))
            print('matrix:\n',matrix)
        if False and verbose:
            correlation = numpy.zeros([nParams,nParams])
            if debug: print("Cov shape",matrix.shape,", Corr shape",correlation.shape)
            # print "\nCov diagonals:",[matrix[i,i] for i in range(npars)]
            # print "\nCov diagonals:\n",numpy.array_repr(numpy.diagonal(matrix),max_line_width=100,precision=3)
            print("Diagonal uncertainties:\n",numpy.array_repr(numpy.sqrt(numpy.diagonal(matrix)),max_line_width=100,precision=4))
            for i in range(nParams):
                for j in range(nParams): 
                    t = matrix[i,i]*matrix[j,j]
                    if t !=0: correlation[i,j] = matrix[i,j] / t**0.5

            from scipy.linalg import eigh
            eigval,evec = eigh(matrix)
            if debug:
                print("  Covariance eigenvalue     Vector")
                for kk in range(nParams):
                    k = nParams-kk - 1
                    print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print("Covariance eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))

            eigval,evec = eigh(correlation)
            if debug:
                print("  nParams eigenvalue     Vector")
                for kk in range(npars):
                    k = nParams-kk - 1
                    print(k,"%11.3e " % eigval[k] , numpy.array_repr(evec[:,k],max_line_width=200,precision=3, suppress_small=True))
            else:
                print("Correlation eivenvalues:\n",numpy.array_repr(numpy.flip(eigval[:]),max_line_width=100,precision=3))

        GNDSmatrix = arrayModule.Flattened.fromNumpyArray(matrix, symmetry=arrayModule.Symmetry.lower)
        Type=covarianceEnumsModule.Type.absolute
        covmatrix = covarianceModelParametersModule.ParameterCovarianceMatrix('eval', GNDSmatrix,parameters, type=Type )

        rowData = covarianceSectionModule.RowData(gnds.resonances.resolved.evaluated, root='')
        parameterSection = covarianceModelParametersModule.ParameterCovariance("resolved resonances", rowData)
        parameterSection.add(covmatrix)

        covarianceSuite = covarianceSuiteModule.CovarianceSuite( proj, targ, 'EDA R-matrix covariances', interaction='nuclear')
        covarianceSuite.parameterCovariances.add(parameterSection)
        evalStyle = gnds.styles.getEvaluatedStyle().copy()
        covarianceSuite.styles.add( evalStyle )
        if verbose: print(covarianceSuite.toXML())

        if hasattr(gnds, 'loadCovariances'): 
            for oldCov in gnds.loadCovariances():
                gnds.externalFiles.pop(oldCov.label)
        gnds.addCovariance(covarianceSuite)

        if debug: covarianceSuite.saveToFile('CovariancesSuite.xml')
        if debug: gnds.saveToFile('gnds.xml')

    else:
        if noCov: print("     Covariance data ignored")
        else:     print("     No covariance data found")

    return gnds
        
################################# MAIN PROGRAM
if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Translate R-matrix evaluations from Ryaml to GNDS')

    parser.add_argument('inFile', type=str, help='The input file you want to read as yaml.' )
    parser.add_argument("-e", "--emin", type=float, help="Min projectile lab energy")
    parser.add_argument("-E", "--Emax", type=float, help="Max projectile lab energy")
    parser.add_argument("-n", "--noCov", action="store_true", help="Ignore covariance matrix")
    parser.add_argument("-x", "--x4", type=str, help="List of exfor subentry names")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug output (more than verbose)")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot covariance matrix with matplotlib")
# Process command line options
    args = parser.parse_args()
        
    x4dict = {}
    if args.x4 is not None:
        lines = open(args.x4,'r').readlines( )
        for line in lines:
            name,subentry,*_  = line.split()
            x4dict[name] = subentry
#         print('x4dict:',x4dict)
#         print('x4dict entries',len(x4dict.keys()))

    gnds = read_Ryaml(args.inFile, x4dict, args.emin,args.Emax, args.noCov,args.plot, args.verbose,args.debug)

    output = args.inFile+'.xml'
    files = gnds.saveAllToFile( output , covarianceDir = '.' )
    print('Files written:\n',output,'\n',str(files[0]))
