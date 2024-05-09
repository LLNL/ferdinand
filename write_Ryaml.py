#! /usr/bin/env python3


##############################################
#                                            #
#    Ferdinand 0.60, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma,tex,yaml    #
#                                            #
##############################################

import math,numpy
from pqu import PQU as PQUModule
from fudge import reactionSuite as reactionSuiteModule

from fudge.processing.resonances.getCoulombWavefunctions import *
import fudge.resonances.resolved as resolvedResonanceModule
from fudge import documentation as documentationModule
import masses
from PoPs.chemicalElements.misc import *

import json,sys
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


import os,pwd

##############################################  write_Ryaml / Rjson

def write_Ryaml(gnds, verbose,debug):
  
    domain = gnds.styles.getEvaluatedStyle().projectileEnergyDomain
    energyUnit = domain.unit
    
    PoPs = gnds.PoPs
    rrr = gnds.resonances.resolved
    Rm_Radius = gnds.resonances.getScatteringRadius()
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs(energyUnit)
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs(energyUnit)

    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes  
    approximation = RMatrix.approximation

    Header = {}
    R_Matrix = {}
    Particles = {}
    Reactions = {}
    SpinGroups = {}
    Data = {}
    Covariances = {}
    

    proj,targ = gnds.projectile,gnds.target
    elasticChannel = '%s + %s' % (proj,targ)
    PoPs = gnds.PoPs    

# HEADER
    Header['covarianceFile'] = '' # covFile
    Header['projectile'] = proj
    Header['target'] = targ
    Header['evaluation'] = gnds.evaluation
    Header['frame'] = str(gnds.projectileFrame)
    Header['energyUnit'] = energyUnit
    Header['emin'] = emin
    Header['emax'] = emax
    Header['scatteringRadius'] = Rm_global
        
# R_Matrix
    R_Matrix['approximation'] = str(approximation)
    R_Matrix['reducedWidthAmplitudes'] = IFG
    R_Matrix['boundaryCondition'] = str(BC)
    R_Matrix['boundaryConditionValue'] = str(BV)
    
    
# REACTIONS

    reactionOrder = []
    reactionBCOverrides = 0
    for pair in RMatrix.resonanceReactions:
        kp = pair.label
        reac = kp.replace(' ','')
        reactionOrder.append(reac)
        reaction = pair.link.link
        p,t = pair.ejectile,pair.residual
        projectile = PoPs[p];
        target     = PoPs[t];
        pMass = projectile.getMass('amu');   tMass =     target.getMass('amu');
        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus

        pZ    = projectile.charge[0].value;  tZ  = target.charge[0].value
        pA    = int(pMass+0.5);              tA  = int(tMass+0.5)
        pZA = pZ*1000 + pA;                  tZA = tZ*1000+tA
        cZA = pZA + tZA   # compound nucleus
        if pair.Q is not None:
            QI = pair.Q.getConstantAs(energyUnit)
        else:
            QI = reaction.getQ(energyUnit)
        if pair.getScatteringRadius() is not None:
            prmax =  pair.getScatteringRadius().getValueAs('fm')
        else:
            prmax = Rm_global
        if pair.hardSphereRadius is not None:
            hsrad = pair.hardSphereRadius.getValueAs('fm')
        else:
            hsrad = prmax
            
        if verbose: print(pMass, tMass, cZA,masses.getMassFromZA( cZA ))
        CN = idFromZAndA(cZA//1000,cZA % 1000)
        
#   proplines = ['Particle & Mass & Charge & Spin & Parity & $E^*$  \\\\ \n','\\hline \n']
        jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt,tt,et =     target.spin[0].float('hbar'), target.parity[0].value,     target.energy[0].pqu(energyUnit).value
        except:
            jt,tt,et = None,None,None

        Particles[p] = {'gndsName':p, 'gsMass':pMass, 'charge':pZ, 'spin':jp, 'parity':pt, 'excitation':float(ep)}
        Particles[t] = {'gndsName':t, 'gsMass':tMass, 'charge':tZ, 'spin':jt, 'parity':tt, 'excitation':float(et)}

        Particles[CN] = {'gndsName':CN, 'gsMass':masses.getMassFromZA( cZA ), 'charge':tZ+pZ, 'excitation':0.0}
        try:
            CN_PoPs = PoPs[CN]
            jCN = CN_PoPs.spin[0].float('hbar')
            pCN = CN_PoPs.parity[0].value
            Particles[CN]['spin'] = jCN
            Particles[CN]['parity'] = pCN
        except:
            pass

        Reactions[reac] = {'label':kp, 'ejectile':p,  'residual':t, 'Q':QI} 
        if prmax != Rm_global:  Reactions[reac]['scatteringRadius'] = prmax
        if hsrad != prmax:      Reactions[reac]['hardSphereRadius'] = hsrad
        
        B = pair.boundaryConditionValue
        if B is not None:       
            Reactions[reac]['B'] = B
            reactionBCOverrides += 1
        
        if pair.label == elasticChannel: 
            lab2cm = tMass / (pMass + tMass)    
            Qelastic = QI

    if debug: print("Elastic channel Q=",Qelastic," with lab2cm factor = %.4f" % lab2cm)
    Reactions['order'] = reactionOrder

### R-MATRIX PARAMETERS
    maxChans = 0
    for Jpi in RMatrix.spinGroups: maxChans = max(maxChans,len(Jpi.channels))
    cols = maxChans + 1

    width_unitsi = 'unknown'
    channelBCOverrides = 0

    if BC is None:
        btype = 'S'
    elif BC==resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction:
        btype = 'S'
    elif BC==resolvedResonanceModule.BoundaryCondition.NegativeOrbitalMomentum:
        btype = '-L'
    elif BC==resolvedResonanceModule.BoundaryCondition.Brune:
        btype = 'Brune'
    elif BC==resolvedResonanceModule.BoundaryCondition.Given:
        btype = BV
    else:
        print("Boundary condition BC <%s> not recognized" % BC,"in write_tex")
        raise SystemExit
    if BV is None: BV = ''
        
    if BC != resolvedResonanceModule.BoundaryCondition.Brune: BC = "B = %s" % btype
    boundary = " in the %s basis" %  BC
    if reactionBCOverrides + channelBCOverrides > 0: 
        print('  Reaction BC overrides: %s,  Partial-wave BC overrides %s' % (reactionBCOverrides,channelBCOverrides) )

    frame = 'lab'
    widthUnit = energyUnit + ('**(1/2)' if IFG==1 else '')
    if debug: print('Boundary conditions are %s : %s in units %s' % (BC,BV,width_unitsi))
    
    index = 0
#     nParameters = 0
    spinGroupOrder = []
    for Jpi in RMatrix.spinGroups:
        jtot = str(Jpi.spin)
        parity = int(Jpi.parity)
        pi = '+' if parity>0 else '-'
        if verbose: print("\nSpin group:",jtot,pi)
        
        spinGroup = jtot + pi
        spinGroupOrder.append(spinGroup)
        group = {}
        
        R = Jpi.resonanceParameters.table
        poleEnergies = R.getColumn('energy',energyUnit)
        widths = [R.getColumn( col.name, widthUnit ) for col in R.columns if col.name != 'energy']
        rows = len(poleEnergies)
        if rows > 0:
            columns = len(R[0][:])
            print(jtot,pi,' ',rows,'poles, each with',columns-1,'widths:',rows*columns,'parameters')
#             nParameters += rows*columns
        else:
            columns = 0
        
        channels = []
        for ch in Jpi.channels:
            n = ch.columnIndex
            rr = ch.resonanceReaction
            rreac = RMatrix.resonanceReactions[rr]
            label = rreac.label
            lch = ch.L
            sch = float(ch.channelSpin)
            B = ch.boundaryConditionValue            
            channels.append([str(rr),lch,sch,B])
            if B is not None: channelBCOverrides += 1
            
        poleData = {}
        for i in range(rows):
            tag = 'pole'+str(i)+':'+"%.3f" % R[i][0]
            par = [ [index, R[i][0]] , [] ]
            index += 1
            for c in range(1,columns):
                par[1].append( [index, R[i][c] ]  )
                index += 1
            poleData[ tag ] = par
            
        if verbose:
            print('poleData',poleData)
            print('channels',channels)

        group['channels'] = channels
        group['poles']    = poleData
        
        SpinGroups[spinGroup] = group
    
    numVariables = index
    SpinGroups['order'] = spinGroupOrder
    print('Number of R-matrix parameters:',numVariables) #,nParameters)
    if reactionBCOverrides + channelBCOverrides > 0: 
        print('  Reaction BC overrides: %s,  Partial-wave BC overrides %s' % (reactionBCOverrides,channelBCOverrides) )
        
# Data

    docVars = []
    docData = []
    computerCodeFit = None
    previousFit = False
    try:
        computerCodeFit = RMatrix.documentation.computerCodes['R-matrix fit']
        ddoc    = computerCodeFit.inputDecks[-1]
        for line in ddoc.body.split('\n'):
            if '&variable' in line.lower() :  docVars += [line]
            if '&data'     in line.lower() :  docData += [line]
        previousFit = True
    except:
        pass
        
    fittedData = {}
    fixedData = {}
    for line in docVars:
        if 'kind=5' in line:
            dataDict = {}
            name = line.split("'")[1].strip()
            name = name.replace('r:','')
            datanorm = float(line.split('datanorm=')[1].split()[0])
            dataDict['datanorm'] = datanorm
#             if datanorm == 1.0: continue  # just default norm
            try:
                filename = line.split('reffile=')[1].split("'")[1]
                dataDict['file'] = filename
            except:
                 pass
#                 print(line.split('reffile=')[1].split("'")[1])
            covIndex = None
            if covIndex is not None:
                dataDict['covIndex'] = covIndex
              
            if datanorm == 1.0:
                fixedData[name]  = dataDict
            else: 
                fittedData[name] = dataDict
            if verbose:
                print("Previous norm for %-20s is %10.5f from %s in cov at %s" % (name,datanorm,filename,covIndex) )

            
    Data['Fitted'] = fittedData
    Data['Fixed']  = fixedData
            
# COVARIANCES

    if hasattr(gnds, 'loadCovariances'): 
        covFileName = []
        for externalFile in gnds.externalFiles:
            covFileName.append(externalFile.path)
            print('Covariances from',covFileName[-1])
            Header['covarianceFile'] = covFileName[-1] # covFile
            
        covariances = gnds.loadCovariances()
        if len(covariances)>0:
    #         print('loaded')
    #         print(covariances[0].toXML())
            evalCovs = covariances[0].parameterCovariances[0].evaluated
            
            covArray = evalCovs.matrix.constructArray()
            if debug: print(covArray)
    #         print(dir(covArray))
            print("Covariance matrix is",covArray.shape)
            Covariances['square matrix'] = covArray.tolist()
#             if plotcovs:
#                 evalCovs.plot()
        else:
            print('No covariances linked')  
    else:
        print('No covariance data')
        

 # Keep Parts order:
 
    Info = {'Header':Header, 'R_Matrix':R_Matrix, 'Particles':Particles, 'Reactions':Reactions, 
            'Spin Groups': SpinGroups, 'Data': Data, 'Covariances':Covariances }
#     output = dump(Info, Dumper=Dumper) 

    Parts = ['Header', 'R_Matrix', 'Particles', 'Reactions', 'Spin Groups', 'Data', 'Covariances']  # ordered
    
    output = ''
    for part in Parts:  # Data.keys():
        d = {part:Info[part]}
        output += dump(d, Dumper=Dumper)
    
    return(output)
    
################################# MAIN PROGRAM
if __name__=="__main__":

    import sys, os,argparse
    # Process command line options
    parser = argparse.ArgumentParser(description='Translate R-matrix Evaluations')

    parser.add_argument('inFile', type=str, help='The input file you want to write as yaml.' )
    parser.add_argument("-j", "--json", action="store_true", help="print json file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug output (more than verbose)")

    args = parser.parse_args()
    
    gnds = reactionSuiteModule.ReactionSuite.readXML_file( args.inFile )
    
    otype = 'Rjson' if args.json else 'Ryaml'
    outFile = args.inFile + '.' + otype
    output = write_Ryaml(gnds, args.verbose,args.debug)

    ofile = open(outFile,'w')
    print(output, file=ofile)