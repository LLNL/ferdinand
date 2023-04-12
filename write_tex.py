##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma,tex         #
#                                            #
##############################################

import math,numpy
from pqu import PQU as PQUModule
from fudge.processing.resonances.getCoulombWavefunctions import *
import fudge.resonances.resolved as resolvedResonanceModule
from fudge import documentation as documentationModule
import masses
from PoPs.chemicalElements.misc import *

import os,pwd

##############################################  write_tex

def write_tex(gnd,inFile,outFile,background, printEcm,printEComp,squeeze,zero, verbose,debug):

    print("inFile:",inFile)
    latex = open(outFile ,'w')
    if verbose and background is not None: print("Background poles are above ",background)
    title = "R-matrix parameters in %s" % inFile
    title = title.replace('_','\_')

    PoPs = gnd.PoPs
    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.getScatteringRadius()
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    emin = PQUModule.PQU(rrr.domainMin,rrr.domainUnit).getValueAs('MeV')
    emax = PQUModule.PQU(rrr.domainMax,rrr.domainUnit).getValueAs('MeV')

    BC = RMatrix.boundaryCondition
    BV = RMatrix.boundaryConditionValue
    IFG = RMatrix.reducedWidthAmplitudes  
    fmscal = 0.0478450
    etacns = 0.1574855
    amu = 931.494013

    # printEcm = True   # Print Cm energies in elastic channel, otherwise lab projectile energies
    # printEComp = False # Make E=0 the threshold of the elastic channel
    # printEComp = True # Make E=0 the ground state of the composite system


    docHeader = """
\\documentclass[aps]{revtex4}
\\usepackage{longtable}
\\begin{document}

\\title{ %s }
\\author{%s} 
\\date{ \\today }
\\maketitle
        """
    docFooter = """
\\end{document}
        """

    propTableHeader = "\
 \n\
\\begin{table}[h] \n\
\\caption{ Particle Properties. \\\\ Masses are in amu, and excitation energies in MeV. } \n\
\\begin{tabular}{| @@@@ |} \n\
\\hline\\hline \n"

    chanTableHeader = "\
 \n\
\\begin{table}[h] \n\
\\caption{ Channel Properties. Q values are in MeV, and radii in fm.} \n\
\\begin{tabular}{| @@@@ |} \n\
\\hline\\hline \n"

    RtableHeader = "\
SSSSS \\begin{longtable}{c|@@@@} \n\
\\caption{ \n\
R-matrix parameters !!!!! +++++ ^^^^ \n\
}\\\\[1pt] \n\
\\hline\\hline \n\
\\endhead \n\
\\hline \n\
\\endfoot \n"


    
    proj,targ = gnd.projectile,gnd.target
    elasticChannel = '%s + %s' % (proj,targ)
    print('Elastic is <%s>\n' % elasticChannel)
    PoPs = gnd.PoPs    
    
    npairs = 0
    proplines = []
    chanlines = ['GNDS Label~~~   & Projectile & Target & Q value  & Radius & Compound & Eliminated  \\\\ \n','\\hline \n']
    for pair in RMatrix.resonanceReactions:
        npairs += 1
        kp = pair.label
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
            QI = pair.Q.getConstantAs('MeV')
        else:
            QI = reaction.getQ('MeV')
        if pair.getScatteringRadius() is not None:
            prmax =  pair.getScatteringRadius().getValueAs('fm')
        else:
            prmax = Rm_global

        if verbose: print(pMass, tMass, cZA,masses.getMassFromZA( cZA ))
        # cnDefect = (pMass + tMass - masses.getMassFromZA( cZA ))*amu # MeV
        cnDefect = (masses.getMassFromZA( pZA ) + masses.getMassFromZA( tZA ) - masses.getMassFromZA( cZA ))*amu # MeV
        #print pMass, tMass, cZA,masses.getMassFromZA( cZA ),cnDefect
        compoundName = idFromZAndA(cZA//1000,cZA % 1000)
        
#   proplines = ['Particle & Mass & Charge & Spin & Parity & $E^*$  \\\\ \n','\\hline \n']
        jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try:
            jt,tt,et =     target.spin[0].float('hbar'), target.parity[0].value,     target.energy[0].pqu('MeV').value
        except:
            jt,tt,et = None,None,None
        p = p.replace('_','\_')
        t = t.replace('_','\_')
        kp=kp.replace('_','\_')
        proplines += [' %s & %s & %s & %s & $%s$ & $%s$ \\\\ \n' % (p,pMass,pZ,jp,pt,ep)]
        proplines += [' %s & %s & %s & %s & $%s$ & $%s$ \\\\ \n' % (t,tMass,tZ,jt,tt,et)]

        chanlines += [' %s & %s & %s & $%s$ & $%s$ & %s & %s \\\\ \n' % (kp,p,t,QI,prmax,compoundName,pair.eliminated)]

        if pair.label == elasticChannel: 
            lab2cm = tMass / (pMass + tMass)    
            Qelastic = QI
            elasticDefect = cnDefect

    print("Elastic channel Q=",Qelastic," with lab2cm factor = %.4f" % lab2cm, "   cnDefect = %.4f" % elasticDefect)
    tableFooter = '\\hline\n \\end{tabular}\n \\end{table}\n\n'

### HEADER
    latex.writelines(docHeader % (title,pwd.getpwuid(os.getuid())[4]))

### PARTICLE PROPERTIES
    colMarkers = 'llcc c l'
    tHead = propTableHeader.replace('@@@@',colMarkers)
    latex.writelines(tHead)

    proplines = ['Particle & Mass & Charge & Spin & Parity & $E^*$  \\\\ \n','\\hline \n'] + sorted(set(proplines))
    latex.writelines(proplines)

    latex.writelines(tableFooter)
    
### CHANNEL PROPERTIES
    colMarkers = 'llccc c l'
    tHead = chanTableHeader.replace('@@@@',colMarkers)
    latex.writelines(tHead)

    # print chanlines
    latex.writelines(chanlines)

    latex.writelines(tableFooter)
    


### R-MATRIX PARAMETERS
    maxChans = 0
    for Jpi in RMatrix.spinGroups: maxChans = max(maxChans,len(Jpi.channels))
    cols = maxChans + 1
    print("Making a table with %i columns" % cols)

    width_unitsi = 'unknown'
    Overrides = 0
    for Jpi in RMatrix.spinGroups:
        R = Jpi.resonanceParameters.table
        if len(R.columns)>1: width_unitsi =  R.columns[1].unit
        for ch in Jpi.channels:
            if ch.boundaryConditionValue is not None: Overrides += 1

    colMarkers = 'r' * maxChans
    squeezer = '' if not squeeze else '\\squeezetable \n'
    tHead = RtableHeader.replace('@@@@',colMarkers).replace('SSSSS',squeezer)
    Ecap = ''
    if not printEcm:
        Ecap += "\\\\ Pole energies in the laboratory frame of the elastic channel %s." % elasticChannel
    if printEcm and not printEComp: 
        Ecap += "\\\\ Pole energies in the centre-of-mass frame of the elastic channel." 
    if printEcm and printEComp: 
        Ecap += "\\\\ Pole energies are relative to ground state of composite %s system at $%.3f$ MeV below threshold." % (compoundName, elasticDefect)
    tHead = tHead.replace('+++++',Ecap)
    
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
        
    if BC != resolvedResonanceModule.BoundaryCondition.Brune: BC = "$B = %s$" % btype
    boundary = " in the %s basis" %  BC
    if Overrides: boundary += ' (with %s overrides)' % Overrides
    tHead = tHead.replace('!!!!!',boundary+'.')


    frame = 'cm' if printEcm else 'lab'
    if IFG:
        widths = '\\\\ Reduced width amplitudes $\gamma_c$ in units of %s$^{1/2}$ (%s).' % (width_unitsi,frame)
    else:
        widths = '\\\\ Formal widths $\Gamma_c$ (in the ENDF6 convention) in units of %s (%s).' % (width_unitsi,frame)
    widths += '\\\\ Boundary conditions are %s : %s' % (BC,BV)
    if zero: widths += " Channels with all zero widths are not printed."
    tHead = tHead.replace('^^^^',widths)
    
    latex.writelines(tHead)    
    
    
    lines = []
    for Jpi in RMatrix.spinGroups:
        jtot = Jpi.spin
        parity = int(Jpi.parity)
        pi = '+' if parity>0 else '-'

        R = Jpi.resonanceParameters.table
        poleEnergies = R.getColumn('energy','MeV')
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']
        Lmax = -1
        LmaxNZ = -1
        zeros = [True for n in range(1+len(Jpi.channels))]
        Overrides = False
        for ch in Jpi.channels:
            n = ch.columnIndex
            Lmax = max(ch.L,Lmax)
            bndx = ch.boundaryConditionValue
            if bndx is not None: Overrides = True
            for i in range(len(poleEnergies)):
                if widths[n-1][i]!=0.0 and ch.L>LmaxNZ: LmaxNZ = ch.L
                zeros[n] = zeros[n] and widths[n-1][i]==0.0

        if verbose: print("\nWF: J,pi =",jtot,pi)
        largeL = (' (zero for all L $\geq$ %i)' % (LmaxNZ+1)) if LmaxNZ < Lmax else '' 
        line = "\\hline\\multicolumn{%i}{l}{ $J^\pi = %.1f^%s$ %s \\rule{0pt}{7pt}}\\\\[1pt]\\hline\n" % (cols, jtot,pi,largeL)
        #print 'Line:',line
        lines += line

        line = 'E'
        lastChannel = ''
        for ch in Jpi.channels:
            n = ch.columnIndex
            rr = ch.resonanceReaction
            rreac = RMatrix.resonanceReactions[rr]
            label = rreac.label
            lch = ch.L
            if lch > LmaxNZ: continue
            if zero and zeros[n]: continue
            sch = float(ch.channelSpin)
            lab = label.replace(' ','').replace('_e0','').replace('_e','\_')
            if label==lastChannel: #  and squeeze:
                line += '& ' 
            else:
                line += '& %s ' % (lab)
            lastChannel = label
        lines += line+'\\\\[1pt]\n'

        line = '(MeV)'
        lastChannel = ''
        for ch in Jpi.channels:
            n = ch.columnIndex
            rr = ch.resonanceReaction
            rreac = RMatrix.resonanceReactions[rr]
            label = rreac.label
            lch = ch.L
            if lch > LmaxNZ: continue
            if zero and zeros[n]: continue
            sch = float(ch.channelSpin)
            SS = str(sch)
            if sch-int(sch)>0: 
                SS = str(int(sch*2))+'/2'
            else:
                SS = int(sch)
            if label==lastChannel: #  and squeeze:
                line += '& %i, %s ' % (lch,SS)
            elif  rreac.eliminated:
                line += '& (damping)'
            else:
                line += '& LS: %i, %s ' % (lch,SS)
            lastChannel = label
        lines += line+'\\\\[1pt]\n'

        for i in range(len(poleEnergies)):
            e = poleEnergies[i] + Qelastic/lab2cm
            if printEComp : e += elasticDefect/lab2cm
            if printEcm: e *= lab2cm 
            line = '$%.6f$ ' % e

            if background is not None:
                if e > background:  line = '%.3f B ' % e

            for ch in Jpi.channels:
                n = ch.columnIndex
                rr = ch.resonanceReaction
                rreac = RMatrix.resonanceReactions[rr]
                lch = ch.L
                if lch > LmaxNZ: continue
                if zero and zeros[n]: continue
                sch = float(ch.channelSpin)
                width = widths[n-1][i]
                if printEcm: width *= lab2cm**0.5 if IFG else lab2cm
                #if debug: print "W,cm,print",widths[n-1][i],printEcm,width
                
                #line += '%.5f %i %.1f &' % (width,lch,sch)
                if width==0:
                    w = '& 0.0 '
                elif abs(width)<10:
                    w = '& $%.5f$ ' % (width)
                elif abs(width)<1e3: # large
                    w = '& $%.2f$ ' % (width)
                else:   # very large!
                    w = '& $%.2e$ ' % (width)
                    pp = w.split('e')
                    # pp[1] = pp[1].replace('+','')
                    pp[1] = pp[1].replace('+','{+}').replace('-','{-}')
                    w = ''.join(pp)
                line += w
                
            line += '\\rule{0pt}{8pt}\\\\[1pt]\n'
            #print 'Line:',line
            lines += line

        if Overrides:
            line = 'B overrides'
            for ch in Jpi.channels:
                bndx = ch.boundaryConditionValue
                if bndx is not None:
                    line += '& $%s$ ' % bndx
                else:
                    line += '& '
            lines += line+'\\\\[1pt]\n'



    tableFooter = '\\end{longtable}\n'

    latex.writelines(lines)
    latex.writelines(tableFooter)

    latex.writelines(docFooter)
    
    
#     \\hline\\hline \n\
#            &      & \\% natural\\hfill{} &   ENDL2009    & ENDL2011.0 & reason & reviewer \\rule{0pt}{7pt} \\\\[-1pt] \n\
#      & $ZA$ & abundance    &   source & source used& given  &  \\rule{0pt}{7pt}     \\\\[1pt] \n\
