#!/usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

from pqu import PQU as PQUModule

##############################################  write_hyrma

def write_hyrma(gnd,outFile,verbose,debug):
    file = open(outFile,'w')

    rrr = gnd.resonances.resolved
    Rm_Radius = gnd.resonances.scatteringRadius
    Rm_global = Rm_Radius.getValueAs('fm')
    RMatrix = rrr.evaluated
    PoPs = gnd.PoPs    
    nch = len(RMatrix.resonanceReactions)

    if RMatrix.reducedWidthAmplitudes: #  This is meaning of widths WITHIN given gnd evaluation
        print("Writing hyrma files using rwa (not widths) not yet implemented\nTry turning off any -a option\n Stop")
        raise SystemExit

    proj,targ = gnd.projectile,gnd.target
    elasticChannel = '%s + %s' % (proj,targ)
    elastic = None
    for pair in RMatrix.resonanceReactions:
        p,t = pair.ejectile,pair.residual
        if p==proj and t==targ: elastic=pair.label
    if elastic is None:
        print("Error: no elastic channel found!")
        raise SystemExit
# elastic = particle pair with incident projectile. 

    string = ' '+str(nch-1)+'                 ! number of separated pairs\n'
    file.write(string)

    labo = -1
    intLabels = {}
    for pair in RMatrix.resonanceReactions:
        reaction = pair.reactionLink.link
        labo += 1
        string = ' '+str(labo)+'                 ! '+pair.label+'\n'
        file.write(string)

        if pair.Q is not None:
            Q_MeV = pair.Q.getConstantAs('MeV')
        else:
            Q_MeV = reaction.getQ('MeV')

        pex,tex = pair.ejectile,pair.residual
        intLabels[pair.label] = labo

        projectile = PoPs[pex];         target = PoPs[tex];
        pA = projectile.getMass('amu');  tA = target.getMass('amu')

        if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
        if hasattr(target, 'nucleus'):     target = target.nucleus

        pZ = projectile.charge[0].value; tZ = target.charge[0].value   # nuclear charge, not atomic !!
        jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
        try: 
            jt,tt,et =     target.spin[0].float('hbar'),     target.parity[0].value,    target.energy[0].float('MeV')
        except: #  e.g. generic Reich-Moore capture nucleus
            jt,tt,et = 0.0,+1,0.0

        string = ' %f %i  %i %i !  A Z 2*I_t pi_t \n %f %i  %i %i !  A Z 2*I_p pi_p \n' % (tA,tZ, jt*2,tt, pA,pZ, jp*2,pt)
        file.write(string)
        if reaction.label == elastic:
            lab2cm = tA/(pA + tA)
            widfac = lab2cm

    if debug: print("labo =",labo,", intLabels:",intLabels)
    file.writelines('\n')
    for Jpi in RMatrix.spinGroups:
        if verbose: print('J/pi:',Jpi.spin,Jpi.parity)
        string =  ' '+str(2*Jpi.spin)+'  '+str(Jpi.parity)+'                ! 2*J, parity\n'
        file.write(string)
        string = '    '+str(len(Jpi))+'                ! dimension\n'    
        file.write(string)
    
        string = '    '+str(labo)+'                ! #of channels\n\n' #counting from 0;
        file.write(string)
        
        R = Jpi.resonanceParameters.table
        energy = R.getColumn('energy','MeV')
        widths = [R.getColumn( col.name, 'MeV' ) for col in R.columns if col.name != 'energy']

        string = ['  '+str(energy[i]) for i in range (0,len(energy))]
        file.writelines(string)
        file.writelines('\n')
        string = ['       '+str(0.0)+'  ' for i in range (0,len(energy))]
        file.writelines(string)
        file.writelines('\n\n')
    
        for ch in Jpi.channels:
            n = ch.columnIndex
            kp = ch.resonanceReaction
            L = ch.L
            sch = ch.channelSpin
            channelSpin = int (2 * sch)
            pair = RMatrix.resonanceReactions[kp]
            p,t = pair.ejectile,pair.residual
            
            projectile,target =gnd.PoPs[p],gnd.PoPs[t]
            if hasattr(projectile, 'nucleus'): projectile = projectile.nucleus
            if hasattr(target, 'nucleus'):     target = target.nucleus
            jp,pt,ep = projectile.spin[0].float('hbar'), projectile.parity[0].value, 0.0
            try: 
                jt,tt,et = target.spin[0].float('hbar'), target.parity[0].value, target.energy[0].float('MeV')
            except: #  e.g. Reich-Moore capture nucleus
                jt,tt,et = 0.0,+1,0.0

            string = '    '+str(intLabels[kp])+'  '+str(abs(channelSpin))+'  '+str(L)+'    ! channel #, 2*s, L\n'
            file.write(string)
            string = '    %i  %i       ! 2*I_t pi_t\n    %i  %i       ! undefined\n' % (jt*2,tt,jp*2,pt)    # TEMPORARY
            file.write(string)
            string = ['       '+str(widths[n-1][i]*widfac) for i in range (0,len(R))]
            file.writelines(string)
            string = ['       '+str(0.0)+'  ' for i in range (0,len(R))]
            file.writelines('\n')
            file.writelines(string)
            file.writelines('\n\n')

    file.writelines(' %s elastic\n' % intLabels[elastic])

    print("\nHYRMA input written: %s" % outFile)
