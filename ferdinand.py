#!/usr/bin/env python3

##############################################
#                                            #
#    Ferdinand 0.41, Ian Thompson, LLNL      #
#                                            #
#  gnds,endf,fresco,azure,eda,amur,rac,hyrma #
#                                            #
##############################################

import argparse,sys

from fudge import fudgeVersion
if fudgeVersion.FUDGE_MAJORVERSION != 5:
    print('Need Fudge version 5')
    sys.exit()
from fudge import reactionSuite as reactionSuiteModule
from brownies.legacy.converting.endfFileToGNDS import endfFileToGNDS

from pqu import PQU as PQUModule
import brownies.legacy.toENDF6.toENDF6     # this import adds 'toENDF6' methods to many GNDS classes
from brownies.legacy.converting import toGNDSMisc
from fudge.processing import processingInfo
import fudge.styles as stylesModule
import fudge.resonances.resolved as resolvedResonanceModule


from read_azure import read_azure
from write_azure import write_azure
from read_fresco import read_fresco
from write_fresco import write_fresco
from write_hyrma import write_hyrma
from read_eda import read_eda
from write_eda import write_eda
from read_amur import read_amur
from read_rac import read_rac
from write_tex import write_tex
from gndtransform import gndTransform
from reconstructPointwise import reconstructPointwise

############################################## main

print('\nFerdinand')
cmd = ' '.join([t if '*' not in t else ("'%s'" % t) for t in sys.argv[:]])
print('Command:',cmd ,'\n')

# Process command line options
parser = argparse.ArgumentParser(description='Translate R-matrix Evaluations')
parser.add_argument('inFile', type=str, help='The input file you want to translate. Formats: fresco, sfresco, eda, amur, rac, endf, azure, gnd=gnds=xml, ..' )
parser.add_argument('finalformat', type=str,  help="Output source format: fresco, sfresco, eda,  hyrma, endf, azure, gnd=gnds=xml, tex.")
parser.add_argument("-c", "--covFile", type=str, help="Input file with covariance matrix")
parser.add_argument(      "--noCov", action="store_true", help="Ignore input covariance matrices")
parser.add_argument("-i", "--initial", metavar="in-form", type=str, help="Input source format: endf, gnd=gnds=xml=gnds.xml, fresco, eda, amur, apar, rac, sfresco, sfrescoed, hyrma, azure, ...\n This is expected suffix of input file")
parser.add_argument('-o', dest='output', metavar="outFile", default=None, help='''Specify the output file. Otherwise use ``inFile`` with expected suffix removed if present.''' )

parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
parser.add_argument("-d", "--debug", action="store_true", help="Debugging output (more than verbose)")

## For input translations;
parser.add_argument("-w", "--rwa", action="store_true", help="Reading first makes GNDS with reducedWidthAmplitudes")
parser.add_argument("-W", "--RWA", action="store_true", help="When reading azure files, amplitudes are already as reduced width amplitudes, and B=-L.")
parser.add_argument("-L", "--Lvals", type=int, nargs='+', help="When reading fresco files, or writing EDA files, set partial waves up to this list value in each pair.")

parser.add_argument("-e", "--elastic", type=str,  help="ResonanceReaction label of elastic particle-pair in input file")
parser.add_argument("-l", "--lower", metavar="Emin", type=float,  default="0.01", help="Lower energy of R-matrix evaluation")
parser.add_argument("-u", "--upper", metavar="Emax", type=float,  default="20.", help="Upper energy of R-matrix evaluation")
parser.add_argument("-D", "--Distant", metavar="Edist", type=float, default="25",  help="Pole energy above which are all distant poles, to help with labeling. Fixed in sfresco searches.")
parser.add_argument("-B", "--Bound", metavar="Ebound", type=float, default="-0.01",  help="Pole energy below which are all bound poles, to help with labeling. Fixed in sfresco searches.")

## For gndtransform.py:
parser.add_argument("-b", "--boundary", metavar="B", type=str, help="Boundary condition in output: 'Brune'; '-L' or 'L' for B=-L; or 'X' for B=float(X).")
parser.add_argument("-a", "--amplitudes", action="store_true", help="Convert intermediate gnd file stores to reduced width amplitudes, not widths. If not -a or -G, leave unchanged.")
parser.add_argument("-G", "--Gammas", action="store_true", help="Convert intermediate gnd file stores to formal widths, not reduced width amplitudes. Overrides -a.")
parser.add_argument("-E", "--Elastic", metavar="new_label", type=str,  help="ResonanceReaction label of new elastic particle-pair after transforming input.")

parser.add_argument("-Q", "--Q", action="store_true", help="Allow elastic Q values to be non-zero.")
parser.add_argument("-g", "--nogamma", action="store_true", help="Omit gamma channels")
parser.add_argument("-R", "--ReichMoore", type=float, help="Add a Reich-Moore gamma channel with this value")
parser.add_argument("-r", "--noreac", action="store_true", help="Omit all nonelastic (reaction) channels")
parser.add_argument("-f", "--filter", type=str,  help="Filter of csv list of particle-pair-labels to include. Overrides -g,-r options")

parser.add_argument(      "--nocm", action="store_true", help="No incoming transformations of cm-lab pole energies: for old mistaken files")
parser.add_argument("-A", "--Adjust", metavar="Eadjust", type=str, help="Adjust pole energies: give arithmetic function of E, such as 'E+5000 if E>2e6 else E'. Applied after any Barker but before any Brune transformations")
parser.add_argument("-F", "--File", metavar="Efile", type=str, help="Data file for reading R-matrix data")


## For output translations;
parser.add_argument("-z", "--zero", action="store_true", help="Omit zero widths")
parser.add_argument("-n", "--nonzero", type=float, help="Replace zero widths by this value.")
parser.add_argument("-V", "--Volts", metavar="Eunit", type=str, help="Energy units for conversion after making gnds, before output conversions. Not checked.")
parser.add_argument("-6", "--p6"   , action="store_true", help="Limit energies and widths to ENDF6 precision.")
parser.add_argument(     '--lineNumbers', action='store_true', help='Add line numbers to ENDF6 format files')


parser.add_argument("-p", "--pointwise", metavar='dE', type=float, default='0', help="Reconstruct angle-integrated cross sections using Fresco for given E step")
parser.add_argument("-P", "--Pointwise", metavar='Ang', type=float, nargs=2, help="Reconstruct also angle-dependent cross sections, given thmin, thinc (in deg)")

parser.add_argument("-t", "--tf", metavar='dE', type=float, default='0', help="Reconstruct angle-integrated cross sections using TensorFlow for given E step")

parser.add_argument("-M", "--Ecm", action="store_true",  help="Print poles in latex table in CM energy scale of elastic channel.")
parser.add_argument("-C", "--Comp", action="store_true", help="Print poles in latex table in CM energy scale of compound nucleus.")
parser.add_argument("-S", "--Squeeze", action="store_true", help="Squeeze table of printed poles in latex")
parser.add_argument(      "--CN", metavar='CN', type=str, default=(0,1), nargs=2, help="Spin and parity of compound nucleus, if needed")

## MAIN:
recognized_in = ['endf', 'gnd', 'gnds', 'xml', 'gnds.xml', 'fresco', 'sfresco', 'sfrescoed', 'eda', 'amur', 'azr', 'azure', 'apar', 'rac', 'azure2']    #  List of file extensions recognized for input format
recognized_out= ['endf', 'gnd', 'gnds', 'xml', 'gnds.xml', 'fresco', 'sfresco', 'eda', 'azr', 'azure', 'azure2', 'hyrma', 'tex', 'all']    #  List of formats recognized for outputs
outputs_all = ['xml', 'eda', 'sfresco', 'endf', 'hyrma', 'azr']  # Formats all generated when final=='all'

args = parser.parse_args()
debug = args.debug
verbose = args.verbose or debug
in_s = args.inFile.split('.')[-1].lower()
initial = args.initial
if initial==None and in_s in recognized_in: initial = in_s  

if initial==None:  # or suffix not recognized
    print("\nInitial format (-i) not specified or recognized from file name!\n Recognized input formats: ",recognized_in,"\n Stop")
    raise SystemExit

base = args.inFile #.replace( '.'+initial, '')
 #if verbose: base = args.inFile   # TEMP
elastic = args.elastic
#print 'boundary',args.boundary
cov = None

########################### READ INPUT FORMAT TO GNDS ###########################

if initial=='xml' or initial=='gnd' or initial=='gnds' or initial=='gnd.xml' or initial=='gnds.xml':
    gnd=reactionSuiteModule.readXML(args.inFile)
    
elif initial=='endf':
    rce = endfFileToGNDS( args.inFile, toStdOut=False, skipBadData=True, continuumSpectraFix = True, reconstructResonances=False , doCovariances = not args.noCov )
    gnd=rce['reactionSuite']
    if debug: open( args.inFile + ".echo", mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
    
elif initial=='azr' or initial=='azure' or initial=='azure2':
    if elastic == None and False: 
        print("Elastic channel not specified for AZURE input.\n Assuming elastic is pair in first channel")
    gnd,cov=read_azure(args.inFile,args.covFile,elastic,args.RWA,args.noCov, args.lower,args.upper, verbose,debug,args.rwa)
    elastic = None

    if debug: open( base + '.azr2gnd', mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )

elif initial=='eda':
    gnd,cov=read_eda(args.inFile,args.covFile,elastic,args.rwa,args.noCov,args.lower,args.upper, verbose,debug)

    if debug: open( base + '.eda2gnd', mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
    if debug: write_eda(gnd, base + '.eda2eda',None,verbose,debug)
    elastic = None

elif initial=='amur':
    gnd=read_amur(args.inFile,elastic,args.rwa,args.lower,args.upper, verbose,debug)

    if debug: open( base + '.amur2gnd', mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
    elastic = None

elif initial=='rac' or initial=='apar':
    gnd=read_rac(args.inFile,elastic,args.rwa,args.lower,args.upper, args.Lvals, args.zero, verbose,debug)

    if debug: open( base + '.rac2gnd', mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
    elastic = None

elif initial=='fresco' or initial=='sfresco' or initial=='sfrescoed':
    gnd,cov=read_fresco(args.inFile, args.rwa, args.Lvals, args.CN, args.nonzero, args.noCov, verbose,debug)

    if debug: 
        open( base + '.sfresco2gnd', mode='w' ).writelines( line+'\n' for line in gnd.toXMLList( ) )
    if debug and cov is not None: 
        open( base + '.sfresco2cov', mode='w' ).writelines( line+'\n' for line in cov.toXMLList( ) )

else:
    print("\nInitial format <"+initial+"> not recognized!\n Recognized input formats: ",recognized_in,"\n Stop")
    raise SystemExit

############### Modify if requested
## CHANGE:
gndout = gndTransform(gnd,args.nocm, args.Elastic,args.nogamma,args.noreac,args.filter,args.amplitudes,args.Gammas,
                        args.Adjust,args.File,args.ReichMoore,args.Q,args.boundary,args.p6, verbose,debug)

RMatrix = gndout.resonances.resolved.evaluated
finalStyleName = 'eval'


############### Pointwise Reconstruction with Fresco, if requested

if args.pointwise>0 :
    print("Reconstruct pointwise cross sections using Fresco")
    if args.finalformat not in ['endf', 'gnd', 'gnds', 'xml', 'gnd.xml', 'gnds.xml']:
        print(" Only valid for output formats endf, gnd, gnds, xml, gnd.xml, gnds.xml. Stop")
        raise
    thin = True
    finalStyleName = 'recon'
    reconstructedStyle = stylesModule.crossSectionReconstructed( finalStyleName,
            derivedFrom=gnd.styles.getEvaluatedStyle().label )
    reconstructPointwise(gndout,base,verbose,debug,args.pointwise,args.Pointwise,thin,reconstructedStyle)

############### Pointwise Reconstruction with TensorFlow, if requested

if args.tf>0 :
    from reconstructTensorFlow import reconstructTensorFlow
    print("Reconstruct pointwise cross sections using TensorFlow")
    if args.finalformat not in ['endf', 'gnd', 'gnds', 'xml', 'gnd.xml', 'gnds.xml']:
        print(" Only valid for output formats endf, gnd, gnds, xml, gnd.xml, gnds.xml. Stop")
        raise
    thin = True
    finalStyleName = 'recon'
    reconstructedStyle = stylesModule.crossSectionReconstructed( finalStyleName,
            derivedFrom=gnd.styles.getEvaluatedStyle().label )
    reconstructTensorFlow(gndout,base,verbose,debug,args.tf,args.Pointwise,thin,reconstructedStyle)

###############   CONVERT ENERGY UNITS IF REQUESTED

if args.Volts is not None:
    if RMatrix.reducedWidthAmplitudes:
        print("Energy conversion not implemented for amplitudes:  conversion to ",args.Volts," ignored")
    else:
        gndout.convertUnits( {'eV':args.Volts} )
        gndout.convertUnits( {'MeV':args.Volts} )

###############   WRITE OUTPUT FROM GNDS
##outputs_all = ['endf', 'xml', 'fresco', 'sfresco', 'eda', 'tex', 'hyrma']  # Formats all generated when final=='all'

final   = args.finalformat
outputList = [final]
if final=='all': outputList = outputs_all
bnd = args.boundary

for final in outputList:
    # Compute output file names
    suffix = '+'
    if args.Lvals: suffix = ':L'+''.join([str(L) for L in args.Lvals])+suffix
    if args.nocm: suffix = 'c'+suffix
    if args.zero:    suffix = 'z'+suffix
    if args.noreac:    suffix = 'r'+suffix
    if args.nonzero: suffix = 'n'+suffix
    if args.amplitudes: suffix = 'a'+suffix
    if args.Gammas: suffix = 'Gam'+suffix
    if args.Adjust: suffix = 'A'+suffix
    if args.File: suffix = 'F'+suffix
    if args.Volts: suffix = 'V'+suffix
    if args.Ecm: suffix = 'M'+suffix
    if args.Comp: suffix = 'C'+suffix
    if args.Squeeze: suffix = 'S'+suffix
    if args.rwa: suffix = 'w'+suffix
    if args.RWA: suffix = 'W'+suffix
    if args.covFile: suffix = 'C'+suffix
    if args.Q: suffix = 'Q'+suffix
    if args.pointwise: suffix = 'pt'+str(args.pointwise)+suffix
    if args.tf:        suffix = 'tf'+str(args.tf       )+suffix
    if args.Pointwise: suffix = 'P'+str(args.Pointwise[0])+','+str(args.Pointwise[1])+suffix
    
    if args.filter != None:
        suffix = 'f'+args.filter+suffix
        pfilter = args.filter.split(',')
        print(' List of particle pairs to include: ',pfilter)
    else:
        if args.nogamma: suffix = 'nog'+suffix
    if args.ReichMoore is not None:  suffix = ('R%s' % args.ReichMoore) + suffix
    if suffix != '+': suffix = '-'+suffix
    if args.elastic != None and final != 'azr' and final != 'azure':
        suffix = '_e'+str(args.elastic)+suffix
    if args.Elastic != None and final != 'azr' and final != 'azure':
        suffix = '_E'+str(args.Elastic)+suffix
    if bnd != None:
        suffix = '_b' + bnd + suffix

    if args.output != None:
        outFile = args.output
        covFile = outFile+'.cov'
    else:
        outFile = base + suffix + '.' + final
        covFile = base + suffix + '-cov.' + final
    outFile = outFile.replace(' ','')
    covFile = covFile.replace(' ','')
    note = ''

#####  CHECK GNDS FORMATS AS NEEDED FOR OUTPUTS ####@

#   if 'az' in final and RMatrix.reducedWidthAmplitudes:
#       print "Azure outputs need widths not amplitudes. Use -G option\nNo output generated"
#       break
    if 'az' in final and RMatrix.boundaryCondition==resolvedResonanceModule.BoundaryCondition.EliminateShiftFunction: 
        print("Warning: Azure outputs not accurate for  BND = 'S'.")

########################### WRITE OUTPUT FORMAT ###########################
    if final == 'hyrma' :
        write_hyrma(gndout,outFile,verbose,debug)
        covFile = None
        
    elif final == 'fresco':    # Write 3 file: frin, pars, search
        write_fresco(gndout,outFile,verbose,debug,args.zero,args.Distant,args.Bound,False,0.0,False)
        covFile = None
        
    elif final == 'sfresco':   #  Write combo 'sfresco'
        write_fresco(gndout,outFile,verbose,debug,args.zero,args.Distant,args.Bound,True,0.0,False)
        covFile = None
        
    elif final == 'eda':
        if  args.noreac or args.elastic != None: print("Options -r and -e not implemented for EDA outputs")
        write_eda(gndout,outFile,args.Lvals,verbose,debug)
        covFile = None
    
    elif final == 'azure' or final == 'azure2' or final == 'azr':
        if  args.noreac or args.elastic != None: print("Options -r and -e not implemented for Azure outputs")
        write_azure(gndout,outFile,verbose,debug,args.zero)
        covFile = None
    
    elif final == 'gnd' or final == 'gnds' or final == 'xml' or final == 'gnd.xml' or final=='gnds.xml':
        open( outFile, mode='w' ).writelines( line+'\n' for line in gndout.toXMLList( ) )
        if cov is not None:
            open( covFile, mode='w' ).writelines( line+'\n' for line in cov.toXMLList( ) )
        else:
            covFile = None
        
    elif final == 'tex':
        write_tex(gndout,args.inFile,outFile,args.Distant,args.Ecm,args.Comp,args.Squeeze,args.zero, verbose,debug)
        covFile = None
       
    elif final == 'endf' :
        gndout.convertUnits( {'MeV':'eV'} )
        flags = processingInfo.tempInfo( )
        flags['verbosity'] = 0
        with open( outFile, 'w' ) as fout:
            fout.write( gndout.toENDF6( finalStyleName, flags, covarianceSuite = cov , lineNumbers = args.lineNumbers) )
        if cov is not None: note = ' (includes covariances)'
        covFile = None
    else:
        print("\nFinal format <"+final+"> not recognized!\n Recognized formats =",recognized_out)
        outFile = None

    print("\nWritten file ",outFile,note)
    if covFile is not None: print("Written file ",covFile)

### END
