

##############################################
#                                            #
#    Ferdinand 0.50, Ian Thompson, LLNL      #
#                                            #
#    gnd,endf,fresco,azure,hyrma             #
#                                            #
##############################################

def Whittaker(Lmax,RHO,ETA):

    import math
#
#     Calculates  Whittaker  function  W_L(rho,eta  with
#     asymptotic  form  Exp(-(rho + eta(Log(2 rho)))
#
#     Returns (F,FD,IE):
#     Arrays F[L] = W
#            FD[L]= W'  for L=1,..,LMAX  (L in range(LMAX+1))
#			Note: W' = dW/d(rho)
#     Integer IE
#     If IE = 0, allowed to return result e**IE larger than Whittaker,
#                for the IE value returned.
#     If IE > 0, must scale results by that amount.
#
# Note: arrays subscripts are 1 less than in original Fortran code

    def recurse_up() :  # starting from F[0] and FD[0]
        # 61
        C=1.0/RHO
        for M in range(Lmax):
            A=ETA/float(M+1)
            B=A+C*float(M+1)
            F[M+1]=(B*F[M]-FD[M])/(A+1.0)
            FD[M+1]=(A-1.0)*F[M]-B*F[M+1]

    FPMAX = 1e250
    IE = 0
    S = [0. for i in range(7)]
    F = [0. for L in range(Lmax+1)]
    FD= [0. for L in range(Lmax+1)]
    if Lmax > 50:
        LM = Lmax + 10
    else:
        LM = 60
    LMP1 = LM+1
    IS = 6
    PJE=100.0*RHO+1.0
    H=RHO / max(int(PJE),4)
    RHOA=10.0*(ETA+1.0)
    IFcontained = RHOA <= RHO
    if IFcontained: RHOA = RHO
    PJE=RHOA/H+0.5
    RHOA=H*int(PJE)
    if not IFcontained and RHOA < RHO+1.5*H: RHOA = RHO+2.0*H
    
    # 55
    for IS in [6, 5]:
        C=1.0/RHOA # asymptotic expansion
        A=1.0
        B=1.0-C*ETA
        F[0]=A
        FD[0]=B
        for M in range (1,27): # 				DO 56 M=1,26
            D=0.5*(ETA+float(M-1))*(ETA+float(M))*C/float(M)
            A=-A*D
            B=-B*D-A*C
            F[0]=F[0]+A
            FD[0]=FD[0]+B
     
        A=-ETA*math.log(2.0*RHOA)-RHOA
        FPMINL = -math.log(FPMAX)
        if IE==0 and A < FPMINL: IE = int(FPMINL-A)
        A=math.exp(A+IE)
        F[0]=A*F[0]
        FD[0]=A*FD[0] * (-1.0 - 2*ETA/(RHOA))
        
        if IFcontained:  # asymptotic expansion already at the right place
            recurse_up()
            return F,FD,IE

        S[IS] = F[0]
        RHOA = RHOA + H
            
    # Integrate inward from S[6] # RHOA and S[5] @ RHOA-H to RHO with Numerov
    EE = -1.0
    A=2.0-10.0/12.0*H*H*EE
    B=1.0/6.0*H*ETA
    C=1.0+1.0/12.0*H*H*EE
    M1=int(RHOA/H-0.5)
    M2=int(RHO/H-1.5)
    T1=B/float(M1+1)
    T2=B/float(M1)
    JS=M1
    for IS in range(M2-1,M1):
        for I in range(6): S[I] = S[I+1]  # ending with S4=S5;  S5=S6
        T0=T1
        T1=T2
        T2=B/float(JS-1)
        S[6]=((A+10.0*T1)*S[5]-(C-T0)*S[4])/(C-T2)
        JS=JS-1
        if abs(S[6]) > FPMAX: 
           for i in range(1,6):
               S[i] = S[i] / FPMAX
    F[0] = S[2]
    FD[0]  = (1.0/60.0*(S[0]-S[6])+0.15*(S[5]-S[1])+0.75*(S[2]-S[4]))/H

    recurse_up()
    return F,FD,IE

if __name__=="__main__":

    import sys
    L = int(sys.argv[1])
    rho = float(sys.argv[2])
    eta = float(sys.argv[3])

    W,WD,IE = Whittaker(L,rho,eta)
    S = rho*WD[L]/W[L]
    print('  L,rho,eta =%i %.5f %.5f W,W'' = %e %e ie =%i S=%f' %(L,rho,eta,W[L],WD[L],IE,S))
