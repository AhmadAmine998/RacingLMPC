from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from cvxopt.solvers import qp
import numpy as np
import datetime
import pdb
# This class is not generic and is tailored to the autonomous racing problem.
# The only method need the LT-MPC and the LMPC is regressionAndLinearization, which given a state-action pair
# compute the matrices A,B,C such that x_{k+1} = A x_k + Bu_k + C

class PredictiveModel():
    def __init__(self,  n, d, map, trToUse):
        self.map = map
        self.n = n # state dimension
        self.d = d # input dimention
        self.xStored = []
        self.uStored = []
        self.MaxNumPoint = 7 # max number of point per lap to use 
        self.h = 8.0 # bandwidth of the Kernel for local linear regression
        self.lamb = 0.0000001 # regularization
        self.dt = 0.1
        self.scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0]])

        self.stateFeatures    = [0, 1, 2]
        self.inputFeaturesVx  = [1]
        self.inputFeaturesLat = [0]
        self.usedIt = [i for i in range(trToUse)]
        self.lapTime = []
    

    def addTrajectory(self, x, u):
        if self.lapTime == [] or x.shape[0] >= self.lapTime[-1]:
            self.xStored.append(x)
            self.uStored.append(u)
            self.lapTime.append(x.shape[0])
        else:
            for i in range(0, len(self.xStored)):
                if x.shape[0] < self.lapTime[i]:
                    self.xStored.insert(i, x) 
                    self.uStored.insert(i, u) 
                    self.lapTime.insert(i, x.shape[0]) 
                    break

    def regressionAndLinearization(self, x, u):
        Ai = np.zeros((self.n, self.n))
        Bi = np.zeros((self.n, self.d))
        Ci = np.zeros(self.n)

        # Compute Index to use for each stored lap
        xuLin = np.hstack((x[self.stateFeatures], u[:]))
        self.indexSelected = []
        self.K = []
        for ii in self.usedIt:
            indexSelected_i, K_i = self.computeIndices(xuLin, ii)
            self.indexSelected.append(indexSelected_i)
            self.K.append(K_i)
        # print("xuLin: ",xuLin)
        # print("aaa indexSelected: ", self.indexSelected)

        # =========================
        # ====== Identify vx ======
        Q_vx, M_vx = self.compute_Q_M(self.inputFeaturesVx, self.usedIt)

        yIndex = 0
        b_vx = self.compute_b(yIndex, self.usedIt, M_vx)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesVx], Ci[yIndex] = self.LMPC_LocLinReg(Q_vx, b_vx, self.inputFeaturesVx)

        # =======================================
        # ====== Identify Lateral Dynamics ======
        Q_lat, M_lat = self.compute_Q_M(self.inputFeaturesLat, self.usedIt)

        yIndex = 1  # vy
        b_vy = self.compute_b(yIndex, self.usedIt, M_lat)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_vy, self.inputFeaturesLat)

        yIndex = 2  # wz
        b_wz = self.compute_b(yIndex, self.usedIt, M_lat)
        Ai[yIndex, self.stateFeatures], Bi[yIndex, self.inputFeaturesLat], Ci[yIndex] = self.LMPC_LocLinReg(Q_lat, b_wz, self.inputFeaturesLat)

        # ===========================
        # ===== Linearization =======
        vx = x[0]; vy   = x[1]
        wz = x[2]; epsi = x[3]
        s  = x[4]; ey   = x[5]
        dt = self.dt

        if s < 0:
            print("s is negative, here the state: \n", x)

        startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
        cur = self.map.curvature(s)
        cur = self.map.curvature(s)
        den = 1 - cur * ey

        # ===========================
        # ===== Linearize epsi ======
        # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
        depsi_vx   = -dt * np.cos(epsi) / den * cur
        depsi_vy   = dt * np.sin(epsi) / den * cur
        depsi_wz   = dt
        depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
        depsi_s    = 0  # Because cur = constant
        depsi_ey   = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

        Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
        Ci[3]    = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x)
        # ===========================
        # ===== Linearize s =========
        # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
        ds_vx   = dt * (np.cos(epsi) / den)
        ds_vy   = -dt * (np.sin(epsi) / den)
        ds_wz   = 0
        ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
        ds_s    = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
        ds_ey   = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * (-cur)

        Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
        Ci[4]    = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x)

        # ===========================
        # ===== Linearize ey ========
        # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
        dey_vx   = dt * np.sin(epsi)
        dey_vy   = dt * np.cos(epsi)
        dey_wz   = 0
        dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
        dey_s    = 0
        dey_ey   = 1

        Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
        Ci[5]    = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x)

        endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

        return Ai, Bi, Ci

    def compute_Q_M(self, inputFeatures, usedIt):
        Counter = 0
        X0   = np.empty((0,len(self.stateFeatures)+len(inputFeatures)))
        Ktot = np.empty((0))

        for it in usedIt:
            X0 = np.append( X0, np.hstack((self.xStored[it][np.ix_(self.indexSelected[Counter], self.stateFeatures)],self.uStored[it][np.ix_(self.indexSelected[Counter], inputFeatures)])), axis=0 )
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        M = np.hstack( (X0, np.ones((X0.shape[0], 1))) )
        Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
        Q = matrix(Q0 + self.lamb * np.eye(Q0.shape[0]))

        return Q, M

    def compute_b(self, yIndex, usedIt, M):
        Counter = 0
        y = np.empty((0))
        Ktot = np.empty((0))

        for it in usedIt:
            y       = np.append(y, np.squeeze(self.xStored[it][self.indexSelected[Counter] + 1, yIndex]))
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
        return b

    def LMPC_LocLinReg(self, Q, b, inputFeatures):
        # Solve QP
        res_cons = qp(Q, b) # This is ordered as [A B C]
        # Unpack results
        result = np.squeeze(np.array(res_cons['x']))
        A = result[0:len(self.stateFeatures)]
        B = result[len(self.stateFeatures):(len(self.stateFeatures)+len(inputFeatures))]
        C = result[-1]
        return A, B, C

    def computeIndices(self, x, it):
        oneVec = np.ones( (self.xStored[it].shape[0]-1, 1) )
        xVec = (np.dot( np.array([x]).T, oneVec.T )).T
        DataMatrix = np.hstack((self.xStored[it][0:-1, self.stateFeatures], self.uStored[it][0:-1, :]))

        diff  = np.dot(( DataMatrix - xVec ), self.scaling)
        norm = la.norm(diff, 1, axis=1)
        indexTot =  np.squeeze(np.where(norm < self.h))
        if indexTot.shape == ():
            indexTot = np.array([indexTot])
        if (indexTot.shape[0] >= self.MaxNumPoint):
            index = np.argsort(norm)[0:self.MaxNumPoint]
        else:
            index = indexTot

        K  = ( 1 - ( norm[index] / self.h )**2 ) * 3/4
        # if norm.shape[0]<500:
        #     print("norm: ", norm, norm.shape)

        return index, K

class LinearizedModel():
    def __init__(self,  n, d, map, trToUse):
        self.map = map
        self.n = n # state dimension
        self.d = d # input dimention
        self.xStored = []
        self.uStored = []
        self.dt = 0.1

        self.lapTime = []
    

    def addTrajectory(self, x, u):
        if self.lapTime == [] or x.shape[0] >= self.lapTime[-1]:
            self.xStored.append(x)
            self.uStored.append(u)
            self.lapTime.append(x.shape[0])
        else:
            for i in range(0, len(self.xStored)):
                if x.shape[0] < self.lapTime[i]:
                    self.xStored.insert(i, x) 
                    self.uStored.insert(i, u) 
                    self.lapTime.insert(i, x.shape[0]) 
                    break

    def regressionAndLinearization(self, x, u):
        Ai = np.zeros((self.n, self.n))
        Bi = np.zeros((self.n, self.d))
        Ci = np.zeros(self.n)

        # ===========================
        # ===== Linearization =======
        a  = u[1]; delta = u[0]
        vx = x[0]; vy   = x[1]
        wz = x[2]; epsi = x[3]
        s  = x[4]; ey   = x[5]
        dt = self.dt

        if s < 0:
            s = s % self.map.TrackLength
            print("s is negative, here the state: \n", x)

        startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
        cur = self.map.curvature(s)
        cur = self.map.curvature(s)
        den = 1 - cur * ey


        # friction
        mu = 1.1

        # Vehicle Parameters
        m  = 1225.887
        lf = 0.88392
        lr = 1.50876
        Iz = 1538.853371
        Fzf = (m * 9.81) * (lr / (lf + lr))
        Fzr = (m * 9.81) * (lf / (lf + lr)) 
        ky1 = 21.92 # Lateral slip stiffness Kfy/Fz at Fznom
        Cf = 1.3507
        Df = mu * Fzf
        Kf = Fzf * ky1
        Bf = Kf / (Cf * Df)
        Cr = 1.3507
        Dr = mu * Fzr
        Kr = Fzr * ky1
        Br = Kr / (Cr * Dr)

        # Compute tire split angle
        alpha_f = delta - np.arctan2( vy + lf * wz, vx )
        alpha_r = - np.arctan2( vy - lf * wz , vx)

        # Compute lateral force at front and rear tire
        Fyf = Df * np.sin( Cf * np.arctan(Bf * alpha_f ) )
        Fyr = Dr * np.sin( Cr * np.arctan(Br * alpha_r ) )
    
        # ===========================
        # ===== Linearize vx =======
        # vx_{k+1} = vx + dt * (a - 1 / m * Df * np.sin( Cf * np.arctan(Bf * delta - np.arctan2( vy + lf * wz, vx ) ) ) * np.sin(delta) + wz * vy)
        x0 = lf*wz + vy
        x1 = delta - np.arctan2(x0, vx)
        x2 = Bf*Cf*Df*np.sin(delta)*np.cos(Cf*np.arctan(Bf*x1))/(m*(vx**2 + x0**2)*(Bf**2*x1**2 + 1))
        x3 = vx*x2
        sympy_generated = np.array([[-dt*x0*x2 + 1, dt*(wz + x3), dt*(lf*x3 + vy), 0, 0, 0]])
        Ai[0, :] = sympy_generated.flatten().copy()
        
        x0 = delta - np.arctan2(lf*wz + vy, vx)
        x1 = Cf*np.arctan(Bf*x0)
        x2 = Df/m
        sympy_generated = np.array([[dt*(-Bf*Cf*x2*np.sin(delta)*np.cos(x1)/(Bf**2*x0**2 + 1) - x2*np.sin(x1)*np.cos(delta)), dt]])
        Bi[0, :] = sympy_generated.flatten().copy()

        Ci[0] = vx + dt * (a - 1 / m * Df * np.sin( Cf * np.arctan(Bf * delta - np.arctan2( vy + lf * wz, vx ) ) ) * np.sin(delta) + wz * vy) - np.dot(Ai[0, :], x)
    
        # ===========================
        # ===== Linearize vy =======
        # vy_{k+1} = vy + dt * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
        x0 = m**(-1.0)
        x1 = lr*wz - vy
        x2 = -x1
        x3 = np.arctan2(x2, vx)
        x4 = (Br**2*x3**2 + 1)**(-1.0)
        x5 = vx**2
        x6 = (x2**2 + x5)**(-1.0)
        x7 = np.cos(Cr*np.arctan(Br*x3))
        x8 = Br*Cr*Dr*x4*x6*x7
        x9 = lf*wz + vy
        x10 = delta - np.arctan2(x9, vx)
        x11 = Bf*Cf*Df*np.cos(delta)*np.cos(Cf*np.arctan(Bf*x10))/((x5 + x9**2)*(Bf**2*x10**2 + 1))
        x12 = vx*x11
        sympy_generated = np.array([[dt*(-wz + x0*(-x1*x8 + x11*x9)), dt*x0*(-vx*x8 - x12) + 1, dt*(-vx + x0*(Br*Cr*Dr*lr*vx*x4*x6*x7 - lf*x12)), 0, 0, 0]])
        Ai[1, :] = sympy_generated.flatten().copy()

        x0 = delta - np.arctan2(lf*wz + vy, vx)
        x1 = Cf*np.arctan(Bf*x0)
        sympy_generated = np.array([[dt*(Bf*Cf*Df*np.cos(delta)*np.cos(x1)/(Bf**2*x0**2 + 1) - Df*np.sin(delta)*np.sin(x1))/m, 0]])
        Bi[1, :] = sympy_generated.flatten().copy()
        
        Ci[1] = vy + dt * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx) - np.dot(Ai[1, :], x)

        # ===========================
        # ===== Linearize wz =======
        # wz_{k+1} = wz + dt * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr))
        x0 = lr*wz - vy
        x1 = vx**2
        x2 = -x0
        x3 = (x1 + x2**2)**(-1.0)
        x4 = np.arctan2(x2, vx)
        x5 = np.cos(Cr*np.arctan(Br*x4))
        x6 = (Br**2*x4**2 + 1)**(-1.0)
        x7 = lf*wz + vy
        x8 = delta - np.arctan2(x7, vx)
        x9 = Bf*Cf*Df*np.cos(delta)*np.cos(Cf*np.arctan(Bf*x8))/((x1 + x7**2)*(Bf**2*x8**2 + 1))
        x10 = lf*x9
        x11 = dt/Iz
        sympy_generated = np.array([[x11*(Br*Cr*Dr*lr*x0*x3*x5*x6 + x10*x7), x11*(Br*Cr*Dr*lr*vx*x3*x5*x6 - vx*x10), x11*(-Br*Cr*Dr*lr**2*vx*x3*x5*x6 - lf**2*vx*x9) + 1, 0, 0, 0]])
        Ai[2, :] = sympy_generated.flatten().copy()

        x0 = delta - np.arctan2(lf*wz + vy, vx)
        x1 = Cf*np.arctan(Bf*x0)
        x2 = Df*lf
        sympy_generated = np.array([[dt*(Bf*Cf*x2*np.cos(delta)*np.cos(x1)/(Bf**2*x0**2 + 1) - x2*np.sin(delta)*np.sin(x1))/Iz, 0]])
        Bi[2, :] = sympy_generated.flatten().copy()

        Ci[2] = wz + dt * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr)) - np.dot(Ai[2, :], x)

        # ===========================
        # ===== Linearize epsi ======
        # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
        depsi_vx   = -dt * np.cos(epsi) / den * cur
        depsi_vy   = dt * np.sin(epsi) / den * cur
        depsi_wz   = dt
        depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
        depsi_s    = 0  # Because cur = constant
        depsi_ey   = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)

        Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
        Ci[3]    = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x)
        
        # ===========================
        # ===== Linearize s =========
        # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
        ds_vx   = dt * (np.cos(epsi) / den)
        ds_vy   = -dt * (np.sin(epsi) / den)
        ds_wz   = 0
        ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
        ds_s    = 1  # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
        ds_ey   = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * (-cur)

        Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
        Ci[4]    = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x)

        # ===========================
        # ===== Linearize ey ========
        # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
        dey_vx   = dt * np.sin(epsi)
        dey_vy   = dt * np.cos(epsi)
        dey_wz   = 0
        dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
        dey_s    = 0
        dey_ey   = 1

        Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
        Ci[5]    = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x)

        endTimer = datetime.datetime.now(); deltaTimer_tv = endTimer - startTimer

        return Ai, Bi, Ci
