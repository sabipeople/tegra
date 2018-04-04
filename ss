import numpy as np
import pdb
def normalization(pt):
#    print("in normalization pt\n", pt.shape)

    mean_pt = pt.mean(1)
#    print("mean\n", mean_pt)
    #move origin to centre of mass of point
    transed_pt=pt-mean_pt
    #scaling
    mean_dist=np.sum(np.sqrt(np.diagonal(transed_pt.getT()*transed_pt)))/pt.shape[1]
    with np.errstate(divide='raise'):
        try :
            scale_factor=np.sqrt(2)/mean_dist
        except FloatingPointError:
            print("normalization divide by 0")
            return None, None, None
        except :
            print("something wrong")
    norm_pt=transed_pt*scale_factor
    return norm_pt, mean_pt, scale_factor

def denormalization(pt, scale_f, mean_pt):
    denorm_mat=np.zeros((3,3))
    denorm_mat[0,0]=denorm_mat[1,1]=1/scale_f
    denorm_mat[0,2]=mean_pt[0,0]
    denorm_mat[1,2]=mean_pt[1,0]
    denorm_mat[2,2]=1
    return np.matrix(np.matrix(denorm_mat)*pt)

def toHomo(pt):
    homo_pt=np.zeros((pt.shape[0]+1,pt.shape[1]))
    homo_pt[range(pt.shape[0]),:]=pt
    homo_pt[pt.shape[0],:]=1
    return np.matrix(homo_pt)
def decomposeRot_ZYX(R):
    pitch=np.arctan2(R[2,1],R[2,2])*180/np.pi
    yaw=-np.arcsin(R[2,0])*180/np.pi
    roll=np.arctan2(R[1,0],R[0,0])*180/np.pi
    return pitch,yaw,roll
def decomposeRot_xyz(R):
    pitch=np.arctan2(-R[1,2],R[2,2])*180/np.pi
    yaw=np.arcsin(R[0,2])*180/np.pi
    roll=np.arctan2(R[0,1],-R[0,0])/np.pi
    return pitch,yaw,roll

def getRotMat(x,y,z):
    '''
    it return rx,ry,rz rot mat
    '''
    rx=np.deg2rad(x)
    ry=np.deg2rad(y)
    rz=np.deg2rad(z)
    cx,sx=np.cos(rx),np.sin(rx)
    cy,sy=np.cos(ry),np.sin(ry)
    cz,sz=np.cos(rz),np.sin(rz)
    Rx=np.matrix([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry=np.matrix([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz=np.matrix([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rx,Ry,Rz

def getHomography(h_norm_pw,h_norm_pi):
    #ignore z axis in the world coordinates because it is 0
    h_xy_norm_pw=np.matrix(h_norm_pw[[0,2,3],:])
    #define matrix A where Ah=0
    A=np.zeros((h_norm_pw.shape[1]*2,9))
    
    for i in range(h_norm_pw.shape[1]):
        idx=i*2
        A[idx  ,[3,4,5]] =-h_norm_pi[2,i] * h_xy_norm_pw[:,i].getT() #-w'Xt
        A[idx  ,[6,7,8]] = h_norm_pi[1,i] * h_xy_norm_pw[:,i].getT() # y'Xt
        A[idx+1,[0,1,2]] = h_norm_pi[2,i] * h_xy_norm_pw[:,i].getT() # w'Xt
        A[idx+1,[6,7,8]] =-h_norm_pi[0,i] * h_xy_norm_pw[:,i].getT() #-x'Xt
    
    #get svd of A
    u,s,v=np.linalg.svd(A)
    u=np.matrix(u)
    v=np.matrix(v)
    vt=v.getT()
    h=vt[:,vt.shape[1]-1]
    return h


def findHomography(pw,pi,numpt=4,inlier_threshold=3,db_msg=1):
    '''
    this function returns homography matrix which explains pi=H*pw
    param[in] pw: world 3d coloumn vector [x,y,z]t
    param[in] pi: 2d coloumn vector
    param[in] k:  camera intrinsic parameter
    param[in] numpt: number of sample to get init homography using ransac
    param[in] inlier_threshold: reprojection threshold
    return H 3x3 homography matrix
    '''
    best_H=[]
    best_L2Error=0
    final_outlier_idx=[]
    final_inlier_idx=[]
    best_L2Error_Vec=[]
    best_inlier_num=0
    best_col=[]
    best_h=[]

    idx=np.arange(pw.shape[1])
    for i in range(500):
        np.random.shuffle(idx)
        col=idx[np.arange(numpt)]
        #col=np.array([1,2,4,5])
        #copy sample point
        s_pw=np.matrix(pw[:,col])
        s_pi=np.matrix(pi[:,col])
        '''
        #normalize points
        norm_pw,mean_pw,scale_fw=normalization(s_pw)
        if (norm_pw is None) or (mean_pw is None) or (scale_fw is None):
            #print("viside by 0")
            continue
        norm_pi,mean_pi,scale_fi=normalization(s_pi)
        if (norm_pi is None) or (mean_pi is None) or (scale_fi is None):
            #print("viside by 0")
            continue
        #make normalization matrix
        pdb.set_trace()
        norm_mat_w=np.matrix([[scale_fw,0,-mean_pw[0]*scale_fw],[0,scale_fw,-scale_fw*mean_pw[1]],[0,0,1]])
        norm_mat_i=np.matrix([[scale_fi,0,-mean_pi[0]*scale_fi],[0,scale_fi,-scale_fi*mean_pi[1]],[0,0,1]])
        '''
        
        #represents homogeneous coordinates
        h_norm_pw=toHomo(s_pw)
        h_norm_pi=toHomo(s_pi)
        
        if isInGeneralPosition(h_norm_pw[[0,2,3],:]):
           # print("points are in general position\n",col)
            continue
        if isInGeneralPosition(h_norm_pi):
            #print("points are in general position\n")
            continue
        #get homography
        if False:
            h_norm_pw=toHomo(s_pw)
            h_norm_pi=toHomo(s_pi)
    
        h=getHomography(h_norm_pw,h_norm_pi)
        #denormalization h
        H=np.matrix(h.reshape(3,3))
#        H=norm_mat_i.getI()*H*norm_mat_w
    
        #change all point to homogeneous coordinate to do validation
        h_pw=toHomo(pw)
        h_pi=toHomo(pi)

        #check reprojection error x'=Hx
        repro_pi=H*h_pw[[0,2,3],:]
        repro_pi=repro_pi/repro_pi[2,:]
        error_vec=repro_pi-h_pi
        #error distance
        ET_E=error_vec.getT()*error_vec

        L2Error=np.sqrt(np.diag(ET_E))
        inlier_idx=np.matrix(list(np.where(L2Error<=inlier_threshold)),dtype=np.int32)
        outlier_idx=np.matrix(list(np.where(L2Error>inlier_threshold)),dtype=np.int32)
        num_inlier=np.transpose(np.where(L2Error<inlier_threshold)).shape[0]
        inlier_L2Error=np.sum(L2Error[inlier_idx])
        outlier_L2Error=np.sum(L2Error[outlier_idx])
      #  print("num_inlier:", num_inlier)
      #  print("inlier_L2Error:", inlier_L2Error)
      #  print("outlier_L2Error:", outlier_L2Error)
        if num_inlier>=3 and ((best_inlier_num < num_inlier) or ((best_inlier_num==num_inlier) and (inlier_L2Error <=best_L2Error))):
            best_h=h
            best_inlier_num=num_inlier
            best_L2Error_Vec=L2Error
            final_inlier_idx=inlier_idx
            final_outlier_idx=outlier_idx
            best_col=col
            best_H=H
            best_inlier_L2Error=inlier_L2Error
    final_inlier_idx=convertMat2Array(final_inlier_idx)
    if db_msg:
        print("\n\n\nbefore fit model using inlier")
        printDebugInfo(pw,pi,best_H,best_inlier_num,final_inlier_idx, final_outlier_idx, best_L2Error_Vec)
#    print("H", best_H)
#    print("h\n", best_h)
#    pdb.set_trace()
#    return best_H, best_inlier_num,final_inlier_idx,final_outlier_idx, norm_mat_w, norm_mat_i 
    #fit model using inlier

    #norm_pw,mean_pw,scale_fw=normalization(np.matrix(pw[:,final_inlier_idx]))
    #norm_pi,mean_pi,scale_fi=normalization(np.matrix(pi[:,final_inlier_idx]))

    #norm_mat_w=np.matrix([[scale_fw,0,-mean_pw[0]*scale_fw],[0,scale_fw,-scale_fw*mean_pw[1]],[0,0,1]])
    #norm_mat_i=np.matrix([[scale_fi,0,-mean_pi[0]*scale_fi],[0,scale_fi,-scale_fi*mean_pi[1]],[0,0,1]])
    try: 
        h_norm_pw=toHomo(np.matrix(pw[:,final_inlier_idx]))
        h_norm_pi=toHomo(np.matrix(pi[:,final_inlier_idx]))
    except IndexError as e:
        pdb.set_trace()

    if isInGeneralPosition(h_norm_pw[[0,2,3],:]):
        print(" %d of %d points are in general points"%(best_inlier_num-1,best_inlier_num))
        quit()
    if isInGeneralPosition(h_norm_pi):
        print(" %d of %d points are in general points"%(best_inlier_num-1,best_inlier_num))
        quit()

    h=getHomography(h_norm_pw,h_norm_pi)
    H=np.matrix(h.reshape(3,3))
#    H=norm_mat_i.getI()*H*norm_mat_w

    #reprojection test
    h_pw=toHomo(pw)
    h_pi=toHomo(pi)
    repro_pi=H*h_pw[[0,2,3],:]
    repro_pi=repro_pi/repro_pi[2,:]
    error_vec=repro_pi-h_pi
    #error distance
    L2Error_vec=np.diag(error_vec.getT()*error_vec)
    L2Error_vec=np.sqrt(L2Error_vec)
    
    return H, best_inlier_num,final_inlier_idx,final_outlier_idx, L2Error_vec 
#    inlier_idx=np.matrix(list(np.where(L2Error_vec<=inlier_threshold)),dtype=np.int32).A.flatten()
#    outlier_idx=np.matrix(list(np.where(L2Error_vec>inlier_threshold)),dtype=np.int32).A.flatten()
#    num_inlier=inlier_idx.shape[0]
#    pdb.set_trace()
#    if db_msg:
#        print("\n\n\nafter fit model using inlier")
#        printDebugInfo(pw,pi,H,num_inlier,inlier_idx,outlier_idx,L2Error_vec)
#    return H, num_inlier, inlier_idx, outlier_idx, L2Error_vec

def printDebugInfo(pw,pi,H,inlier_num, inlier_idx, outlier_idx,L2Error_vec):
    
    print("num_inlier_pt=", inlier_num)
    print("final_inlier_idx=\n", inlier_idx)

    print("mean_inlier_L2Error_pt=", np.mean(L2Error_vec[inlier_idx]))
    print("inlier_L2Error_list\n", L2Error_vec[inlier_idx])
    
    print("outlier L2Error=", np.mean(L2Error_vec[outlier_idx]))
    print("outlier_L2Error_list\n", L2Error_vec[outlier_idx])

    #print("in_pw=\n", pw[:,inlier_idx])
    #print("in_pw=\n", pi[:,inlier_idx])
    h_pw=toHomo(pw[:,:])
    h_pi=toHomo(pi[:,:])
    repro_pi=H*h_pw[[0,1,3],:]
    repro_pi=repro_pi/repro_pi[2,:]
    print("all point", repro_pi)
    print("pi=", h_pi)

    
def optimize_extrinsic_NewthonMethod(pw,pi,H):
    '''
    this function optimize extrinsic parameter using newton method
    pw=3x1 homogenous coordinates(z axis must be ignored)
    pi=3x1 homogenous coordinates
    H= DLT result homography
    
    '''
    pw=np.matrix(pw)
    pi=np.matrix(pi)
    H=np.matrix(H)
    min_error=0
    error_list=[]

    ppi=H*pw       #ppi is projected pi 
    ppi=ppi/ppi[2,:]
    error=ppi-pi # e0=Hx-x'
    min_error=getMeanError(error)
    error_list=[min_error]
    learning_rate=0.01
    cnt=0
    for i in range(100):
        J=np.matrix(np.zeros((3,9)))
        tmp_h=np.matrix(np.zeros((9,1)))
        for i in range(error.shape[1]):
            J[0,[0,1,2]]=pw[:,i].getT()
            J[1,[3,4,5]]=pw[:,i].getT()
            J[2,[6,7,8]]=pw[:,i].getT()
            u,s,v=np.linalg.svd(J,full_matrices=False)
            u=np.matrix(u)
            d=np.matrix(np.diag(s))
            v=np.matrix(v)
            inv_J=v.getI()*d.getI()*u.getI()
            delta = -inv_J*error[:,i]
            tmp_h = tmp_h+learning_rate*delta

        tmp_H=H+tmp_h.reshape(3,3)
        ppi=tmp_H*pw       #ppi is projected pi 
        ppi=ppi/ppi[2,:]
        tmp_error_vec=ppi-pi # e0=Hx-x'
        tmp_e=getMeanError(tmp_error_vec)
        if tmp_e < min_error:
            min_error=tmp_e
            error=tmp_error_vec
            error_list.append(tmp_e)
            H=tmp_H
        
        else:
            learning_rate= learning_rate/2
    return H

def getMeanError(error):
    squared_error=np.diag(error.getT()*error)
   # print("mean_squared_error=", np.mean(squared_error))
    return np.mean(squared_error)

def isInGeneralPosition(pt, draw_flag=False, iter_num=20):
    '''
    pt: should be homogenious coordinate whether it is world point or image coordinates

    '''
    num_pt=pt.shape[1] #0,1,2,3,....,n
    idx=np.arange(num_pt)
    for i in range(iter_num):
        np.random.shuffle(idx)
        col=idx[0:num_pt-1]
        u,s,v=np.linalg.svd(pt[:,col].getT())
        u=np.matrix(u)
        v=np.matrix(v)
        #draw fitted line
#        A=v.getT()[:,np.argmin(s)]
        A=v.getT()[:,v.getT().shape[1]-1]
        
        with np.errstate(divide='raise'):
            try :
                A=A/(-A[1,0])
            except FloatingPointError:
                print("func \"isInGeneralPosition()\" divide by 0")
                return True
            except :
                print("something wrong")
                return True
    
        diff_y=A[0,0]*pt[0,:] + A[1,0]*pt[1,:]+A[2,0] 
        dist_vec=np.absolute(diff_y)
        num_inlier=dist_vec[np.where(dist_vec<2)].A.flatten().shape[0]
        try:
           # print(dist_vec<0.5)
            #print(np.where(dist_vec<0.5))
            mean_dist=np.mean(dist_vec)
        except:
         #   pdb.set_trace()
            continue
        if num_inlier >= (num_pt-1):
            print("num_inlier, num_pt-1 :", num_inlier, num_pt-1)
            if draw_flag:
                x=np.arange(np.min(pt[0,:]),np.max(pt[0,:]),1)
                y=A[0,0]*x +A[2,0]
            return True
    return False

def convertMat2Array(mat):
    if isinstance(mat,np.matrixlib.defmatrix.matrix):
        return mat.A.flatten()
    else:
        return np.asarray(mat,dtype=np.float64)
