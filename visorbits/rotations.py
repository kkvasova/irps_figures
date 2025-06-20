'''
    rotations.py

    This file containts functions to create roational matrices.
    -------------------------------------------------------------------------------------------------------
    rot_matrix  :   simple rotational matrix (axis, angle, dimension)
    vect_m      :   vector product
    reverse_tr  :   for given point and vectortau, with phase angle phi4, create 4 consequent rotational matrices
                    to align final X axis with point and Y axis in vectortau direction. The point than is shifted
                    that its positive position angle is phi4.
    -------------------------------------------------------------------------------------------------------
    NOTE:       For 3 dimensions. 
    -------------------------------------------------------------------------------------------------------
'''

import numpy as np

def rot_matrix(phi, ax=0, dim=3):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    phi      :   angle of rotatione (counter clockwise if coordinates;
                 clockwise if vector)
    ax       :   around which axis to rotate
    dim      :   dimension (rank) of roational matrix
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 np.ndarray (dim, dim)
    ---------------------------------------------------------------------
    NOTE     :   Counterclockwise rot per pi of coord.sys.
    ---------------------------------------------------------------------
    '''
    init = np.zeros((dim,dim))
    for i in np.asarray(range(dim)):
        for j in np.asarray(range(dim)):
            if i!= j:
                init[i][j] = np.sin(phi)* ((j-i)/abs(i-j))*(-1)**(i+j)
            else: init[i][j] = np.cos(phi)

    for i in np.asarray(range(dim)):
        for j in np.asarray(range(dim)):
            if i==ax or j==ax:
                init[i][j] = 0.
            if i==j==ax:
                init[i][j] = 1
    return init

def vect_m(a, b):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    a        :   first vector
    b        :   second vector
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 [axb] vector roduct (only (3,))
    ---------------------------------------------------------------------
    NOTE     :
    ---------------------------------------------------------------------
    '''
    return np.array([a[1]*b[2]-a[2]*b[1],\
                  -1*a[0]*b[2]+a[2]*b[0],\
                     a[0]*b[1]-a[1]*b[0]])
                     
                     
def reverse_tr(x, y, z, vtau, phi4):
    '''
    ---------------------------------------------------------------------
    Arguments:
    --------------------------------------------------------------------
    x,y,z    :   coordinates of the point from the final plane
                 (final X is alignted with the radvect of point)
    vtau     :   direction of point motion; final Y
    phi4     :   positional angle of point in a final plane
                 (counterclockwise from original X0 (1, 0, 0))
    ---------------------------------------------------------------------
    Output:
    ---------------------------------------------------------------------
                 r1, r2, r3, r4 - four np.ndarray (dim, dim)
                 rotational matrices 
    ---------------------------------------------------------------------
    NOTE     :
    ---------------------------------------------------------------------
    ''' 
    p0 = np.array([x,y,z])
    v0 = vtau    
    alpha = np.arctan(abs(p0[1]/p0[0]))
    if p0[0]<0:
        phi1 = np.pi+alpha   if p0[1]<0 else np.pi-alpha
    else:
        phi1 = 2*np.pi-alpha if p0[1]<0 else alpha
    r1 = rot_matrix(phi1, ax=2)

    p1 = np.dot(rot_matrix(-1*phi1, ax=2), p0)
    v1 = np.dot(rot_matrix(-1*phi1, ax=2), v0)
    
    #print('Updated_point: ', p1)
    beta  = np.arctan(abs(p1[2]/p1[0]))

    if p1[0]<0:
        phi2 = np.pi+beta    if p1[2]>0 else np.pi-beta
    else:
        phi2 = 2*np.pi-beta  if p1[2]>0 else beta
    r2 = rot_matrix(phi2, ax=1)

    p2 = np.dot(rot_matrix(-1*phi2, ax=1), p1)
    v2 = np.dot(rot_matrix(-1*phi2, ax=1), v1)    
    
    #print('Vtau_test: ', v2)
    gamma = np.arctan(abs(v2[2]/v2[1]))
    
    if v2[1]<0: # updated y
        phi3 = np.pi+gamma   if v2[2]<0 else np.pi-gamma
    else:
        phi3 = 2*np.pi-gamma if v2[2]<0 else gamma

    r3 = rot_matrix(phi3, ax=0)
    r4 = rot_matrix(phi4*(-1), ax=2) # <-> phase angle: place pericenter phi4 behind the ch. point with the phase angle phi4.
    
    #print(alpha*180/np.pi, beta*180/np.pi, gamma*180/np.pi, phi1*180/np.pi, phi2*180/np.pi,phi3*180/np.pi,phi4*180/np.pi)
    
    return r1, r2, r3, r4
