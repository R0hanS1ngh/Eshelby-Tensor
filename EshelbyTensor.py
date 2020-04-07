import sympy as sym
from sympy.abc import x,y,z,phi,theta
import numpy as np
from scipy import integrate

def generateTensor(A):
    for j in range(0, 2):
        for k in range(j + 1, 3):
            A[j][j][k][j] = A[j][j][j][k]
            A[j][k][j][j] = A[j][j][j][k]
            A[k][j][j][j] = A[j][j][j][k]

            A[k][j][k][k] = A[j][k][k][k]
            A[k][k][j][k] = A[j][k][k][k]
            L[k][k][k][j] = L[j][k][k][k]

            A[k][k][j][j] = A[j][j][k][k]

            A[j][k][k][j] = A[j][k][j][k]
            A[k][j][j][k] = A[j][k][j][k]
            A[k][j][k][j] = A[j][k][j][k]

    A[0][0][2][1] = A[0][0][1][2]
    A[1][2][0][0] = A[0][0][1][2]
    A[2][1][0][0] = A[0][0][1][2]

    A[2][0][1][1] = A[0][2][1][1]
    A[1][1][0][2] = A[0][2][1][1]
    A[1][1][2][0] = A[0][2][1][1]

    A[1][0][2][2] = A[0][1][2][2]
    A[2][2][0][1] = A[0][1][2][2]
    A[2][2][1][0] = A[0][1][2][2]

    for n in range(0, 2):
        for m in range(1, n + 2):
            A[0][m][2][n] = A[0][m][n][2]
            A[m][0][n][2] = A[0][m][n][2]
            A[m][0][2][n] = A[0][m][n][2]
            A[n][2][0][m] = A[0][m][n][2]
            A[n][2][m][0] = A[0][m][n][2]
            A[2][n][0][m] = A[0][m][n][2]
            A[2][n][m][0] = A[0][m][n][2]
    return A

def det(a):
    return (sym.expand((a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
           -a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
           +a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]))))

def integration(B,a1,a2,a3,i,j,k,l):
    val, err = integrate.dblquad(sym.lambdify([phi, theta], B[i][j][k][l], 'numpy'), -np.pi / 2, np.pi / 2, lambda phi: -np.pi, lambda phi: np.pi)
    a= np.sqrt(a1**2+a2**2+a3**2)
    return val * a1 * a2 * a3 / ((a ** 3) * 2 * np.pi)

def inv(L):
    Lopen = np.ones([9, 9])
    for i in range(0, 3):
        for j in range(0, 3):
            Lopen[i][j] = L[0][i][0][j]
            Lopen[i][j + 3] = L[0][i][1][j]
            Lopen[i][j + 6] = L[0][i][2][j]
            Lopen[i + 3][j] = L[1][i][0][j]
            Lopen[i + 3][j + 3] = L[1][i][1][j]
            Lopen[i + 3][j + 6] = L[1][i][2][j]
            Lopen[i + 6][j] = L[2][i][0][j]
            Lopen[i + 6][j + 3] = L[2][i][1][j]
            Lopen[i + 6][j + 6] = L[2][i][2][j]
    delta = np.eye(3)
    I = np.ones([3, 3, 3, 3])
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                for l in range(0, 3):
                    I[i][j][k][l] = delta[i][l] * delta[j][k]
    Iopen = np.ones([9, 9])
    for i in range(0, 3):
        for j in range(0, 3):
            Iopen[i][j] = I[0][i][0][j]
            Iopen[i][j + 3] = I[0][i][1][j]
            Iopen[i][j + 6] = I[0][i][2][j]
            Iopen[i + 3][j] = I[1][i][0][j]
            Iopen[i + 3][j + 3] = I[1][i][1][j]
            Iopen[i + 3][j + 6] = I[1][i][2][j]
            Iopen[i + 6][j] = I[2][i][0][j]
            Iopen[i + 6][j + 3] = I[2][i][1][j]
            Iopen[i + 6][j + 6] = I[2][i][2][j]
    Linvopen1 = np.linalg.pinv(Lopen)
    Linvopen2 = np.matmul(Linvopen1, Iopen)
    Linv = np.ones([3, 3, 3, 3])
    for i in range(0, 3):
        for j in range(0, 3):
            Linv[0][i][0][j] = Linvopen2[i][j]
            Linv[0][i][1][j] = Linvopen2[i][j + 3]
            Linv[0][i][2][j] = Linvopen2[i][j + 6]
            Linv[1][i][0][j] = Linvopen2[i + 3][j]
            Linv[1][i][1][j] = Linvopen2[i + 3][j + 3]
            Linv[1][i][2][j] = Linvopen2[i + 3][j + 6]
            Linv[2][i][0][j] = Linvopen2[i + 6][j]
            Linv[2][i][1][j] = Linvopen2[i + 6][j + 3]
            Linv[2][i][2][j] = Linvopen2[i + 6][j + 6]
    return Linv

val=int(input("State the type of stiffness tensor: Triclinic(1), Cubic(2), Isotropic(3): "))

W= np.ones([21])

if val== 1:
    print('Enter Stiffness inputs in order: L1111, L1112, L1212, L1113, L1213, L1313, L1122, L1222, L1322, L2222, L1123, L1223, L1323, L2223, L2323, L1133, L1233, L1333, L2233, L2333, L3333')
    for i in range (0,21):
        print('Stiffness input',i+1,' (in Pa): ')
        W[i]= float(input())
elif val== 2:
    for i in range (0,21):
        W[i]= 0.0
    W[0]= W[2]= W[5]= float(input("L1111 (in Pa): "))
    W[1]= W[3]= W[4]= float(input("L1122 (in Pa): "))
    W[9]= W[14]= W[20]= float(input("L1212 (in Pa): "))
elif val== 3:
    for i in range (0,21):
        W[i]=0.0
    E= float(input("E (in Pa): "))
    nu= float(input("Poisson's ratio: "))
    omega= E*nu/((1+nu)*(1-2*nu))
    mu= E/(2*(1+nu))
    W[0] = W[2] = W[5] = omega+ 2*mu
    W[1] = W[3] = W[4] = omega
    W[9] = W[14] = W[20] = mu

val1= int(input("Enter the type of inclusion: Ellipsoidal(1), Spherical(2), Elliptical Flat Plate(3), Circular Flat Plate(4), Right Elliptical Cylinder(5), Right Circular Cylinder(6): "))
if val1== 1:
    a1= float(input("a1: "))
    a2= float(input("a2: "))
    a3= float(input("a3: "))
elif val1 == 2:
    a1= a2= a3= float(input("r: "))
elif val1 == 3:
    a1= float(input("Enter Plate Thickness { << other dimensions} : "))
    a2= float(input("a: "))
    a3= float(input("b: "))
elif val1 == 4:
    a1= float(input("Enter Plate Thickness { << other dimensions} : "))
    a2= a3= float(input("r: "))
elif val1 == 5:
    a1= float(input("Enter Cylinder Length { >> other dimensions} : "))
    a2= float(input("a: "))
    a3= float(input("b: "))
elif val1 == 6:
    a1= float(input("Enter Cylinder Length { >> other dimensions} : "))
    a2= a3= float(input("r: "))

print('Euler/ Tait-Bryan Angles (z-x-z/3-1-3 type): ')
e1= float(input("Alpha: "))
e2= float(input("Beta: "))
e3= float(input("Gamma: "))

L= np.ones([3,3,3,3])

L[0][0][0][0]=W[0]
L[0][0][0][1]=W[1]
L[0][1][0][1]=W[2]
L[0][0][0][2]=W[3]
L[0][1][0][2]=W[4]
L[0][2][0][2]=W[5]
L[0][0][1][1]=W[6]
L[0][1][1][1]=W[7]
L[0][2][1][1]=W[8]
L[1][1][1][1]=W[9]
L[0][0][1][2]=W[10]
L[0][1][1][2]=W[11]
L[0][2][1][2]=W[12]
L[1][1][1][2]=W[13]
L[1][2][1][2]=W[14]
L[0][0][2][2]=W[15]
L[0][1][2][2]=W[16]
L[0][2][2][2]=W[17]
L[1][1][2][2]=W[18]
L[1][2][2][2]=W[19]
L[2][2][2][2]=W[20]

generateTensor(L)
print("L=", L)

x= (a1*sym.cos(phi)*sym.cos(theta)*np.cos(e1)-a2*sym.cos(phi)*sym.sin(theta)*np.sin(e1))*np.cos(e3)-((a1*sym.cos(phi)*sym.cos(theta)*np.sin(e1)+a2*sym.cos(phi)*sym.sin(theta)*np.cos(e1))*np.cos(e2)-a3*sym.sin(phi)*np.sin(e2))*np.sin(e3)
y= (a1*sym.cos(phi)*sym.cos(theta)*np.cos(e1)-a2*sym.cos(phi)*sym.sin(theta)*np.sin(e1))*np.sin(e3)+((a1*sym.cos(phi)*sym.cos(theta)*np.sin(e1)+a2*sym.cos(phi)*sym.sin(theta)*np.cos(e1))*np.cos(e2)-a3*sym.sin(phi)*np.sin(e2))*np.cos(e3)
z= (a1*sym.cos(phi)*sym.cos(theta)*np.sin(e1)+a2*sym.cos(phi)*sym.sin(theta)*np.cos(e1))*np.sin(e2)+a3*sym.sin(phi)*np.cos(e2)

U=np.array([x,y,z])

K= np.array([[y,y,y],[x,x,x],[z,z,z]])
Q= np.array([[y,y,y],[x,x,x],[z,z,z]])

for i in range(0,3):
    for k in range(0,3):
        K[i][k]= L[i][0][k][0]*U[0]*U[0]+L[i][0][k][1]*U[0]*U[1]+L[i][0][k][2]*U[0]*U[2]+L[i][1][k][0]*U[1]*U[0]+L[i][1][k][1]*U[1]*U[1]+L[i][1][k][2]*U[1]*U[2]+L[i][2][k][0]*U[2]*U[0]+L[i][2][k][1]*U[2]*U[1]+L[i][2][k][2]*U[2]*U[2]

b= det(K)

N= np.array([[y, y, y], [x, x, x], [z, z, z]])
N[0][0]=sym.expand(K[1][1]*K[2][2]-K[1][2]*K[2][1])
N[0][1]=sym.expand(-(K[1][0]*K[2][2]-K[1][2]*K[2][0]))
N[0][2]=sym.expand(K[1][0]*K[2][1]-K[1][1]*K[2][0])
N[1][0]=sym.expand(-(K[0][1]*K[2][2]-K[0][2]*K[2][1]))
N[1][1]=sym.expand(K[0][0]*K[2][2]-K[0][2]*K[2][0])
N[1][2]=sym.expand(-(K[0][0]*K[2][1]-K[0][1]*K[2][0]))
N[2][0]=sym.expand(K[0][1]*K[1][2]-K[0][2]*K[1][1])
N[2][1]=sym.expand(-(K[0][0]*K[1][2]-K[1][0]*K[0][2]))
N[2][2]=sym.expand(K[0][0]*K[1][1]-K[1][0]*K[0][1])

H= np.array([[[[y,y,y],[x,x,x],[z,z,z]],[[y,y,y],[x,x,x],[z,z,z]],[[y,y,y],[x,x,x],[z,z,z]]],[[[y,y,y],[x,x,x],[z,z,z]],[[y,y,y],[x,x,x],[z,z,z]],[[y,y,y],[x,x,x],[z,z,z]]],[[[y,y,y],[x,x,x],[z,z,z]],[[y,y,y],[x,x,x],[z,z,z]],[[y,y,y],[x,x,x],[z,z,z]]]])

for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                H[i][j][k][l]= (N[i][k]*U[j]*U[l] + N[j][k]*U[i]*U[l] + N[i][l]*U[j]*U[k] + N[j][l]*U[i]*U[k])/b

P= np.ones([3,3,3,3])

P[0][0][0][0]= integration(H,a1,a2,a3,0,0,0,0)
P[1][1][1][1]= integration(H,a1,a2,a3,1,1,1,1)
P[2][2][2][2]= integration(H,a1,a2,a3,2,2,2,2)

P[0][1][0][1]= integration(H,a1,a2,a3,0,1,0,1)
P[0][2][0][2]= integration(H,a1,a2,a3,0,2,0,2)
P[0][0][1][1]= integration(H,a1,a2,a3,0,0,1,1)
P[1][2][1][2]= integration(H,a1,a2,a3,1,2,1,2)
P[0][0][2][2]= integration(H,a1,a2,a3,0,0,2,2)
P[1][1][2][2]= integration(H,a1,a2,a3,1,1,2,2)

P[0][0][0][1]= integration(H,a1,a2,a3,0,0,0,1)
P[0][0][0][2]= integration(H,a1,a2,a3,0,0,0,2)
P[0][1][1][1]= integration(H,a1,a2,a3,0,1,1,1)
P[1][1][1][2]= integration(H,a1,a2,a3,1,1,1,2)
P[0][2][2][2]= integration(H,a1,a2,a3,0,2,2,2)
P[1][2][2][2]= integration(H,a1,a2,a3,1,2,2,2)

P[0][1][0][2]= integration(H,a1,a2,a3,0,1,0,2)
P[0][2][1][1]= integration(H,a1,a2,a3,0,2,1,1)
P[0][0][1][2]= integration(H,a1,a2,a3,0,0,1,2)
P[0][1][1][2]= integration(H,a1,a2,a3,0,1,1,2)
P[0][2][1][2]= integration(H,a1,a2,a3,0,2,1,2)
P[0][1][2][2]= integration(H,a1,a2,a3,0,1,2,2)

generateTensor(P)

S= np.einsum('ijkl,klpq->ijpq',P,L)

for i in range(0,3):
    for j in range(i,3):
        for k in range(0,3):
            for l in range(k,3):
                print('S', i + 1, j + 1, k + 1, l + 1, '=', S[i][j][k][l])

print('S=',S) # Eshelby Tensor

delta = np.eye(3)
I = np.ones([3, 3, 3, 3])
for i in range(0, 3):
    for j in range(0, 3):
        for k in range(0, 3):
            for l in range(0, 3):
                I[i][j][k][l] = delta[i][l] * delta[j][k]

val2= int(input("State the type of inhomogeneity stiffness tensor: Triclinic(1), Cubic(2), Isotropic(3): "))
Y= np.ones([21])
if val2== 1:
    for i in range (0,21):
        print('Stiffness input',i+1,':')
        Y[i]= float(input())
elif val2== 2:
    for i in range (0,21):
        Y[i]= 0.0
    Y[0]= Y[2]= Y[5]= float(input("L1111:"))
    Y[1]= Y[3]= Y[4]= float(input("L1122:"))
    Y[9]= Y[14]= Y[20]= float(input("L1212:"))
elif val2== 3:
    for i in range (0,21):
        Y[i]=0.0
    E= float(input("E:"))
    nu= float(input("Poisson's ratio:"))
    omega= E*nu/((1+nu)*(1-2*nu))
    mu= E/(2*(1+nu))
    Y[0] = Y[2] = Y[5] = omega+ 2*mu
    Y[1] = Y[3] = Y[4] = omega
    Y[9] = Y[14] = Y[20] = mu

L1= np.ones([3,3,3,3])
L1[0][0][0][0]=Y[0]
L1[0][0][0][1]=Y[1]
L1[0][1][0][1]=Y[2]
L1[0][0][0][2]=Y[3]
L1[0][1][0][2]=Y[4]
L1[0][2][0][2]=Y[5]
L1[0][0][1][1]=Y[6]
L1[0][1][1][1]=Y[7]
L1[0][2][1][1]=Y[8]
L1[1][1][1][1]=Y[9]
L1[0][0][1][2]=Y[10]
L1[0][1][1][2]=Y[11]
L1[0][2][1][2]=Y[12]
L1[1][1][1][2]=Y[13]
L1[1][2][1][2]=Y[14]
L1[0][0][2][2]=Y[15]
L1[0][1][2][2]=Y[16]
L1[0][2][2][2]=Y[17]
L1[1][1][2][2]=Y[18]
L1[1][2][2][2]=Y[19]
L1[2][2][2][2]=Y[20]

generateTensor(L1)
print("L1=", L1)

L2= L-L1
Linv= inv(L)
Z= np.einsum('ijkl,klpq,pqrs->ijrs',S,Linv,L2)

T= inv(Z+I)
print('T=',T)
