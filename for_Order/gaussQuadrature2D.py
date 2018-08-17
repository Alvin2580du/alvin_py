import pandas as pd
import numpy as np
import scipy

local_data_set1 = pd.read_csv('local_3_40.dat_set1.csv')
local_data_set2 = pd.read_csv('local_3_40.dat_set2.csv')
local_data_set3 = pd.read_csv('local_3_40.dat_set3.csv')

local_data_set12 = pd.read_csv('local_3_40.dat_set12.csv')


def find(v, l):
    k = 0
    for x in l:
        k += 1
        if x == v:
            return k


local_zhi = local_data_set12.head(6)
local_shun_coord1 = local_zhi.values
quan_local_shun_pe1 = np.zeros((len(local_shun_coord1), 8))
print(quan_local_shun_pe1.shape)
for ii in range(1, 6):
    if ii == 1:
        number_local_shun1 = len(local_shun_coord1)
        quan_local_shun_pe1 = np.zeros((number_local_shun1, 8))

    if ii == 2:
        local_shun_coord2 = local_shun_coord1
        number_local_shun2 = len(local_shun_coord2)
        quan_local_shun_pe2 = np.zeros((number_local_shun2, 8))

    if ii == 3:
        local_wen_coord = local_shun_coord1
        number_local_wen = len(local_wen_coord)
        quan_local_wen_pe = np.zeros((number_local_wen, 8))

    if ii == 4:
        local_zhi = local_data_set3.head(6)
        local_shun_pe1 = local_zhi.values
        length = len(local_shun_pe1)
        for mm in range(length):
            hangshu = find(local_shun_pe1[mm, 0], local_shun_coord1[:, 0])
            hangshu1 = hangshu + local_shun_pe1[mm, 1] - 1
            if hangshu1 < 6:
                quan_local_shun_pe1[hangshu1, 3:8] = local_shun_pe1[mm, 3:8]
    if ii == 5:
        local_zhi = local_data_set3.head(6)
        local_shun_pe2 = local_zhi.values
        length = len(local_shun_pe2)

        for mm in range(length):
            hangshu = find(local_shun_pe1[mm, 0], local_shun_coord1[:, 0])
            hangshu1 = hangshu + local_shun_pe1[mm, 1] - 1
            if hangshu1 < 6:
                quan_local_shun_pe2 = quan_local_shun_pe1
                quan_local_shun_pe2[hangshu1, 3:8] = local_shun_pe1[mm, 3:8]

    if ii == 6:
        local_zhi = local_data_set3.head(6)
        local_wen_pe = local_zhi.values
        length = len(local_wen_pe)

        for mm in range(length):
            local_wen_coord = local_shun_coord1
            hangshu = find(local_wen_pe[mm, 0], local_wen_coord[:, 0])
            hangshu1 = hangshu + local_shun_pe1[mm, 1] - 1
            if hangshu1 < 6:
                quan_local_wen_pe = quan_local_shun_pe1
                quan_local_wen_pe[hangshu1, 3:8] = local_wen_pe[mm, 3:8]

glob_file = 'global-3-40.dat.csv'
global_zhi = pd.read_csv(glob_file)
global_shun_coord1 = global_zhi.values

for jj in range(1, 4):
    if jj == 1:
        global_shun_coord1 = global_zhi.values
    if jj == 2:
        global_shun_coord2 = global_zhi.values
    if jj == 3:
        global_shun_coord3 = global_zhi.values


def norm(a):
    return (a - np.min(a)) / (np.max(a) - np.min(a))


origin = [0, 0, 0]
x = [1, 0, 0]
y = [0, 1, 0]
z = [0, 0, 1]
x_unit = (np.array(x) - np.array(origin)) / norm((np.array(x) - np.array(origin)))
y_unit = (np.array(y) - np.array(origin)) / norm((np.array(y) - np.array(origin)))
z_unit = (np.array(z) - np.array(origin)) / norm((np.array(z) - np.array(origin)))
R_gl = [np.transpose(x_unit), np.transpose(y_unit), np.transpose(z_unit)]
R_lg = np.array(np.transpose(R_gl))

glob_zuobiao_shun1 = global_shun_coord1[:, 3: 5]
glob_local_shun1 = np.zeros((len(glob_zuobiao_shun1[:, 1]), 5))
for kk in range(1, len(glob_zuobiao_shun1[:, 1])):
    glob_local_shun1[kk, 3:5] = R_lg * np.transpose(glob_zuobiao_shun1[kk, :]) - R_lg * np.transpose(origin)
    glob_local_shun1[kk, 1: 2] = global_shun_coord1[kk, 1: 2]

glob_zuobiao_shun2 = global_shun_coord1[:, 3: 5]
glob_local_shun2 = np.zeros((len(glob_zuobiao_shun2[:, 1]), 5))
glob_local_shun1 = np.zeros((len(glob_zuobiao_shun2[:, 1]), 5))
for kk in range(1, len(glob_zuobiao_shun1[:, 1])):
    glob_local_shun2[kk, 3:5] = R_lg * np.transpose(glob_zuobiao_shun2[kk, :]) - R_lg * np.transpose(origin)
    glob_local_shun2[kk, 1: 2] = global_shun_coord1[kk, 1: 2]

glob_zuobiao_shun1 = global_shun_coord1[:, 3: 5]
glob_local_wen = np.zeros((len(glob_zuobiao_shun1[:, 1]), 5))
for kk in range(1, len(glob_zuobiao_shun1[:, 1])):
    glob_local_wen[kk, 3:5] = R_lg * np.transpose(glob_zuobiao_shun1[kk, :]) - R_lg * np.transpose(origin)
    glob_local_wen[kk, 1: 2] = global_shun_coord1[kk, 1: 2]

G2_L2_juli = [0, 0, 0, -720, 0]

for mm in range(1, len(glob_local_shun2[:, 1])):
    glob_local_shun2[mm, :] = glob_local_shun2[mm, :] + G2_L2_juli

LL = [[-1, 1, 1],
      [-1, -1, 1],
      [-1, 1, -1],
      [-1, -1, -1],
      [1, 1, 1],
      [1, -1, 1],
      [1, 1, -1],
      [1, -1, -1]]

a = 1
b = 1
c = 1

a0 = LL[:, 1] * a
b0 = LL[:, 2] * b
c0 = LL[:, 3] * c
N = 1 / 8 * (1 + a0) * (1 + b0) * (1 + c0)

S0 = [[N(1), 0, 0, N(2), 0, 0, N(3), 0, 0, N(4), 0, 0, N(5), 0, 0, N(6), 0, 0, N(7), 0, 0, N(8), 0, 0],
      [0, N(1), 0, 0, N(2), 0, 0, N(3), 0, 0, N(4), 0, 0, N(5), 0, 0, N(6), 0, 0, N(7), 0, 0, N(8), 0],
      [0, 0, N(1), 0, 0, N(2), 0, 0, N(3), 0, 0, N(4), 0, 0, N(5), 0, 0, N(6), 0, 0, N(7), 0, 0, N(8)]]

S = [[N(1), 0, 0, 0, 0, 0, N(2), 0, 0, 0, 0, 0, N(3), 0, 0, 0, 0, 0, N(4), 0, 0, 0, 0, 0, N(5), 0, 0, 0, 0, 0, N(6), 0,
      0, 0, 0, 0, N(7), 0, 0, 0, 0, 0, N(8), 0, 0, 0, 0, 0],
     [0, N(1), 0, 0, 0, 0, 0, N(2), 0, 0, 0, 0, 0, N(3), 0, 0, 0, 0, 0, N(4), 0, 0, 0, 0, 0, N(5), 0, 0, 0, 0, 0, N(6),
      0, 0, 0, 0, 0, N(7), 0, 0, 0, 0, 0, N(8), 0, 0, 0, 0],
     [0, 0, N(1), 0, 0, 0, 0, 0, N(2), 0, 0, 0, 0, 0, N(3), 0, 0, 0, 0, 0, N(4), 0, 0, 0, 0, 0, N(5), 0, 0, 0, 0, 0,
      N(6), 0, 0, 0, 0, 0, N(7), 0, 0, 0, 0, 0, N(8), 0, 0, 0],
     [0, 0, 0, N(1), 0, 0, 0, 0, 0, N(2), 0, 0, 0, 0, 0, N(3), 0, 0, 0, 0, 0, N(4), 0, 0, 0, 0, 0, N(5), 0, 0, 0, 0, 0,
      N(6), 0, 0, 0, 0, 0, N(7), 0, 0, 0, 0, 0, N(8), 0, 0],
     [0, 0, 0, 0, N(1), 0, 0, 0, 0, 0, N(2), 0, 0, 0, 0, 0, N(3), 0, 0, 0, 0, 0, N(4), 0, 0, 0, 0, 0, N(5), 0, 0, 0, 0,
      0, N(6), 0, 0, 0, 0, 0, N(7), 0, 0, 0, 0, 0, N(8), 0],
     [0, 0, 0, 0, 0, N(1), 0, 0, 0, 0, 0, N(2), 0, 0, 0, 0, 0, N(3), 0, 0, 0, 0, 0, N(4), 0, 0, 0, 0, 0, N(5), 0, 0, 0,
      0, 0, N(6), 0, 0, 0, 0, 0, N(7), 0, 0, 0, 0, 0, N(8)]]


def sort(JL_data):
    return 1, 2


def solve(a, b):
    return scipy.linalg.solve(a, b)

A_gl_t1 = np.zeros((len(global_shun_coord1[:, 1]), 8))

for i in range(1, len(global_shun_coord1[:, 1])):
    d = np.sqrt(np.square((local_shun_coord1[:, 3] - glob_local_shun1[i, 3])) +
                np.square((local_shun_coord1[:, 4] - glob_local_shun1[i, 4])) +
                np.square((local_shun_coord1[:, 5] - glob_local_shun1[i, 5])))
    JL_data = [d]
    u, v = sort(JL_data)
    eight_ip = (np.ceil(v(1) / 8) - 1) * 8 + range(1, 8)
    GL0 = [np.transpose(local_shun_coord1[eight_ip(1), 3:5]),
           np.transpose(local_shun_coord1[eight_ip(2), 3:5]),
           np.transpose(local_shun_coord1[eight_ip(3), 3:5]),
           np.transpose(local_shun_coord1[eight_ip(4), 3:5]),
           np.transpose(local_shun_coord1[eight_ip(5), 3:5]),
           np.transpose(local_shun_coord1[eight_ip(6), 3:5]),
           np.transpose(local_shun_coord1[eight_ip(7), 3:5]),
           np.transpose(local_shun_coord1[eight_ip(8), 3:5])]
    global_local_int = S0 * GL0
    print(glob_local_shun1[i, 3])
    print(global_local_int[0])
    a = np.array([global_local_int[0]) == glob_local_shun1[i, 3],
                 [global_local_int[1] == glob_local_shun1[i, 4]],
                 [global_local_int[2] == glob_local_shun1[i, 5]])

    b = np.array([a, b, c])
    B = solve(a, b)

    #       B1=vpa(B.a);%vpa控制精度
    #       zz=subs(B1,1000);%利用1000代替B1中的默认符号
    #       zzz=abs(zz)-1;%模或者绝对值计算
    #       [zhi,wei]=min(abs(zzz));
    #       B1=real(B1(wei));%实部计算
    #       B1=double(B1);
    #       qqq1(i)=B1

    GL = [np.transpose(quan_local_shun_pe1[eight_ip(1), 3:8]),
          np.transpose(quan_local_shun_pe1[eight_ip(2), 3:8]),
          np.transpose(quan_local_shun_pe1[eight_ip(3), 3:8]),
          np.transpose(quan_local_shun_pe1[eight_ip(4), 3:8]),
          np.transpose(quan_local_shun_pe1[eight_ip(5), 3:8]),
          np.transpose(quan_local_shun_pe1[eight_ip(6), 3:8]),
          np.transpose(quan_local_shun_pe1[eight_ip(7), 3:8]),
          np.transpose(quan_local_shun_pe1[eight_ip(8), 3:8])]
    PL = S * GL
    A_gl_t1[i,:] = np.transpose(subs(PL, {a, b, c}, {B1, B2, B3}))

ddd='elset_p750N_40mm.txt'
fid_elset_p1=open(ddd,'w')

eee='S_p750N_40mm.txt'
fid_pe_p1=open(eee,'w')


D = np.zeros((len(glob_local_shun1)/8, 8))
DDD = np.zeros((len(glob_local_shun1)/8, 8))


for j in range(1, len(glob_local_shun1)/8):
    aaa = glob_local_shun1[(j-1)*8+1,1]
    num=range(1,8) + (j-1)*8
    C=[np.transpose(A_gl_t1[num(1),:]),
       np.transpose(A_gl_t1[num(2), :]),
       np.transpose(A_gl_t1[num(3), :]),
       np.transpose(A_gl_t1[num(4), :]),
       np.transpose(A_gl_t1[num(5), :]),
       np.transpose(A_gl_t1[num(6), :]),
       np.transpose(A_gl_t1[num(7), :]),
       np.transpose(A_gl_t1[num(8), :])]
    CZ = S * C
    D[j,:]=subs(CZ,{a,b,c},{0,0,0})
    ebsino_L=[[D[j,1], D[j,4],D[j,5]],
              [D[j, 4], D[j, 2], D[j, 6]],
              [D[j, 5], D[j, 6], D[j, 3]]]

    ebsino_G = R_gl * ebsino_L * np.transpose(R_gl)
    DDD[j,:]=[ebsino_G[1, 1], ebsino_G[2, 2], ebsino_G[3, 3], ebsino_G[1, 2], ebsino_G[1, 3], ebsino_G[2, 3]]
    fid_elset_p1.writelines('{},{},{},{},{},{}'.format(ebsino_G[1, 1], ebsino_G[2, 2], ebsino_G[3, 3], ebsino_G[1, 2], ebsino_G[1, 3], ebsino_G[2, 3]))



