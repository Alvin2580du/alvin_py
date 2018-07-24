import pandas as pd
import numpy as np
import scipy.linalg
from tqdm import trange

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
quan_local_shun_pe2 = quan_local_shun_pe1

local_zhi = local_data_set3.head(6)
local_shun_pe1 = local_zhi.values

length = len(local_shun_pe1)
for mm in range(length):
    hangshu = find(local_shun_pe1[mm, 0], local_shun_coord1[:, 0])
    hangshu1 = int(hangshu + local_shun_pe1[mm, 1] - 1)
    if hangshu1 < 6:
        quan_local_shun_pe1[hangshu1, 3:8] = local_shun_pe1[mm, 3:8]

quan_local_wen_pe = quan_local_shun_pe1

glob_file = 'global-3-40.dat.csv'
global_zhi = pd.read_csv(glob_file)
global_shun_coord1 = global_zhi.values
global_shun_coord2 = global_shun_coord1
global_shun_coord3 = global_shun_coord1


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

glob_zuobiao_shun1 = global_shun_coord1[:, 2: 5]
glob_local_shun1 = np.zeros((len(glob_zuobiao_shun1[:, 1]), 5))
for kk in range(len(glob_zuobiao_shun1[:, 1])):
    val1 = R_lg * np.transpose(glob_zuobiao_shun1[kk, :])
    val2 = R_lg * np.transpose(origin)
    glob_local_shun1[kk, 0: 2] = global_shun_coord1[kk, 0: 2]
    glob_local_shun1[kk, 2: 5] = (val1 - val2)[0]
print("glob_local_shun1.shape {}".format(glob_local_shun1.shape))
glob_zuobiao_shun2 = global_shun_coord1[:, 2: 5]
glob_local_shun2 = np.zeros((len(glob_zuobiao_shun2[:, 1]), 5))
for kk in range(len(glob_zuobiao_shun1[:, 1])):
    glob_local_shun2[kk, 2:5] = (R_lg * np.transpose(glob_zuobiao_shun2[kk, :]) - R_lg * np.transpose(origin))[0]
    glob_local_shun2[kk, 0: 2] = global_shun_coord1[kk, 0: 2]

glob_zuobiao_shun1 = global_shun_coord1[:, 2: 5]
glob_local_wen = np.zeros((len(glob_zuobiao_shun1[:, 1]), 5))
for kk in range(len(glob_zuobiao_shun1[:, 1])):
    glob_local_wen[kk, 2:5] = (R_lg * np.transpose(glob_zuobiao_shun1[kk, :]) - R_lg * np.transpose(origin))[0]
    glob_local_wen[kk, 0: 2] = global_shun_coord1[kk, 0: 2]

G2_L2_juli = [0, 0, 0, -720, 0]

for mm in range(1, len(glob_local_shun2[:, 1])):
    glob_local_shun2[mm, :] = glob_local_shun2[mm, :] + G2_L2_juli

LL = np.array([[-1, 1, 1],
               [-1, -1, 1],
               [-1, 1, -1],
               [-1, -1, -1],
               [1, 1, 1],
               [1, -1, 1],
               [1, 1, -1],
               [1, -1, -1]])

a = 1
b = 1
c = 1

a0 = LL[:, 0]
b0 = LL[:, 1]
c0 = LL[:, 2]
N = (1 / 8 * (1 + a0) * (1 + b0) * (1 + c0)).tolist()
S0 = np.array([[N[0], 0, 0, N[1], 0, 0, N[2], 0, 0, N[3], 0, 0, N[4], 0, 0, N[5], 0, 0, N[6], 0, 0, N[7], 0, 0],
               [0, N[0], 0, 0, N[1], 0, 0, N[2], 0, 0, N[3], 0, 0, N[4], 0, 0, N[5], 0, 0, N[6], 0, 0, N[7], 0],
               [0, 0, N[0], 0, 0, N[1], 0, 0, N[2], 0, 0, N[3], 0, 0, N[4], 0, 0, N[5], 0, 0, N[6], 0, 0, N[7]]])
print('S0:{}'.format(S0.shape))
S = np.array(
    [[N[0], 0, 0, 0, 0, 0, N[1], 0, 0, 0, 0, 0, N[2], 0, 0, 0, 0, 0, N[3], 0, 0, 0, 0, 0, N[4], 0, 0, 0, 0, 0, N[5], 0,
      0, 0, 0, 0, N[6], 0, 0, 0, 0, 0, N[7], 0, 0, 0, 0, 0],
     [0, N[0], 0, 0, 0, 0, 0, N[1], 0, 0, 0, 0, 0, N[2], 0, 0, 0, 0, 0, N[3], 0, 0, 0, 0, 0, N[4], 0, 0, 0, 0, 0, N[5],
      0, 0, 0, 0, 0, N[6], 0, 0, 0, 0, 0, N[7], 0, 0, 0, 0],
     [0, 0, N[0], 0, 0, 0, 0, 0, N[1], 0, 0, 0, 0, 0, N[2], 0, 0, 0, 0, 0, N[3], 0, 0, 0, 0, 0, N[4], 0, 0, 0, 0, 0,
      N[5], 0, 0, 0, 0, 0, N[6], 0, 0, 0, 0, 0, N[7], 0, 0, 0],
     [0, 0, 0, N[0], 0, 0, 0, 0, 0, N[1], 0, 0, 0, 0, 0, N[2], 0, 0, 0, 0, 0, N[3], 0, 0, 0, 0, 0, N[4], 0, 0, 0, 0, 0,
      N[5], 0, 0, 0, 0, 0, N[6], 0, 0, 0, 0, 0, N[7], 0, 0],
     [0, 0, 0, 0, N[0], 0, 0, 0, 0, 0, N[1], 0, 0, 0, 0, 0, N[2], 0, 0, 0, 0, 0, N[3], 0, 0, 0, 0, 0, N[4], 0, 0, 0, 0,
      0, N[5], 0, 0, 0, 0, 0, N[6], 0, 0, 0, 0, 0, N[7], 0],
     [0, 0, 0, 0, 0, N[0], 0, 0, 0, 0, 0, N[1], 0, 0, 0, 0, 0, N[2], 0, 0, 0, 0, 0, N[3], 0, 0, 0, 0, 0, N[4], 0, 0, 0,
      0, 0, N[5], 0, 0, 0, 0, 0, N[6], 0, 0, 0, 0, 0, N[7]]])
print('S:{}'.format(S.shape))


def solve(a, b):
    return scipy.linalg.solve(a, b)


A_gl_t1 = np.zeros((len(global_shun_coord1[:, 0]), 8))

for i in trange(len(global_shun_coord1[:, 0])):
    d = np.sqrt(np.square((local_shun_coord1[:, 1] - np.nan_to_num(glob_local_shun1)[i, 1])) +
                np.square((local_shun_coord1[:, 2] - np.nan_to_num(glob_local_shun1)[i, 2])) +
                np.square((local_shun_coord1[:, 3] - np.nan_to_num(glob_local_shun1)[i, 3])))
    JL_data = d
    u = np.sort(JL_data)
    eight_ip = [np.ceil(JL_data.tolist().index(d[0]) / 8) - 1] * 8 + np.array(range(8))
    GL0 = [np.transpose(local_shun_coord1[int(eight_ip[0]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[1]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[2]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[3]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[4]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[5]), 3:5]),
           # np.transpose(local_shun_coord1[int(eight_ip[6]), 3:5]),
           # np.transpose(local_shun_coord1[int(eight_ip[7]), 3:5])
           ]
    global_local_int = S0
    aaaa = np.array([np.nan_to_num(glob_local_shun1)[i, 2], np.nan_to_num(glob_local_shun1)[i, 3], np.nan_to_num(glob_local_shun1)[i, 4]])
    bbbb = np.array([global_local_int[0], global_local_int[1], global_local_int[2]])
    B = solve(aaaa, bbbb)
    B1 = B[0]
    B2 = B[1]
    B3 = B[2]
    GL = [
        np.transpose(quan_local_shun_pe1[int(eight_ip[0]), 3:8]),
        np.transpose(quan_local_shun_pe1[int(eight_ip[1]), 3:8]),
        np.transpose(quan_local_shun_pe1[int(eight_ip[2]), 3:8]),
        np.transpose(quan_local_shun_pe1[int(eight_ip[3]), 3:8]),
        np.transpose(quan_local_shun_pe1[int(eight_ip[4]), 3:8]),
        np.transpose(quan_local_shun_pe1[int(eight_ip[5]), 3:8]),
        # np.transpose(quan_local_shun_pe1[int(eight_ip[6]), 3:8]),
        # np.transpose(quan_local_shun_pe1[int(eight_ip[7]), 3:8]),
    ]
    S_t = np.transpose(S)
    PL = S_t * GL
    A_gl_t1[i, :] = np.transpose(PL)


A_gl_t2 = np.zeros((len(global_shun_coord1[:, 0]), 8))

for i in trange(len(global_shun_coord1[:, 0])):
    d = np.sqrt(np.square((local_shun_coord1[:, 1] - np.nan_to_num(glob_local_shun1)[i, 1])) +
                np.square((local_shun_coord1[:, 2] - np.nan_to_num(glob_local_shun1)[i, 2])) +
                np.square((local_shun_coord1[:, 3] - np.nan_to_num(glob_local_shun1)[i, 3])))
    JL_data = d
    u = np.sort(JL_data)
    eight_ip = [np.ceil(JL_data.tolist().index(d[0]) / 8) - 1] * 8 + np.array(range(8))
    GL0 = [np.transpose(local_shun_coord1[int(eight_ip[0]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[1]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[2]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[3]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[4]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[5]), 3:5]),
           # np.transpose(local_shun_coord1[int(eight_ip[6]), 3:5]),
           # np.transpose(local_shun_coord1[int(eight_ip[7]), 3:5])
           ]
    global_local_int = S0
    aaaa = np.array([np.nan_to_num(glob_local_shun1)[i, 2], np.nan_to_num(glob_local_shun1)[i, 3], np.nan_to_num(glob_local_shun1)[i, 4]])
    bbbb = np.array([global_local_int[0], global_local_int[1], global_local_int[2]])
    B = solve(aaaa, bbbb)
    B1 = B[0]
    B2 = B[1]
    B3 = B[2]
    GL = [
        np.transpose(quan_local_shun_pe2[int(eight_ip[0]), 3:8]),
        np.transpose(quan_local_shun_pe2[int(eight_ip[1]), 3:8]),
        np.transpose(quan_local_shun_pe2[int(eight_ip[2]), 3:8]),
        np.transpose(quan_local_shun_pe2[int(eight_ip[3]), 3:8]),
        np.transpose(quan_local_shun_pe2[int(eight_ip[4]), 3:8]),
        np.transpose(quan_local_shun_pe2[int(eight_ip[5]), 3:8]),
        # np.transpose(quan_local_shun_pe2[int(eight_ip[6]), 3:8]),
        # np.transpose(quan_local_shun_pe2[int(eight_ip[7]), 3:8]),
    ]
    S_t = np.transpose(S)
    PL = S_t * GL
    A_gl_t2[i, :] = np.transpose(PL)
    # subs(PL,{a,b,c},{B1,B2,B3})';


A_gl_s = np.zeros((len(global_shun_coord1[:, 0]), 8))

for i in trange(len(global_shun_coord1[:, 0])):
    d = np.sqrt(np.square((local_shun_coord1[:, 1] - np.nan_to_num(glob_local_shun1)[i, 1])) +
                np.square((local_shun_coord1[:, 2] - np.nan_to_num(glob_local_shun1)[i, 2])) +
                np.square((local_shun_coord1[:, 3] - np.nan_to_num(glob_local_shun1)[i, 3])))
    JL_data = d
    u = np.sort(JL_data)
    eight_ip = [np.ceil(JL_data.tolist().index(d[0]) / 8) - 1] * 8 + np.array(range(8))
    GL0 = [np.transpose(local_shun_coord1[int(eight_ip[0]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[1]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[2]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[3]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[4]), 3:5]),
           np.transpose(local_shun_coord1[int(eight_ip[5]), 3:5]),
           # np.transpose(local_shun_coord1[int(eight_ip[6]), 3:5]),
           # np.transpose(local_shun_coord1[int(eight_ip[7]), 3:5])
           ]
    global_local_int = S0
    aaaa = np.array([np.nan_to_num(glob_local_shun1)[i, 2], np.nan_to_num(glob_local_shun1)[i, 3], np.nan_to_num(glob_local_shun1)[i, 4]])
    bbbb = np.array([global_local_int[0], global_local_int[1], global_local_int[2]])
    B = solve(aaaa, bbbb)
    B1 = B[0]
    B2 = B[1]
    B3 = B[2]
    GL = [
        np.transpose(quan_local_wen_pe[int(eight_ip[0]), 3:8]),
        np.transpose(quan_local_wen_pe[int(eight_ip[1]), 3:8]),
        np.transpose(quan_local_wen_pe[int(eight_ip[2]), 3:8]),
        np.transpose(quan_local_wen_pe[int(eight_ip[3]), 3:8]),
        np.transpose(quan_local_wen_pe[int(eight_ip[4]), 3:8]),
        np.transpose(quan_local_wen_pe[int(eight_ip[5]), 3:8]),
        # np.transpose(quan_local_wen_pe[int(eight_ip[6]), 3:8]),
        # np.transpose(quan_local_wen_pe[int(eight_ip[7]), 3:8]),
    ]
    S_t = np.transpose(S)
    PL = S_t * GL
    A_gl_s[i, :] = np.transpose(PL)


ddd = 'elset_p750N_40mm.txt'
fid_elset_p1 = open(ddd, 'w')

eee = 'S_p750N_40mm.txt'
fid_pe_p1 = open(eee, 'w')

D = np.zeros((int(len(glob_local_shun1) / 8), 8))
print(D.shape)
DDD = np.zeros((int(len(glob_local_shun1) / 8), 8))

# %---------------------------------------------5.1全局模型焊接起始区域写入
for j in range(1, int(len(glob_local_shun1) / 8)):
    aaa = glob_local_shun1[(j - 1) * 8 + 1, 1]
    num = np.array(range(1, 9)) + (j -1) * 8
    print(num)
    C = [np.transpose(A_gl_t1[num[0], :]),
         np.transpose(A_gl_t1[num[1], :]),
         np.transpose(A_gl_t1[num[2], :]),
         np.transpose(A_gl_t1[num[3], :]),
         np.transpose(A_gl_t1[num[4], :]),
         np.transpose(A_gl_t1[num[5], :]),
         np.transpose(A_gl_t1[num[6], :]),
         np.transpose(A_gl_t1[num[7], :])]

    CZ = S * C
    D[j, :] = CZ
    ebsino_L = [[D[j, 1], D[j, 4], D[j, 5]],
                [D[j, 4], D[j, 2], D[j, 6]],
                [D[j, 5], D[j, 6], D[j, 3]]]
    ebsino_G = R_gl * ebsino_L * np.transpose(R_gl)
    DDD[j, :] = [ebsino_G[1, 1], ebsino_G[2, 2], ebsino_G[3, 3], ebsino_G[1, 2], ebsino_G[1, 3], ebsino_G[2, 3]]
    fid_elset_p1.writelines(
        '{},{},{},{},{},{}'.format(ebsino_G[1, 1], ebsino_G[2, 2], ebsino_G[3, 3], ebsino_G[1, 2], ebsino_G[1, 3],
                                   ebsino_G[2, 3]))


# %---------------------------------------------5.2全局模型焊接结束区域写入

for j in range(1, int(len(glob_local_shun2) / 8)):
    aaa = glob_local_shun2[(j - 1) * 8 + 1, 1]
    num = np.array(range(1, 9)) + (j -1) * 8
    C = [np.transpose(A_gl_t2[num[0], :]),
         np.transpose(A_gl_t2[num[1], :]),
         np.transpose(A_gl_t2[num[2], :]),
         np.transpose(A_gl_t2[num[3], :]),
         np.transpose(A_gl_t2[num[4], :]),
         np.transpose(A_gl_t2[num[5], :]),
         np.transpose(A_gl_t2[num[6], :]),
         np.transpose(A_gl_t2[num[7], :])]

    CZ = S * C
    D[j, :] = CZ
    ebsino_L = [[D[j, 1], D[j, 4], D[j, 5]],
                [D[j, 4], D[j, 2], D[j, 6]],
                [D[j, 5], D[j, 6], D[j, 3]]]
    ebsino_G = R_gl * ebsino_L * np.transpose(R_gl)
    DDD[j, :] = [ebsino_G[1, 1], ebsino_G[2, 2], ebsino_G[3, 3], ebsino_G[1, 2], ebsino_G[1, 3], ebsino_G[2, 3]]
    fid_elset_p1.writelines(
        '{},{},{},{},{},{}'.format(ebsino_G[1, 1], ebsino_G[2, 2], ebsino_G[3, 3], ebsino_G[1, 2], ebsino_G[1, 3],
                                   ebsino_G[2, 3]))


#  %---------------------------------------------5.3全局模型焊接稳态区域写入

for j in range(1, int(len(glob_local_wen) / 8)):
    aaa = glob_local_wen[(j - 1) * 8 + 1, 1]
    num = np.array(range(1, 9)) + (j -1) * 8
    C = [np.transpose(A_gl_s[num[0], :]),
         np.transpose(A_gl_s[num[1], :]),
         np.transpose(A_gl_s[num[2], :]),
         np.transpose(A_gl_s[num[3], :]),
         np.transpose(A_gl_s[num[4], :]),
         np.transpose(A_gl_s[num[5], :]),
         np.transpose(A_gl_s[num[6], :]),
         np.transpose(A_gl_s[num[7], :])]

    CZ = S * C
    D[j, :] = CZ
    ebsino_L = [[D[j, 1], D[j, 4], D[j, 5]],
                [D[j, 4], D[j, 2], D[j, 6]],
                [D[j, 5], D[j, 6], D[j, 3]]]
    ebsino_G = R_gl * ebsino_L * np.transpose(R_gl)
    DDD[j, :] = [ebsino_G[1, 1], ebsino_G[2, 2], ebsino_G[3, 3], ebsino_G[1, 2], ebsino_G[1, 3], ebsino_G[2, 3]]


