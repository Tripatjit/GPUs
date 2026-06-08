import numpy as np

mata = np.array([[1, 3], [7, 5]])
matb = np.array([[6, 8], [4, 2]])

res = np.dot(mata, matb)
print(res)

def change_sqr(mat):
    max_dim = max(mat.shape[0], mat.shape[1])
    
    next_power = 1 << max_dim.bit_length() if (max_dim & (max_dim - 1)) != 0 else max_dim
    
    return np.pad(mat, ((0, next_power - mat.shape[0]), (0, next_power - mat.shape[1])), mode='constant')


def rec_sub_mat(a, b):
    n = a.shape[0]
    if(n == 1):
        return np.dot(a, b)
    else:
        a11 = a[:n//2, :n//2]
        a12 = a[:n//2, n//2:]
        a21 = a[n//2:, :n//2]
        a22 = a[n//2:, n//2:]
        b11 = b[:n//2, :n//2]
        b12 = b[:n//2, n//2:]
        b21 = b[n//2:, :n//2]
        b22 = b[n//2:, n//2:]

        p1 = rec_sub_mat(a11, b12-b22)
        p2 = rec_sub_mat(a11+a12, b22)
        p3 = rec_sub_mat(a21 + a22, b11)
        p4 = rec_sub_mat(a22, b21-b11)
        p5 = rec_sub_mat(a11+a22, b11+b22)
        p6 = rec_sub_mat(a12-a22, b21+b22)
        p7 = rec_sub_mat(a11-a21, b11+b12)

        c1 = p5+p4-p2+p6
        c2 = p1+p2
        c3 = p3+p4
        c4 = p5+p1-p3-p7

        left = np.vstack((c1, c3))
        right = np.vstack((c2, c4))
        matc = np.hstack((left, right))

        return matc
    
def strassen_multiply(a, b):
    rows = a.shape[0]
    cols = b.shape[1]
    
    a_new = change_sqr(a)
    b_new = change_sqr(b)
    
    c_new = rec_sub_mat(a_new, b_new)
    
    return c_new[:rows, :cols]

print(strassen_multiply(mata, matb))

testa = np.array([[1,2,3,4],[2,3,4,5],[6,7,8,9],[9,6,4,5]])
testb = np.array([[2,4,6,7], [1,4,5,6],[4,3,2,1], [5,6,7,7]])

print(np.dot(testa, testb))
print(strassen_multiply(testa,testb))

testa = np.array([[1,2,3],[2,3,4],[6,7,8],[4,5,9]])
testb = np.array([[2,4,6,7], [1,4,5,6],[4,3,2,1]])

print(np.dot(testa, testb))
print(strassen_multiply(testa,testb))

