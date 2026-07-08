import torch
import time
import statistics
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def change_sqr(mat):
    max_dim = max(mat.shape[0], mat.shape[1])

    if(max_dim &(max_dim - 1)) != 0:
        next_power = 1 << max_dim.bit_length()
    else:
        next_power = max_dim
    
    return torch.nn.functional.pad(mat, (0, next_power - mat.shape[1], 0, next_power - mat.shape[0]), mode = 'constant')

def wino_rec(a, b):
    n = a.shape[0]
    if(n==1):
        return a*b
    else:
        a11 = a[:n//2, :n//2]
        a12 = a[:n//2, n//2:]
        a21 = a[n//2:, :n//2]
        a22 = a[n//2:, n//2:]
        b11 = b[:n//2, :n//2]
        b12 = b[:n//2, n//2:]
        b21 = b[n//2:, :n//2]
        b22 = b[n//2:, n//2:]

        p1 = wino_rec(a21+a22-a11, b22-b12+b11)
        p2 = wino_rec(a11, b11)
        p3 = wino_rec(a12, b21)
        p4 = wino_rec(a11-a21, b22-b12)
        p5 = wino_rec(a21+a22, b12-b11)
        p6 = wino_rec(a12-a21-a22+a11, b22)
        p7 = wino_rec(a22, b22-b12+b11-b21)

        c1 = p2+p3
        c2 = p1+p2+p5+p6
        c3 = p1+p2+p4-p7
        c4 = p1+p2+p4+p5

        left = torch.vstack((c1,c3))
        right = torch.vstack((c2,c4))
        matc = torch.hstack((left,right))

        return matc



def strassen_multiply_rec(a,b):
    rows = a.shape[0]
    cols = b.shape[1]

    a_new =change_sqr(a)
    b_new =change_sqr(b)

    c_new = wino_rec(a_new, b_new)

    return c_new[:rows, :cols]

def wino_iter(a, b):
    stack_mat = [{
        'A': a,
        'B': b,
        'p1': None,
        'p2': None,
        'p3': None,
        'p4': None,
        'p5': None,
        'p6': None,
        'p7': None,
        'c1': None,
        'c2': None,
        'c3': None,
        'c4': None,
        'location': 1
    }]
    results = []             #stack of results
    
    while len(stack_mat) > 0:
            stack_top = stack_mat[-1]
            a_new = stack_top['A']
            b_new = stack_top['B']
            p1_new = stack_top['p1']
            p2_new = stack_top['p2']
            p3_new = stack_top['p3']
            p4_new = stack_top['p4']
            p5_new = stack_top['p5']
            p6_new = stack_top['p6']
            p7_new = stack_top['p7']
            n = a_new.shape[0]
            stack_mat.pop()

            if n == 1:
              res = a_new * b_new

              if len(stack_mat) == 0:
                  return res

              parent = stack_mat[-1]
              loc = parent["location"] - 1
              parent[f"p{loc}"] = res
              continue

            match stack_top['location']:
                case 1:
                    a11 = a_new[:n//2, :n//2]
                    a12 = a_new[:n//2, n//2:]
                    a21 = a_new[n//2:, :n//2]
                    a22 = a_new[n//2:, n//2:]
                    b11 = b_new[:n//2, :n//2]
                    b12 = b_new[:n//2, n//2:]
                    b21 = b_new[n//2:, :n//2]
                    b22 = b_new[n//2:, n//2:]

                    stack_mat.append({'A': a_new,'B': b_new, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 2})
                    stack_mat.append({'A': (a21+a22-a11),'B': (b22-b12+b11), 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 2:
                    a11 = a_new[:n//2, :n//2]
                    a12 = a_new[:n//2, n//2:]
                    a21 = a_new[n//2:, :n//2]
                    a22 = a_new[n//2:, n//2:]
                    b11 = b_new[:n//2, :n//2]
                    b12 = b_new[:n//2, n//2:]
                    b21 = b_new[n//2:, :n//2]
                    b22 = b_new[n//2:, n//2:]

                    stack_mat.append({'A': a_new,'B': b_new, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 3})
                    stack_mat.append({'A': a11,'B': b11, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 3:
                    a11 = a_new[:n//2, :n//2]
                    a12 = a_new[:n//2, n//2:]
                    a21 = a_new[n//2:, :n//2]
                    a22 = a_new[n//2:, n//2:]
                    b11 = b_new[:n//2, :n//2]
                    b12 = b_new[:n//2, n//2:]
                    b21 = b_new[n//2:, :n//2]
                    b22 = b_new[n//2:, n//2:]

                    stack_mat.append({'A': a_new,'B': b_new, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 4})
                    stack_mat.append({'A': a12,'B': b21, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 4:
                    a11 = a_new[:n//2, :n//2]
                    a12 = a_new[:n//2, n//2:]
                    a21 = a_new[n//2:, :n//2]
                    a22 = a_new[n//2:, n//2:]
                    b11 = b_new[:n//2, :n//2]
                    b12 = b_new[:n//2, n//2:]
                    b21 = b_new[n//2:, :n//2]
                    b22 = b_new[n//2:, n//2:]

                    stack_mat.append({'A': a_new,'B': b_new, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 5})
                    stack_mat.append({'A': (a11-a21),'B': b22-b12, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 5:
                    a11 = a_new[:n//2, :n//2]
                    a12 = a_new[:n//2, n//2:]
                    a21 = a_new[n//2:, :n//2]
                    a22 = a_new[n//2:, n//2:]
                    b11 = b_new[:n//2, :n//2]
                    b12 = b_new[:n//2, n//2:]
                    b21 = b_new[n//2:, :n//2]
                    b22 = b_new[n//2:, n//2:]

                    stack_mat.append({'A': a_new,'B': b_new, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 6})
                    stack_mat.append({'A': a21+a22,'B': b12-b11, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 6:
                    a11 = a_new[:n//2, :n//2]
                    a12 = a_new[:n//2, n//2:]
                    a21 = a_new[n//2:, :n//2]
                    a22 = a_new[n//2:, n//2:]
                    b11 = b_new[:n//2, :n//2]
                    b12 = b_new[:n//2, n//2:]
                    b21 = b_new[n//2:, :n//2]
                    b22 = b_new[n//2:, n//2:]

                    stack_mat.append({'A': a_new,'B': b_new, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 7})
                    stack_mat.append({'A': a12-a21-a22+a11,'B': b22, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 7:
                    a11 = a_new[:n//2, :n//2]
                    a12 = a_new[:n//2, n//2:]
                    a21 = a_new[n//2:, :n//2]
                    a22 = a_new[n//2:, n//2:]
                    b11 = b_new[:n//2, :n//2]
                    b12 = b_new[:n//2, n//2:]
                    b21 = b_new[n//2:, :n//2]
                    b22 = b_new[n//2:, n//2:]

                    stack_mat.append({'A': a_new,'B': b_new, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 8})
                    stack_mat.append({'A': a22,'B': b22-b12+b11-b21, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 8:
                    c1 = p2_new+p3_new
                    c2 = p1_new+p2_new+p5_new+p6_new
                    c3 = p1_new+p2_new+p4_new-p7_new
                    c4 = p1_new+p2_new+p4_new+p5_new

                    left = torch.vstack((c1,c3))
                    right = torch.vstack((c2,c4))
                    matc = torch.hstack((left,right))
                    if len(stack_mat) == 0:
                        return matc

                    parent = stack_mat[-1]
                    loc = parent["location"] - 1
                    parent[f"p{loc}"] = matc
                    continue


def strassen_multiply_iter(a,b):
    rows = a.shape[0]
    cols = b.shape[1]

    a_new =change_sqr(a)
    b_new =change_sqr(b)

    c_new = wino_iter(a_new, b_new)

    return c_new[:rows, :cols]

res_mm = []
res_rec = []
res_int = []

for i in [128, 256, 512, 1024, 2048, 4096]:
    for j in range(3):
        total_mm = []
        total_rec = []
        total_iter = []
        a = torch.rand(i, i, device=device)
        b = torch.rand(i, i, device=device)
        if(device.type=="cuda"):
            torch.cuda.synchronize()
        start_mm = time.perf_counter()
        res = torch.matmul(a,b)
        if(device.type=="cuda"):
            torch.cuda.synchronize()
        end_mm = time.perf_counter()
        total_mm.append(end_mm-start_mm)
        print(str(i) + "x" + str(i) + " matrix took " + str(end_mm-start_mm) + " seconds with matmul.\n")
        
        if(device.type=="cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        c_rec = strassen_multiply_rec(a, b)
        if(device.type=="cuda"):
            torch.cuda.synchronize()
        end = time.perf_counter()
        total_rec.append(end-start)
        print(str(i) + "x" + str(i) + " matrix took " + str(end-start) + " seconds with recursive strassen.\n")

        if(device.type=="cuda"):
            torch.cuda.synchronize()
        start_it = time.perf_counter()
        c_mm = strassen_multiply_iter(a, b)
        if(device.type=="cuda"):    
            torch.cuda.synchronize()
        
        end_it = time.perf_counter()
        total_iter.append(end_it-start_it)
        print(str(i) + "x" + str(i) + " matrix took " + str(end_it-start_it) + " seconds with iterative strassen.\n")
    
    res_mm.append(statistics.mean(total_mm))
    res_rec.append(statistics.mean(total_rec))
    res_int.append(statistics.mean(total_iter))
    
plt.plot([128, 256, 512, 1024, 2048, 4096], res_mm, label = "torch.matmul")
plt.plot([128, 256, 512, 1024, 2048, 4096], res_rec, label = "Recursive Strassen")
plt.plot([128, 256, 512, 1024, 2048, 4096], res_int, label = "Iterative Strassen")
plt.legend()
plt.title("Time Comparisons: Torch.matmul vs. Strassen")
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.savefig("time_comp.png")

