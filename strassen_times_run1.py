import torch 
import time
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

def rec_sub_mat(a, b):
    n = a.shape[0]
    if(n<=32):
        return torch.matmul(a,b)
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

        left = torch.vstack((c1,c3))
        right = torch.vstack((c2,c4))
        matc = torch.hstack((left,right))

        return matc
    
def iter_sub_mat(a, b):
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

            if n<=32:
              res = torch.matmul(a_new,b_new)

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
                    stack_mat.append({'A': a11,'B': b12.sub(b22), 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

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
                    stack_mat.append({'A': a11.add(a12),'B': b22, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

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
                    stack_mat.append({'A': a21.add(a22),'B': b11, 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

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
                    stack_mat.append({'A': a22,'B': b21.sub(b11), 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

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
                    stack_mat.append({'A': a11.add(a22),'B': b11.add(b22), 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

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
                    stack_mat.append({'A': a12.sub(a22),'B': b21.add(b22), 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

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
                    stack_mat.append({'A': a11.sub(a21),'B': b11.add(b12), 'p1': p1_new,'p2': p2_new,'p3': p3_new,'p4': p4_new,'p5': p5_new,'p6': p6_new,'p7': p7_new,'location': 1})

                case 8:
                    c1 = p5_new+p4_new-p2_new+p6_new
                    c2 = p1_new+p2_new
                    c3 = p3_new+p4_new
                    c4 = p5_new+p1_new-p3_new-p7_new

                    left = torch.vstack((c1,c3))
                    right = torch.vstack((c2,c4))
                    matc = torch.hstack((left,right))
                    if len(stack_mat) == 0:
                        return matc

                    parent = stack_mat[-1]
                    loc = parent["location"] - 1
                    parent[f"p{loc}"] = matc
                    continue

#@torch.compile(dynamic=True)
def strassen_multiply(a,b):
    rows = a.shape[0]
    cols = b.shape[1]

    a_new =change_sqr(a)
    b_new =change_sqr(b)

    c_new = rec_sub_mat(a_new, b_new)

    return c_new[:rows, :cols]

#@torch.compile(dynamic=True)
def strassen_iter_multiply(a,b):
    rows = a.shape[0]
    cols = b.shape[1]

    a_new =change_sqr(a)
    b_new =change_sqr(b)

    c_new = iter_sub_mat(a_new, b_new)

    return c_new[:rows, :cols]

#a_comp = torch.rand(64, 64, device=device)
#b_comp = torch.rand(64, 64, device=device)

#c_comp_rec = strassen_multiply(a_comp, b_comp)
#c_comp_iter = strassen_iter_multiply(a_comp, b_comp)

#torch.cuda.synchronize()

res_mm = []
res_rec = []
res_int = []

for i in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    a = torch.rand(i, i, device=device)
    b = torch.rand(i, i, device=device)
    strassen_multiply(a, b)
    if(device.type=="cuda"):
	    torch.cuda.synchronize()
    start_mm = time.perf_counter()
    res = torch.matmul(a,b)
    if(device.type=="cuda"):
	    torch.cuda.synchronize()
    end_mm = time.perf_counter()
    total_mm = end_mm-start_mm
    print(str(i) + "x" + str(i) + " matrix took " + str(total_mm) + " seconds with matmul.\n")
    
    if(device.type=="cuda"):
	    torch.cuda.synchronize()
    start = time.perf_counter()
    c_rec = strassen_multiply(a, b)
    if(device.type=="cuda"):
	    torch.cuda.synchronize()
    end = time.perf_counter()
    total = end - start
    print(str(i) + "x" + str(i) + " matrix took " + str(total) + " seconds with recursive strassen.\n")

    if(device.type=="cuda"):
	    torch.cuda.synchronize()
    start_it = time.perf_counter()
    c_mm = strassen_iter_multiply(a, b)
    if(device.type=="cuda"):    
	    torch.cuda.synchronize()
    
    end_it = time.perf_counter()
    total_it = end_it - start_it
    print(str(i) + "x" + str(i) + " matrix took " + str(total_it) + " seconds with iterative strassen.\n")
    
    res_mm.append(total_mm)
    res_rec.append(total)
    res_int.append(total_it)
    
plt.plot([128, 256, 512, 1024, 2048, 4096, 8192, 16384], res_mm, label = "torch.matmul")
plt.plot([128, 256, 512, 1024, 2048, 4096, 8192, 16384], res_rec, label = "Recursive Strassen")
plt.plot([128, 256, 512, 1024, 2048, 4096, 8192, 16384], res_int, label = "Iterative Strassen")
plt.legend()
plt.title("Time Comparisons: Torch.matmul vs. Strassen")
plt.xlabel("Matrix Size")
plt.ylabel("Time (seconds)")
plt.savefig("time_comp.png")