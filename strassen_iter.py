import torch

mata = torch.tensor([[1,3],[7,5]])
matb = torch.tensor([[6,8],[4,2]])
result_mm = torch.matmul(mata, matb)
print(result_mm)

def change_sqr(mat):
    max_dim = max(mat.shape[0], mat.shape[1])

    if(max_dim &(max_dim - 1)) != 0:
        next_power = 1 << max_dim.bit_length()
    else:
        next_power = max_dim
    
    return torch.nn.functional.pad(mat, (0, next_power - mat.shape[1], 0, next_power - mat.shape[0]), mode = 'constant')


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

def strassen_multiply(a,b):
    rows = a.shape[0]
    cols = b.shape[1]

    a_new =change_sqr(a)
    b_new =change_sqr(b)

    c_new = iter_sub_mat(a_new, b_new)

    return c_new[:rows, :cols]

print(strassen_multiply(mata, matb))

testa = torch.tensor([[1,2,3,4],[2,3,4,5],[6,7,8,9],[9,6,4,5]])
testb = torch.tensor([[2,4,6,7], [1,4,5,6],[4,3,2,1], [5,6,7,7]])

print(torch.matmul(testa, testb))
print(strassen_multiply(testa,testb))

testa = torch.tensor([[1,2,3],[2,3,4],[6,7,8],[4,5,9]])
testb = torch.tensor([[2,4,6,7], [1,4,5,6],[4,3,2,1]])

print(torch.matmul(testa, testb))
print(strassen_multiply(testa,testb))