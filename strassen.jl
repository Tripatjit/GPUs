using LinearAlgebra, BenchmarkTools, CUDA, Statistics, DataFrames, CSV

function strassen(a, b)
    n=size(a,1)
    mid = div(n,2)
    if size(a, 1) <= 512
        return a*b
    else
        a11 = a[begin:mid, begin:mid]
        a12 = a[begin:mid, mid+1:end]
        a21 = a[mid+1:end, begin:mid]
        a22 = a[mid+1:end, mid+1:end]
        b11 = b[begin:mid, begin:mid]
        b12 = b[begin:mid, mid+1:end]
        b21 = b[mid+1:end, begin:mid]
        b22 = b[mid+1:end, mid+1:end]

        p1 = strassen(a11, b12-b22)
        p2 = strassen(a11+a12, b22)
        p3 = strassen(a21 + a22, b11)
        p4 = strassen(a22, b21-b11)
        p5 = strassen(a11+a22, b11+b22)
        p6 = strassen(a12-a22, b21+b22)
        p7 = strassen(a11-a21, b11+b12)

        c1 = p5+p4-p2+p6
        c2 = p1+p2
        c3 = p3+p4
        c4 = p5+p1-p3-p7

        left = vcat(c1,c3)
        right = vcat(c2, c4)
        c = hcat(left, right)
        return c
    end

end

sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
mm_mean = zeros(Float64, length(sizes))
mm_median = zeros(Float64, length(sizes))
mm_std = zeros(Float64, length(sizes))
mm_min = zeros(Float64, length(sizes))
mm_max = zeros(Float64, length(sizes))

str_mean = zeros(Float64, length(sizes))
str_median = zeros(Float64, length(sizes))
str_std = zeros(Float64, length(sizes))
str_min = zeros(Float64, length(sizes))
str_max = zeros(Float64, length(sizes))

for i = 1:length(sizes)
    times_str = zeros(Float64, 10)
    times_mm = zeros(Float64, 10)
    for j = 1:10
        a = randn(sizes[i], sizes[i])
        b = randn(sizes[i], sizes[i])
        a_gpu = cu(a)
        b_gpu = cu(b)
        times_mm[j]  = CUDA.@elapsed a_gpu * b_gpu
        times_str[j] = CUDA.@elapsed strassen(a_gpu, b_gpu)
    end

    mm_mean[i] = mean(times_mm)
    mm_median[i] = median(times_mm)
    mm_std[i] = std(times_mm)
    mm_min[i] = minimum(times_mm)
    mm_max[i] = maximum(times_mm)

    str_mean[i] = mean(times_str)
    str_median[i] = median(times_str)
    str_std[i] = std(times_str)
    str_min[i] = minimum(times_str)
    str_max[i] = maximum(times_str)

    println("Matrix of size $(sizes[i]) done...")
end

df = DataFrame(
    Size = sizes,
    Matmul_mean = mm_mean,
    Matmul_median = mm_median,
    Matmul_std = mm_std,
    Matmul_min = mm_min,
    Matmul_max = mm_max,
    Strassen_mean = str_mean,
    Strassen_median = str_median,
    Strassen_std = str_std,
    Strassen_min = str_min,
    Strassen_max = str_max
)

CSV.write("output.csv", df)