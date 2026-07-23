using CSV, DataFrames, Plots

times = CSV.read("output.csv", DataFrame)

sizes = times.Size
Matmul = times.Matmul_median
Strassen = times.Strassen_median

plot(sizes, [Matmul Strassen], label=["Matmul" "Strassen"])
plot!(legend=:outerbottom, legendcolumns=2)
title!("Time Comparisons for Matmul vs. Strassen (Julia)")
xlabel!("Size")
ylabel!("Median time (s)")
savefig("Julia_times.png")

