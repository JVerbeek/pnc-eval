library(reticulate)
np <- import("numpy")

datasets <- np$load("test_change_type.npz")
datasets$files
datasets$f[["Xs"]]
datasets$f[["ys"]]
