pkgname <- "Smo615"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('Smo615')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
cleanEx()
nameEx("Smo615-package")
### * Smo615-package

flush(stderr()); flush(stdout())

### Name: Smo615-package
### Title: 2D Visualization of SVM using the SMO algorithm
### Aliases: Smo615-package Smo615
### Keywords: SMO; SVM

### ** Examples

  ## Not run: 
##D      ## Example
##D      library(Smo615)
##D      res=smo615(train_data_path, train_label_path, test_data_path, test_label_path, C_regularization, linear_kernel(0) /Gaussian_kernel(1))
##D   
## End(Not run)



### * <FOOTER>
###
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
