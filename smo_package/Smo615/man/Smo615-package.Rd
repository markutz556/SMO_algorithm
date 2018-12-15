\name{Smo615-package}
\alias{Smo615-package}
\alias{Smo615}
\docType{package}
\title{
  2D Visualization of SVM using the SMO algorithm
}
\description{
  We implement a support vector classifier using SMO algorithm in C++ and compile it into this R package. It seeks to find a decision boundary with the greatest margin of separation between the input classes. If the training data are not linearly separable, a "soft-margin" (or regularized) version can be implemented that allows for misclassified samples. The output decision boundary can be linear, as it is with logisitc regression, or non-linear with the use of various kernels that map the input features to a higher-dimensional space.
}
\details{
  We define a linear kernel and a Gaussian (also known as radial basis function or RBF) kernel. There are three main steps to train our model: take_step(), examine_example(), and train(). The train() function implements selection of the first α to optimize via the first choice heuristic and passes this value to examine_example(). Then examine_example() implements the second choice heuristic to choose the second α
to optimize, and passes the index of both α values to take_step(). Finally take_step() carries out the meat of the calculations and computes the two new α values, a new threshold b, and updates the error cache. The train() function uses a while loop to iterate through the α values in a few different ways until no more optimizations can be made, at which point it returns the optimized α vector. With all the trained parameters, we conduct the predictions of the testing data and also plot the decision boundary for both the training data and testing data.
}
\author{
Rui Ma, Shaocheng Wu, Mingyu Du

Maintainer: Biostat 615 <@umich.edu>
}
\references{
  1.Platt, J.: Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machine. Technical Report MSR-TR-98-14. Microsoft research (1998).
  2.Jon Charest: https://jonchar.net/notebooks/SVM/.
  3.Wikipedia contributors. Support Vector Machine. Wikipedia, The Free Encyclopedia. Available at:https://en.wikipedia.org/wiki/Support_vector_machine. Accessed December 1, 2018.
}
\keyword{ SMO; SVM }
\seealso{
  Optional links to other man pages
}
\examples{
  \dontrun{
     ## Example
     library(Smo615)
     res=smo615(train_data_path, train_label_path, test_data_path, test_label_path, C_regularization, linear_kernel(0) /Gaussian_kernel(1))
  }
}
