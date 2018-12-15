dyn.load(paste("src/Smo615", .Platform$dynlib.ext, sep=""))
smo615 <- function(train_data, train_label, test_data, test_label, c, ind){
	res <- .C("smo615", train_data, train_label, test_data, test_label, c, ind, grid=double(10000), wb=double(3))
	
	# plot	
	train_dat = read.table(train_data, sep = ' ', header = F)
	train_lab = read.table(train_label, sep = ' ', header = F)
	test_dat = read.table(test_data, sep = ' ', header = F)
	test_lab = read.table(test_label, sep = ' ', header = F)

	train_dat = scale(train_dat)
	test_dat = scale(test_dat)

	train = cbind(train_dat, train_lab)
	test = cbind(test_dat, test_lab)
	colnames(train)=c('feature1','feature2','label')
	colnames(test)=c('feature1','feature2','label')

	train$label=as.factor(train$label)
	test$label=as.factor(test$label)
	
	if (ind == 1) {
		xrange1 = seq(min(train$feature1),max(train$feature1),length.out = 100)
		yrange1 = seq(min(train$feature2),max(train$feature2),length.out = 100)

		xrange2 = seq(min(test$feature1),max(test$feature1),length.out = 100)
		yrange2 = seq(min(test$feature2),max(test$feature2),length.out = 100)

		grid_mat = matrix(res$grid,nrow=100,ncol=100)

		plot(train$feature1, train$feature2, col=train$label, main='Decision boundary on training data', xlab='Feature1', ylab='Feature2')
		contour(xrange1,yrange1, grid_mat, levels=c(-1,0,1), add=TRUE)

		plot(test$feature1, test$feature2, col=test$label, main='Decision boundary on testing data', xlab='Feature1', ylab='Feature2')
		contour(xrange2,yrange2, grid_mat, levels=c(-1,0,1), add=TRUE)

		return (matrix(res$grid, nrow=100,ncol=100))
	}
	else {
		plot(train$feature1, train$feature2, col=train$label, main='Decision boundary on training data', xlab='Feature1', ylab='Feature2')
		abline(res$wb[3]/res$wb[1], -res$wb[2]/res$wb[1])
		abline((res$wb[3]+res$wb[1])/res$wb[1], -res$wb[2]/res$wb[1], lty=2)
		abline((res$wb[3]-res$wb[1])/res$wb[1], -res$wb[2]/res$wb[1], lty=2)

		plot(test$feature1, test$feature2, col=test$label, main='Decision boundary on testing data', xlab='Feature1', ylab='Feature2')
		abline(res$wb[3]/res$wb[1], -res$wb[2]/res$wb[1])
		abline((res$wb[3]+res$wb[1])/res$wb[1], -res$wb[2]/res$wb[1], lty=2)
		abline((res$wb[3]-res$wb[1])/res$wb[1], -res$wb[2]/res$wb[1], lty=2)

		return (res$wb)	
	}
}


