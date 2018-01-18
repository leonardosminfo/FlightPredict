met_install_libraries <- function() 
{
  install.packages("ROCR")
}

met_load_libraries <- function() 
{
  library(ROCR)
}

croc <- function(method, test, clabel)
{
  classification <- method[[1]]
  class_prob <- classification[c(1,2)]
  data_roc <- class_prob[,2]
  #lab <- clabel
  
  pred <- prediction(data_roc, test[,clabel])
  perf <- performance(pred, "tpr", "fpr")
  plot(perf)
  auc <- performance(pred, "auc")
  aroc <- unlist(slot(auc, "y.values"))
  return (aroc)
}

cpl.pur <- function(method)
{
  ntruest <- 0
  ntruegal <- 0
  nfalsest <- 0
  nfalsegal <- 0
  
  conf_matr_test <- method[[3]]
  
  ntruegal <- conf_matr_test[1,1]
  nfalsegal <- conf_matr_test[2,1]
  ntruest <- conf_matr_test[2,2]
  nfalsest <- conf_matr_test[1,2]
  
  cplgal <- (ntruegal/(ntruegal+nfalsest))*100
  purgal <- (ntruegal/(ntruegal+nfalsegal))*100
  cplst <- (ntruest/(ntruest+nfalsegal))*100
  purst <- (ntruest/(ntruest+nfalsest))*100
  
  print_cplgal = sprintf("Completeza de Galáxias %f", cplgal)
  print_purgal = sprintf("Pureza de Galáxias %f", purgal)
  print_cplst = sprintf("Completeza de Estrelas %f", cplst)
  print_purst = sprintf("Pureza de Estrelas %f", purst)
  
  return (list(print_cplgal, print_purgal, print_cplst, print_purst))
}

accuracy <- function(method)
{
  ntruest <- 0
  ntruegal <- 0
  nfalsest <- 0
  nfalsegal <- 0
  
  conf_matr_test <- method[[3]]
  
  ntruegal <- conf_matr_test[1,1]
  nfalsegal <- conf_matr_test[2,1]
  ntruest <- ifelse(dim(conf_matr_test)[2] == 1, 0, conf_matr_test[2,2])
  nfalsest <- ifelse(dim(conf_matr_test)[2] == 1, 0, conf_matr_test[1,2])
  
  
  acc <- ((ntruegal + ntruest)/(ntruegal+nfalsest+nfalsegal+ntruest))*100
  
  print_acc = sprintf("Acurácia %f", acc)
  
  
  return (acc)
}

sensitivity <- function(method)
{
	tp <- 0
	fn <- 0

	conf_matr_test <- method[[3]]
	
	tp <- conf_matr_test[2, 2]
	fn <- conf_matr_test[2, 1]
	
	sens <- (tp / (tp + fn))*100
	
	print_sens <- sprintf('Sensitivity %f' , sens)
	
	return (sens)
}
