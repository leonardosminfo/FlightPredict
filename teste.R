loadlibrary <- function(x) 
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x, repos='http://cran.us.r-project.org', dep=TRUE)
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

loadlibrary('nnet')
loadlibrary('kernlab')
loadlibrary('rattle')
loadlibrary('RSNNS')
loadlibrary('e1071')  
loadlibrary('class')
loadlibrary('randomForest')  

source("Classification.R")

#CURVATURE
loadlibrary("caret")

#LASSO
loadlibrary("glmnet")
loadlibrary("leaps")

#FSELECTOR
loadlibrary("rJava")
loadlibrary("RWeka")
loadlibrary("RWekajars")
loadlibrary("FSelector")

loadlibrary("doBy")

#BALANCEAMENTO
loadlibrary('unbalanced')

source("Preprocessing.R")
source("BestConfiguration_mlp.R")


loadlibrary("ROCR")
loadlibrary("DMwR")
source("Metrics.R")
####################################################################
pre_install()


#realizando Limpeza
dataset_integrado_low=dataset_integrado[1:50000,]


dataset_limpo=cleanData(dataset_integrado)
dataset_limpo_low=dataset_limpo[1:5000,]

#dataset_integrado   3683735  => 43 colunas
colnames(dataset_integrado)

#[1] "siglaempresa"              "numerovoo"                 "di"                        "tipolinha"                
#[5] "voorealizado"              "partidaprevista"           "partidareal"               "tempoatrasopartida"       
#[9] "diautilpartida"            "aeroportopartida"          "chegadaprevista"           "chegadareal"              
#[13] "tempoatrasochegada"        "aeroportochegada"          "diasemana"                 "mes"                      
#[17] "data_hora_met"             "data_hora_met_chegada"     "ano_partida"               "temperature_dep"          
#[21] "dew_pointc_dep"            "humidity_dep"              "sea_level_pressurehpa_dep" "visibilitykm_dep"         
#[25] "wind_direction_dep"        "wind_speedkm_h_dep"        "gust_speedkm_h_dep"        "precipitationmm_dep"      
#[29] "events_dep"                "conditions_dep"            "winddirdegrees_dep"        "temperature_arr"          
#[33] "dew_pointc_arr"            "humidity_arr"              "sea_level_pressurehpa_arr" "visibilitykm_arr"         
#[37] "wind_direction_arr"        "wind_speedkm_h_arr"        "gust_speedkm_h_arr"        "precipitationmm_arr"      
#[41] "events_arr"                "conditions_arr"            "winddirdegrees_arr"   

#dataset_limpo  25953139 => 24 colunas
colnames(dataset_limpo)
#[1] "siglaempresa"              "numerovoo"                 "partidaprevista"           "diautilpartida"           
#[5] "aeroportopartida"          "chegadaprevista"           "aeroportochegada"          "diasemana"                
#[9] "mes"                       "temperature_dep"           "dew_pointc_dep"            "humidity_dep"             
#[13] "sea_level_pressurehpa_dep" "wind_speedkm_h_dep"        "conditions_dep"            "winddirdegrees_dep"       
#[17] "temperature_arr"           "dew_pointc_arr"            "humidity_arr"              "sea_level_pressurehpa_arr"
#[21] "wind_speedkm_h_arr"        "conditions_arr"            "winddirdegrees_arr"        "alvo"


dataset_transformado=transformData(dataset_limpo)
dataset_transformado_low=dataset_transformado[1:5000,]
colnames(dataset_transformado)

#[1] "siglaempresa"     "numerovoo"        "diautilpartida"   "aeroportopartida" "aeroportochegada" "diasemana"       
#[7] "mes"              "conditions_dep"   "conditions_arr"   "alvo"             "temp_dep"         "dew_dep"         
#[13] "pressure_dep"     "humid_dep"        "wspeed_dep"       "temp_arr"         "dew_arr"          "pressure_arr"    
#[19] "humid_arr"        "wspeed_arr"       "time_dep"         "wdirection_dep"   "time_arr"         "wdirection_arr"  


stratified_sample_0.1_low=stratified_sample_0.1[1:5000,]
colnames(stratified_sample_0.1)
#[1] "alvo"                                        "temp_dep"                                   
#[3] "dew_dep"                                     "pressure_dep"                               
#[5] "humid_dep"                                   "wspeed_dep"                                 
#[7] "temp_arr"                                    "dew_arr"                                    
#[9] "pressure_arr"                                "humid_arr"                                  
#[11] "wspeed_arr"                                  "siglaempresaAZU"                            
#[13] "siglaempresaGLO"                             "siglaempresaONE"                            
#[15] "siglaempresaTAM"                             "aeroportopartidaSBBE"                       
#[17] "aeroportopartidaSBBR"                        "aeroportopartidaSBCF"                       
#[19] "aeroportopartidaSBCT"                        "aeroportopartidaSBEG"                       
#[21] "aeroportopartidaSBFL"                        "aeroportopartidaSBFZ"                       
#[23] "aeroportopartidaSBGL"                        "aeroportopartidaSBGO"                       
#[25] "aeroportopartidaSBGR"                        "aeroportopartidaSBKP"                       
#[27] "aeroportopartidaSBPA"                        "aeroportopartidaSBRF"                       
#[29] "aeroportopartidaSBRJ"                        "aeroportopartidaSBSP"                       
#[31] "aeroportopartidaSBSV"                        "aeroportopartidaSBVT"                       
#[33] "aeroportochegadaSBBE"                        "aeroportochegadaSBBR"                       
#[35] "aeroportochegadaSBCF"                        "aeroportochegadaSBCT"                       
#[37] "aeroportochegadaSBEG"                        "aeroportochegadaSBFL"                       
#[39] "aeroportochegadaSBFZ"                        "aeroportochegadaSBGL"                       
#[41] "aeroportochegadaSBGO"                        "aeroportochegadaSBGR"                       
#[43] "aeroportochegadaSBKP"                        "aeroportochegadaSBPA"                       
#[45] "aeroportochegadaSBRF"                        "aeroportochegadaSBRJ"                       
#[47] "aeroportochegadaSBSP"                        "aeroportochegadaSBSV"                       
#[49] "aeroportochegadaSBVT"                        "mes"                                        
#[51] "diasemana"                                   "time_depafternoon"                          
#[53] "time_depearly evening"                       "time_depearly morning"                      
#[55] "time_deplate evening"                        "time_deplate morning"                       
#[57] "time_depmid morning"                         "time_depnight"                              
#[59] "time_arrafternoon"                           "time_arrearly evening"                      
#[61] "time_arrearly morning"                       "time_arrlate evening"                       
#[63] "time_arrlate morning"                        "time_arrmid morning"                        
#[65] "time_arrnight"                               "conditions_depBlowing Sand"                 
#[67] "conditions_depClear"                         "conditions_depDrizzle"                      
#[69] "conditions_depDust Whirls"                   "conditions_depFog"                          
#[71] "conditions_depHaze"                          "conditions_depHeavy Drizzle"                
#[73] "conditions_depHeavy Fog"                     "conditions_depHeavy Rain"                   
#[75] "conditions_depHeavy Rain Showers"            "conditions_depHeavy Thunderstorms and Rain" 
#[77] "conditions_depHeavy Thunderstorms with Hail" "conditions_depLight Drizzle"                
#[79] "conditions_depLight Fog"                     "conditions_depLight Rain"                   
#[81] "conditions_depLight Rain Showers"            "conditions_depLight Thunderstorm"           
#[83] "conditions_depLight Thunderstorms and Rain"  "conditions_depLight Volcanic Ash"           
#[85] "conditions_depMist"                          "conditions_depMostly Cloudy"                
#[87] "conditions_depOvercast"                      "conditions_depPartial Fog"                  
#[89] "conditions_depPartly Cloudy"                 "conditions_depPatches of Fog"               
#[91] "conditions_depRain"                          "conditions_depRain Showers"                 
#[93] "conditions_depSandstorm"                     "conditions_depScattered Clouds"             
#[95] "conditions_depShallow Fog"                   "conditions_depSmoke"                        
#[97] "conditions_depThunderstorm"                  "conditions_depThunderstorms and Rain"       
#[99] "conditions_depThunderstorms with Hail"       "conditions_depUnknown"                      
#[101] "conditions_depVolcanic Ash"                  "conditions_depWidespread Dust"              
#[103] "conditions_arrBlowing Sand"                  "conditions_arrClear"                        
#[105] "conditions_arrDrizzle"                       "conditions_arrFog"                          
#[107] "conditions_arrHaze"                          "conditions_arrHeavy Drizzle"                
#[109] "conditions_arrHeavy Fog"                     "conditions_arrHeavy Rain"                   
#[111] "conditions_arrHeavy Rain Showers"            "conditions_arrHeavy Thunderstorms and Rain" 
#[113] "conditions_arrHeavy Thunderstorms with Hail" "conditions_arrLight Drizzle"                
#[115] "conditions_arrLight Fog"                     "conditions_arrLight Rain"                   
#[117] "conditions_arrLight Rain Showers"            "conditions_arrLight Thunderstorm"           
#[119] "conditions_arrLight Thunderstorms and Rain"  "conditions_arrLight Volcanic Ash"           
#[121] "conditions_arrMist"                          "conditions_arrMostly Cloudy"                
#[123] "conditions_arrOvercast"                      "conditions_arrPartial Fog"                  
#[125] "conditions_arrPartly Cloudy"                 "conditions_arrPatches of Fog"               
#[127] "conditions_arrRain"                          "conditions_arrRain Showers"                 
#[129] "conditions_arrSandstorm"                     "conditions_arrScattered Clouds"             
#[131] "conditions_arrShallow Fog"                   "conditions_arrSmoke"                        
#[133] "conditions_arrThunderstorm"                  "conditions_arrThunderstorms and Rain"       
#[135] "conditions_arrThunderstorms with Hail"       "conditions_arrUnknown"                      
#[137] "conditions_arrVolcanic Ash"                  "conditions_arrWidespread Dust"              
#[139] "wdirection_depE"                             "wdirection_depENE"                          
#[141] "wdirection_depESE"                           "wdirection_depN"                            
#[143] "wdirection_depNE"                            "wdirection_depNNE"                          
#[145] "wdirection_depNNO"                           "wdirection_depNO"                           
#[147] "wdirection_depO"                             "wdirection_depONO"                          
#[149] "wdirection_depOSO"                           "wdirection_depS"                            
#[151] "wdirection_depSE"                            "wdirection_depSO"                           
#[153] "wdirection_depSSE"                           "wdirection_depSSO"                          
#[155] "wdirection_arrE"                             "wdirection_arrENE"                          
#[157] "wdirection_arrESE"                           "wdirection_arrN"                            
#[159] "wdirection_arrNE"                            "wdirection_arrNNE"                          
#[161] "wdirection_arrNNO"                           "wdirection_arrNO"                           
#[163] "wdirection_arrO"                             "wdirection_arrONO"                          
#[165] "wdirection_arrOSO"                           "wdirection_arrS"                            
#[167] "wdirection_arrSE"                            "wdirection_arrSO"                           
#[169] "wdirection_arrSSE"                           "wdirection_arrSSO"
#transformando os dados

#
save(dataset_integrado_low, dataset_limpo_low,dataset_transformado_low,stratified_sample_0.1_low, file = "data_low.RData")



# Atributos Categóricos
dataset_atrib_categorico <- dataset_transformado[,c("siglaempresa","aeroportopartida","aeroportochegada", 
                                                 "diasemana", "mes", "conditions_dep", "conditions_arr", "time_dep", 
                                                 "time_arr", "wdirection_dep", "wdirection_arr")]

# Atributos Não Categóricos

dataset_atrib_nao_categorico <- dataset_transformado[,c("alvo", "temp_dep","dew_dep","pressure_dep","humid_dep",
                                                     "wspeed_dep","temp_arr","dew_arr","pressure_arr","humid_arr", "wspeed_arr")]

# Transformação de dados Categóricos

dataset_atrib_categorico <- transformCategoricData(dataset_atrib_categorico, c("siglaempresa","aeroportopartida",
                                                                               "aeroportochegada","mes","diasemana","time_dep","time_arr","conditions_dep",
                                                                               "conditions_arr","wdirection_dep","wdirection_arr"))

# Normalização de dados Não Categóricos

dataset_atrib_nao_categorico <- normalize.minmax(dataset_atrib_nao_categorico)
dataset_atrib_nao_categorico <- dataset_atrib_nao_categorico[[1]]

# Junção de Categóricos e Não Categóricos

dataset_preprocessado <- cbind(dataset_atrib_nao_categorico, dataset_atrib_categorico)

save(dataset_preprocessado,file="dataset_preprocessado.Rda")
load("dataset_preprocessado.Rda")

dataset_preprocessado_low<-sample.stratified(data=dataset_preprocessado, clabel="alvo", perc=0.1)
save(dataset_preprocessado_low,file="dataset_preprocessado_low.Rda")

#Conjunto de teste e treino
#training_test <- divideTrainAndTest(data = dataset_preprocessado, percentual = 0.8)
#training_data <- training_test[[1]]
#test_data <- training_test[[2]]

training_test_low <- divideTrainAndTest(data = dataset_preprocessado_low, percentual = 0.8)
training_data_low <- training_test_low[[1]]
test_data_low <- training_test_low[[2]]

#save(training_test,file="training_test.Rda")
#save(training_data,file="training_data.Rda")
#load("training_data.Rda")
#save(test_data ,file="test_data .Rda")

save(training_test_low,file="training_test_low.Rda")
save(training_data_low,file="training_data_low.Rda")
load("training_data_low.Rda")
save(test_data_low,file="test_data_low.Rda")

####################################################################
######################################################################
#Ajustando caractecres especiais#
library(stringr)
colnames(training_data_low) <- str_replace_all(colnames(training_data_low),"[ç]","c")
colnames(training_data_low) <- str_replace_all(colnames(training_data_low),"á","a")

#####################################################################
####################################################################
#Seleção de Atributos#

#LASSO
#train_lasso <- fs.lasso(training_data, "alvo") [[1]]
#save(train_lasso ,file="train_lasso.Rda")
train_lasso_low <- fs.lasso(training_data_low, "alvo") [[1]]
save(train_lasso_low ,file="train_lasso_low.Rda")


#train_cfs <- fs.cfs(training_data, "alvo") [[1]]
#save(train_cfs ,file="train_cfs.Rda")
train_cfs_low <- fs.cfs(training_data_low, "alvo") [[1]]
save(train_cfs_low ,file="train_cfs_low.Rda")

#train_pca <- dt.pca(training_data, "alvo") [[1]]
#save(train_pca ,file="train_pca.Rda")

train_pca_low <- dt.pca(training_data_low, "alvo") [[1]]
save(train_pca_low ,file="train_pca_low.Rda")


#completo deu erro "java.lang.OutOfMemoryError: Java heap space "
options(java.parameters = "-Xmx8g")
#train_ig <- fs.ig(training_data, "alvo") [[1]]
#save(train_ig ,file="train_ig.Rda")
train_ig_low <- fs.ig(training_data_low, "alvo") [[1]]
save(train_ig_low ,file="train_ig_low.Rda")

############################################################################
#Geração do Modelo e Teste
sc <- colnames(train_lasso_low)
test_lasso_low <- test_data_low[,sc]

bestConfig_mlp_train_lasso <- bestConfiguration(train_lasso_low, "alvo")


# geração do modelo
mlp_net_train_lasso <- class_mlp_nnet(train_lasso_low, test_lasso_low, "alvo", 
                                      decay=bestConfig_mlp_train_lasso[2], 
                                      neurons=bestConfig_mlp_train_lasso[3])

mlp_net_train_lasso <- class_mlp_nnet(train_lasso_low,test_lasso_low, "alvo", decay=0.01, neurons=9)
#AVALIACAO DO MODELO
accuracy(mlp_net_train_lasso)
sensitivity(mlp_net_train_lasso)
            
