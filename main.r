source("utils.R")
source("network_architecture.R")
library(tensorflow)
library(keras)
library(foreign)

db <- file.choose()
dataset<-read.spss(db, to.data.frame = TRUE)

map_seqs2char<-map_seq2index(data_set = dataset,start_idx = 167,stop_idx = 310)
vocab <- tail(map_seqs2char,1)[[1]] + 1
map_seqs2char

f_dataSet <- convert_dataSet(dataset,167,310)
f_dataSet

dataSet2tensor <- make_tensor(map_seqs2char,f_dataSet)
training_set <- get_batch(dataSet2tensor,60 ,4000)
input_train <- training_set$x
output_train <- training_set$y

batch <- 16
emb_dim <- 64 
rnn_u <- 64
learning_rate<-0.0001
epochs <- 60
validation_perct<-0.2
checkpoint_path <- "checkpoints/cp.ckpt"

model<- build_model_lstm(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=batch)
summary(model)
train_model(model,epochs,batch,learning_rate,validation_perct,input_train,output_train)


gen_model<- build_model_lstm(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=1)
gen_model %>% load_model_weights_tf(checkpoint_path)
gen_model %>% reset_states()

gen_model %>% save_model_hdf5("generator_model.h5")
summary(gen_model)

gen <-  load_model_hdf5("generator_model.h5")
summary(gen)


# 
# #
# # db <- file.choose()
# # dataset <- read.spss(db, to.data.frame=TRUE)
# # data(dataset)
# 
# 
