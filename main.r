source("utils.R")
source("network_architecture.R")
library(tensorflow)
library(keras)

map_seqs2char<-map_seq2index(data_set = dataset,start_idx = 167,stop_idx = 310)
vocab <- tail(map_seqs2char,1)[[1]] + 1


f_dataSet <- convert_dataSet(dataset,167,310)

dataSet2tensor <- make_tensor(map_seqs2char,f_dataSet)
training_set <- get_batch(dataSet2tensor, 144,4360)
input_train <- training_set$x
output_train <- training_set$y

batch <- 100
emb_dim <- 39 
rnn_u <- 50
learning_rate<-0.01
epochs <- 80
validation_perct<-0.2
checkpoint_path <- "checkpoints/cp.ckpt"

model<- build_model_lstm(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=batch)
summary(model)
compile_and_train(model,epochs,batch,learning_rate,validation_perct,input_train,output_train)

plot(history)

gen_model<- build_model_lstm(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=1)


gen_model %>% compile(
  optimizer=optimizer_adam(learning_rate = learning_rate),
  loss = "sparse_categorical_crossentropy",
  metrics = c("acc")
)


gen_model %>% load_model_weights_tf(checkpoint_path)

# gen_model %>% evaluate(input_eval, output_eval, verbose = 0)
# 
# gen_model %>% keras$Model$build(tf$TensorShape(c(1L,NULL)))
# gen_model %>% reset_states()

gen_model %>% save_model_tf("generator_model")

summary(gen_model)


  
input_eval <- list(map_seqs2char$"Sleeping")
input_eval <- tf$expand_dims(input_eval,0L)
text_generated <- list(seed_word)


for (i in 2:40) {
  predictions <- gen_model(input_eval)
  predictions < tf$squeeze(predictions,0L)
  
  predicted_id <- tf$random$categorical(predictions[1,,], num_samples=1L)
  input_eval<-predicted_id
  converted_pred_id <- as.double(predicted_id[1,1]) + 1
  text_generated[i]<-(names(map_seqs2char)[converted_pred_id])
}
print(text_generated)
write.csv(text_generated, "data_gen.csv")
write.table(text_generated, "data_gen.txt")
  


# 
# #
# # db <- file.choose()
# # dataset <- read.spss(db, to.data.frame=TRUE)
# # data(dataset)
# 
# 
