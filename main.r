source("utils.R")
source("network_architecture.R")
library(tensorflow)
library(keras)

map_seqs2char<-map_seq2index(data_set = dataset,start_idx = 167,stop_idx = 310)
vocab <- tail(map_seqs2char,1)[[1]] + 1


f_dataSet <- convert_dataSet(dataset,167,310)

dataSet2tensor <- make_tensor(map_seqs2char,f_dataSet)
training_set <- get_batch(dataSet2tensor, 100,4000)
input_train <- training_set$x
output_train <- training_set$y

# training_set$x
# 
# mean <- apply(input_train, 2, mean)
# std <- apply(input_train, 2, sd)
# input_train <- scale(input_train, center = mean, scale = std)
# 
# mean <- apply(output_train, 2, mean)
# std <- apply(output_train, 2, sd)
# output_train <- scale(output_train, center = mean, scale = std)

batch <- 100
emb_dim <- 38 
rnn_u <- 32
checkpoint_path <- "checkpoints/cp.ckpt"

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 1
)

callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.1,
  patience = 10,
  verbose = 0,
  mode = c("auto", "min", "max"),
  min_delta = 1e-04,
  cooldown = 0,
  min_lr = 0
)

# model <-build_model_simple_rnn(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=batch)
model2<- build_model_lstm(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=batch)
summary(model2)
# 
# pred <- model(batch_xy$x_batch)
# example_batch_loss <- compute_loss(batch_xy$y_batch, pred)
# 
# example_batch_loss
model2 %>% compile(
  optimizer=optimizer_adam(learning_rate = 0.01),
  loss = "sparse_categorical_crossentropy",
  metrics = c("acc")
)


history <- model2 %>% fit(
  input_train, output_train,
  steps_per_epoch = 100,
  epochs = 10,
  batch_size = batch,
  callbacks = list(cp_callback), # pass callback to training,
  verbose = 2,
  validation_split = 0.2
)

checkpoint_dir <- "./training_checkpoints"
checkpoint_prefix <- file.path(imdb_dir, "my_ckpt")
model2 %>% save_model_weights_tf(checkpoint_path)
# 
# fresh_model <- build_model_simple_rnn(vocab, embedding_dim=embedding_dim, rnn_units=rnn_u, batch_size=batch)
# fresh_model %>% compile(
#   optimizer=optimizer_adam(learning_rate = 0.05),
#   loss = "sparse_categorical_crossentropy",
#   metrics = c("acc")
# )
# x1<-tf$expand_dims(input_train[1:32,],0L)
# x2<-tf$expand_dims(output_train[1:32,],0L)
# x1
# 
# fresh_model %>% evaluate(x1,x2, verbose = 0)
# 
# fresh_model %>% load_model_weights_tf(filepath = checkpoint_prefix)
# fresh_model %>% evaluate(input_train, output_train, verbose = 0)
# 

plot(history)

# model <- build_model_simple_rnn(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=1)
model2<- build_model_lstm(vocab, embedding_dim=emb_dim, rnn_units=rnn_u, batch_size=1)
# # Restore the model weights for the last checkpoint after training
model2 %>% load_model_weights_tf(checkpoint_prefix)
model2 %>% keras$Model$build(tf$TensorShape(c(1L,NULL)))

summary(model2)


input_eval <- list(map_seqs2char$"Sleeping")
input_eval
input_eval <- tf$expand_dims(input_eval,0L)
input_eval
dim(input_eval)

# Empty string to store our results
text_generated = list("Reading including e-books ")

# Here batch size == 1
model2 %>% reset_states()
summary(model)

for (i in 2:160) {
predictions <- model2(input_eval)
predictions < tf$squeeze(predictions,0L)

# print(predictions[1,,])

predicted_id <- tf$random$categorical(predictions[1,,], num_samples=1L)
input_eval<-predicted_id
# predicted_id
converted_pred_id <- as.double(predicted_id[1,1]) + 1
# converted_pred_id
# print(converted_pred_id)
text_generated[i]<-(names(map_seqs2char)[converted_pred_id])

}
print(text_generated)


# gen<-generate_text(model = model,start_string = "Sleeping",map_seqs2char,generation_length = 20)
# gen





# x <- get_batch(dataSet2tensor, 100,100)
# typeof(x$x) 
# x$x
# dim(x$x)
# predictions<-model(x$x)
# p <- predictions[[1]]
# 
# 
# sampled_indices <-tf$random$categorical(predictions[[1]], num_samples=1L)
# sampled_indices <- tf$squeeze(sampled_indices,axis=-1)
# sampled_indices
# s_array <- as.array(sampled_indices)
# typeof(s_array[1])
# # 
# # 
# 
# # 
# # 
# # summary(model)
# 
# #
# # db <- file.choose()
# # dataset <- read.spss(db, to.data.frame=TRUE)
# # data(dataset)
# 
# 
