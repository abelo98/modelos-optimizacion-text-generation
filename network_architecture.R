library(keras)
library(kerasR)

# Create checkpoint callback
checkpoint_path <- "checkpoints/cp.ckpt"

cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 1
)

build_model_simple_rnn <- function(vocab_size, embedding_dim, rnn_units, batch_size){
  model <- keras_model_sequential() %>%
    layer_embedding(input_dim = vocab_size, 
                    output_dim = embedding_dim,
                    batch_size = batch_size) %>%
    layer_dropout(rate = 0.5)%>%
    layer_simple_rnn(
      units = rnn_units, 
      return_sequences=TRUE,
      recurrent_initializer='glorot_uniform',
      kernel_regularizer = regularizer_l2(0.001),
      stateful = TRUE
    ) %>%
    layer_dropout(rate = 0.5)%>%
    layer_simple_rnn(
      units = rnn_units,
      return_sequences=TRUE,
      kernel_regularizer = regularizer_l2(0.001),
      recurrent_initializer='glorot_uniform',
      stateful = TRUE
    ) %>%
    layer_dropout(rate = 0.5)%>%
    layer_dense(vocab_size,activation='sigmoid')
  
  model %>% compile(
    optimizer=optimizer_adam(learning_rate = learning_rate),
    loss = "sparse_categorical_crossentropy",
    metrics = c("acc")
  )
  return(model)
}

build_model_lstm <- function(vocab_size, embedding_dim, rnn_units, batch_size){
  model <- keras_model_sequential() %>%
    layer_embedding(input_dim = vocab_size, 
                    output_dim = embedding_dim,
                    batch_size = batch_size) %>%
    layer_dropout(rate = 0.5)%>%

    layer_lstm(
      units = rnn_units, 
      return_sequences=TRUE,
      kernel_regularizer = regularizer_l2(0.001),
      recurrent_initializer='glorot_uniform',
      recurrent_activation='sigmoid',
      stateful = TRUE
    ) %>%
    
    layer_dropout(rate = 0.5)%>%
    
    layer_dense(vocab_size,activation = "sigmoid")
  
  model %>% compile(
    optimizer=optimizer_adam(learning_rate = learning_rate),
    loss = "sparse_categorical_crossentropy",
    metrics = c("acc")
  )
  return(model)
}

train_model <- function(model,no_epochs,size_batch,learning_rate,validtion,input_train, output_train){
  history <- model %>% fit(
    input_train, output_train,
    epochs = no_epochs,
    batch_size = size_batch,
    callbacks = list(cp_callback), # pass callback to training,
    verbose = 2,
    validation_split = validtion
  )
  plot(history)
  
  model %>% save_model_hdf5("training_model.h5")
}



