library(keras)
library(kerasR)

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
    layer_dense(vocab_size,activation='sigmoid',)
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
      recurrent_initializer='glorot_uniform',
      recurrent_activation='sigmoid',
      stateful = TRUE
    ) %>%
    
    
    layer_dropout(rate = 0.5)%>%
    layer_dense(vocab_size)
  return(model)
}



# ### Prediction of a generated song ###
# 
# generate_text<-function(model, start_string, map_seqs2char, generation_length=1000){
#   # Evaluation step (generating ABC text using the learned RNN model)
# 
# input_eval <- c(map_seqs2char$start_string)
# input_eval <- expand_dims(input_eval,0)
# 
# # Empty string to store our results
# text_generated = list(start_string)
# 
# # Here batch size == 1
# model %>% reset_states()
# 
# for (i in 0:generation_length){
#   predictions <- model(input_eval)
# 
#   predictions < tf$squeeze(predictions,0L)
# 
# 
# predicted_id <-  tf$random$categorical(predictions, num_samples=1)
# as.array(predicted_id)
# 
# # Pass the prediction along with the previous hidden state
# #   as the next inputs to the model
# input_eval = expand_dims(predicted_id, 0)
# 
# text_generated[i]<-(names(map_seqs2char)[predicted_id[1]]) 
# 
# }
# return(text_generated)
# }