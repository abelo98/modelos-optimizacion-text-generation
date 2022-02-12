library(keras)
library(tensorflow)

gen <-  load_model_hdf5("generator_model.h5")
summary(gen)


input_eval <- list(map_seqs2char$Sleeping)
input_eval <- tf$expand_dims(input_eval,0L)
text_generated <- list(seed_word)
size <- 30

for (i in 2:size) {
  predictions <- gen(input_eval)
  predictions < tf$squeeze(predictions,0L)
  
  predicted_id <- tf$random$categorical(predictions[1,,], num_samples=1L)
  input_eval<-predicted_id
  converted_pred_id <- as.double(predicted_id[1,1]) + 1
  text_generated[i]<-(names(map_seqs2char)[converted_pred_id])
}
print(text_generated)
write.csv(text_generated, "data_gen.csv")
write.table(text_generated, file = "data_gen.txt",col.names = FALSE)