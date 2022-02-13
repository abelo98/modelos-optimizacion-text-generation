library(keras)
library(tensorflow)

gen <-  load_model_hdf5("generator_model.h5",compile=FALSE)
summary(gen)

db <- file.choose()
dataset<-read.spss(db, to.data.frame = TRUE)

start_i <- 167
stop_i <- 310
size <- 30
seed_text<-"Sleeping"

map_seqs2char<-map_seq2index(data_set = dataset,start_idx = start_i,stop_idx = stop_i)

text_generation(seed_text,map_seqs2char,size,gen)
