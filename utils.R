library(foreign)
library(TraMineR)
# library(comprehenr)

fix_seq<-function(seq){
  fseq <-gsub("  ","",seq) 
  if (substring(y,nchar(y),nchar(y)) == " "){
    y <- substr(y,1,nchar(y)-1)
  }
  return(fseq)
}

convert_dataSet<-function(data_set, start_cols, stop_cols){
  cols <- (stop_cols + 1) - start_cols  
  fixed_dataset<-matrix(nrow = nrow(data_set), ncol = cols)
  for (i in 1:nrow(data_set)) {
    for (j in 1:cols) {
      seq <- as.character(data_set[i,(start_cols-1)+j])
      fixed_dataset[i,j] <- fix_seq(seq)
    }
  }
  return(fixed_dataset)
}

clean_data<-function(seq_opts){
  neq_seq_opts<-c()
  for (i in 1:length(seq_opts)) {
    seq_opts[i] <- fix_seq(seq_opts[i])
  }
  return(seq_opts)
}

map_seq2index<-function(data_set,start_idx,stop_idx){
  seq2index<-list()
  seqs<-seqstatl(dataset[, start_idx:stop_idx])
  clean_seqs<-clean_data(seqs)
  
  for (seq in clean_seqs) {
    if (!seq %in% names(seq2index)){
      seq2index[[seq]]<-length(seq2index) + 1
    }
  }
  return(seq2index)
}

make_tensor<-function(map, data_set){
  rows = nrow(data_set)
  cols = ncol(data_set)
  tensor <- matrix(nrow = rows,ncol = cols)
  
  for (i in 1:rows) {
    for (j in 1:cols) {
      tensor[i,j] <- map[[data_set[i,j]]]
    }
  }
  return(tensor)
}

get_batch<-function(vectorized_data, seq_legth, batch_size){
  size_of_seq <- ncol(vectorized_data)
  selected_seq <- sample.int(nrow(vectorized_data), batch_size,replace = TRUE)
  st_indx_in_seq <- sample.int(size_of_seq - seq_legth, batch_size,replace = TRUE)
  
  x_batch <- matrix(nrow = batch_size, ncol = seq_legth)
  y_batch <- matrix(nrow = batch_size, ncol = seq_legth)
  
  for (i in 1:batch_size) {
    st <- st_indx_in_seq[i]
    x_batch[i,] <- vectorized_data[selected_seq[i], st:(st + seq_legth - 1)]
    y_batch[i,] <- vectorized_data[selected_seq[i], (st+1):(st + seq_legth)]
  }
  return(list("x" = x_batch, "y" = y_batch))
}
