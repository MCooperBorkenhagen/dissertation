
phonlen = function(x, delimiter='-'){
  return(str_count(x, delimiter)+1)
}


nearest_word = function(dm, word, return_rank=TRUE){
  
  i = match(word, rownames(dm))
  n = names(sort(dm[i, ]))
  if (return_rank){return(match(word, n))}
  else {return(n[1])}
}




nearest_print = function(dm, word, n=10){
  
  
  i = match(word, rownames(dm))
  top = sort(dm[i, ])[1:n]
  
  print(paste('Target word is:', word))
  print(paste('Nearest word is:', names(top[1])))
  print('Top words:')
  print(top)
}


phon = function(d, orth){
  
  i = match(orth, d$word)
  return(d$phon[i])
  
}


freq_for_word = function(d, word){
  
  i = match(word, d$word)
  return(d$freq[i])
  
}


nearest_word_df = function(distance_matrix, words, cols = c('word', 'nearest_phon_rank', 'nearest_phon')){
  
  rownames(distance_matrix) = words
  colnames(distance_matrix) = words
  
  nw = data.frame(matrix(nrow = length(words), ncol = 3))
  colnames(nw) = cols
  
  row = 1
  for (word in words){
    rank = nearest_word(distance_matrix, word, return_rank = T)
    nearest = nearest_word(distance_matrix, word, return_rank = F)
    
    nw$word[row] = word
    nw$nearest_phon_rank[row] = rank
    nw$nearest_phon[row] = nearest
    
    row = row + 1
  }
  
  return(nw)  
}