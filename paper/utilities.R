
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

