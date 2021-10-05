require(kableExtra)
require(glue)


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



z = function(x, MEAN, SD){
  return((x-MEAN)/SD)
}



squarebracket = function(x, y, digits = 3){
  
  return(paste('[', round(x, digits = digits), ', ', round(y, digits = 3), ']', sep = ''))
  
  
}


presentp = function(p_value, round_places = 3){
  return(case_when(p_value < .05 & p_value >= .01 ~ glue('< .05'),
                   p_value < .01 & p_value >= .001 ~ glue('< .01'),
                   p_value < .001 ~ glue('< .001'),
                   TRUE ~ glue('= {round(p_value, digits = round_places)}')))
}


sig_threshold = function(p, b){
  return(ifelse(p < .05, glue('**', b, '**'), glue(b)))
}


sig_level = function(bs, ps, digits = 3){
  out = c()
  stopifnot(length(bs) == length(ps))
  for (i in seq(length(bs))){
    out = c(out, sig_threshold(ps[i], round(bs[i], digits = digits)))}
  return(out)}


model_to_table = function(model, x_names, confints, car_anova, n_sigmas = 3, notes = '', caption = '', format = 'pandoc', include_R_notes = TRUE){
  
  modeldata = summary(model)
  
  if (missing(car_anova)){
    tabledata = data.frame(modeldata$coefficients) %>% 
      mutate(Estimate = sig_level(Estimate, Pr...t..),
             Pr...t.. = presentp(Pr...t..))} 
  if (!missing(car_anova)) {
    tabledata = data.frame(modeldata$coefficients) %>% 
      mutate(p = car_anova[4][[1]]) %>% 
      mutate(Estimate = sig_level(Estimate, p),
             p = presentp(p))}
  
  confints = confints[(n_sigmas+1):nrow(confints), ]
  tabledata$CI = squarebracket(confints[,1], confints[,2], digits = 3)
  
  tabledata$Predictor = x_names
  
  rownames(tabledata) = NULL

  names(tabledata) = c('*b*', '*SE*', '*t*', '*p*', '95% CI', 'Predictor')
  
  if (include_R_notes){
    notes = paste(notes, 'Model estimated using lm() in R with confidence intervals estimated using confint(), both methods from the native R stats package [@R2021]. Bold parameter estimates indicate significant *p*-values below the alpha threshold of .05.')
  }
  
  TABLE = tabledata %>% 
    select(Predictor, everything()) %>% 
    apa_table(digits = 3, format = format, align = 'l', landscape = T,
              note = notes,
              caption = caption)
  
  return(TABLE)}
