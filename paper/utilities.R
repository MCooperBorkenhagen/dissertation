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



difference = function(x){
  stopifnot(length(x)==2)
  return(x[1]-x[2])
}



summarow = function(df, name, na.rm = T){
  
  desc = df %>% 
    summarise(M = mean(!!as.name(name), na.rm = na.rm),
              SD = sd(!!as.name(name), na.rm = na.rm),
              MIN = min(!!as.name(name), na.rm = na.rm),
              MAX = max(!!as.name(name), na.rm = na.rm)) %>% 
    mutate(var = name) %>% 
    select(var, everything())
  
  return(desc)}


summarows = function(df, names, newnames = c(), na.rm = T, digits = 2, manuscript = T, notes = '', caption = '', format = 'pandoc'){
  
  desc = data.frame(matrix(nrow = 0, ncol = 5))
  
  
  for (name in names){
    
    desc = rbind(desc, summarow(df, name, na.rm = na.rm))
    
  }
  
  names(desc) = c('Predictor', '*M*', '*SD*', 'min', 'max')
  
  if (length(newnames) > 0){
    stopifnot(length(names) == length(newnames))
    
    for (i in seq_len(length(newnames))){
      desc$Predictor[i] = newnames[i]}}
  
  
  if (manuscript){
  TABLE = desc %>% 
    apa_table(digits = digits, format = format, align = 'l', landscape = T,
              note = notes,
              caption = caption)
  
  return(TABLE)}
  
  else {return(desc)}
  
}



get_petasq = function(model, residuals = F){
  
  pes = etasq(model)$`Partial eta^2`
  
  if (!residuals){
    return(pes[1:length(pes)-1])
  }
  else {return(pes)}
  
}

get_ps = function(model, intercept=F){
  
  s = summary(model)
  if (intercept){
    return(s$coefficients[,4])}
  else {
    return(s$coefficients[,4][2:nrow(s$coefficients)])
  }
  
}




get_bs = function(model, intercept=F){
  s = summary(model)
  if (intercept){
    return(as.numeric(s$coefficients[,1]))}
  else {
    return(as.numeric(s$coefficients[,1][2:nrow(s$coefficients)]))
  }
  
}



cortests = function(df, dv, vars, value, digits=2){
  
  #' @param df Data containing dv and vars for application of cor.test
  #' @param dv Dependent measure for comparison across all vars supplied
  #' @param vars Variables for the correlations (with dv)
  #' @param value Possible values to return with cor.test: "statistic", "parameter", "p.value", "estimate", "conf.int"
  #' @param digits For rounding if value is supplied with conf.int
  
  out = c()
  for (var in vars){
    
    cr = cor.test(df[[dv]], df[[var]])[[value]]
    if (value == 'conf.int'){
      cr = paste('[', paste(round(cr, digits=digits), collapse = ', '), ']', sep = '')
      
    }
    out = c(out, cr[[1]])
    
  }
  return(out)
}
