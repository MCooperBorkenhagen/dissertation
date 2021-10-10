taraban_testmode %>% 
  filter(taraban == 'nonword') %>% nrow()
  filter(phonemes_proportion < 1) %>% 
  select(word, phon_read)  %>% 
  pull(word) -> pronounced_wrong


taraban_crossval %>% 
  filter(epoch == 27 & run_id ==1) %>% 
  filter(freq_plaut == 'low' & plaut == 'reg_consistent') %>% 
  filter(mse > .02) %>% 
  pull(word)
  #ggplot(aes(mse)) +
  #geom_histogram()
  
  
  




