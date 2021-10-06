taraban_testmode %>% 
  filter(taraban == 'nonword') %>% nrow()
  filter(phonemes_proportion < 1) %>% 
  select(word, phon_read)  %>% 
  pull(word) -> pronounced_wrong


tmp = taraban_crossval %>% 
  filter(epoch == 27 & run_id ==1) %>% 
  select(word, taraban, freq_taraban, taraban_test)
  
  
  
  
taraban_crossval %>% 
  filter(taraban %in% c('reg_inconsistent', 'exception')) %>% 
  filter(epoch == 27) %>% 
  group_by(taraban, freq_taraban, taraban_test) %>% 
  summarise(mse = mean(mse))
  ggplot(aes(freq_taraban, mse, fill=taraban)) +
  geom_bar(stat='summary', position = position_dodge())


taraban_means %>% 
  group_by(condition, frequency) %>% 
  summarise(latency = -(latency - lag(latency, default = latency[1]))) %>% 
  ggplot(aes(frequency, latency, fill = condition)) +
  geom_bar(stat = 'identity', position = position_dodge())


