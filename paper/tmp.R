


STAGE = 'Late'
mono %>% 
  left_join(descriptives_model, by = c('model', 'stage')) %>% 
  group_by(model, stage) %>% 
  mutate(acc = (nearest_phon_rank-MEAN)/SD) %>% 
  ungroup() %>% 
  filter(train_test == 'train') %>% 
  filter(stage == STAGE) %>% 
  #filter(acc > min(acc)) %>%  
  nrow()
  
  
  #filter(word %in% elp_words) %>% 
  #select(word, model, acc, freq_scaled) %>% 
  filter(acc < 3) %>% 
  ggplot(aes(consistency, acc, color = model)) +
  geom_point(size = .6) +
  geom_smooth(method = 'lm', color = 'grey32') +
  scale_color_manual(values = COLORS) +
  facet_grid(~model) +
  labs(x = 'Frequency (scaled)', y = 'Error') +
  theme_bw() +
  theme(legend.position = 'none')


mono %>% 
  filter(model == 'LSTM') %>% 
  filter(stage == STAGE) %>%
  ggplot(aes(consistency, elp_acc)) +
  geom_point() +
  geom_smooth()


