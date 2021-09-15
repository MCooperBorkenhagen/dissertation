
# syllabics
syllabics = mono %>% 
  filter(model == 'LSTM' & stage == 'Late') %>% 
  select(word, elp_rt, consistency, body_neighbors)

syllabics %>% 
  ggplot(aes(consistency, elp_rt)) +
  geom_point() +
  geom_smooth(method = 'lm')




# Taraban words, Plaut effects (fig 5)

mono %>% 
  filter(model == 'LSTM' & stage == 'Middle') %>% 
  right_join(taraban) %>% 
  filter(threek == T) %>% 
  group_by(frequency, condition) %>% 
  summarise(acc = mean(loss, na.rm = T)) %>% 
  ggplot(aes(frequency, acc, shape = condition, group = condition)) +
  geom_point() +
  geom_line()



mono %>% 
  filter(model == 'LSTM' & stage == 'Middle') %>% 
  right_join(taraban) %>% 
  filter(threek == T) %>%
  group_by(frequency) %>% 
  ggplot(aes(frequency, freq, group = condition)) +
  geom_point()
  
# consistency with rt

STAGE = 'Middle'

elp_words = mono %>% 
  select(word, elp_acc) %>% 
  filter(!is.na(elp_acc)) %>% 
  pull(word) %>% 
  unique()


 
descriptives = mono %>%
  filter(train_test == 'train') %>% 
  filter(stage == STAGE) %>% 
  group_by(model, stage) %>% 
  summarise(MEAN = mean(accuracy),
            SD = sd(accuracy))


COLORS = c('LSTM' = 'firebrick', 'Feedforward' = 'goldenrod', '(D) Time-varying' = 'firebrick', '(C) Feedforward' = 'goldenrod', 'Human - Accuracy' = 'Black', 'Human - RT' = 'Grey20', '(A) Human - Accuracy' = 'Black', '(B) Human - RT' = 'Grey20')

descriptives_elp_acc = mono %>% 
  filter(word %in% elp_words) %>% 
  filter(stage == STAGE & model == 'LSTM') %>% #arbitrary because values repeat
  filter(train_test == 'train') %>% 
  summarise(accMEAN = mean(elp_acc),
            accSD = sd(elp_acc),
            rtMEAN = mean(elp_rt),
            rtSD = sd(elp_rt)) %>% 
  mutate(model = '(A) Human - Accuracy')

d_elp_acc = mono %>% 
  filter(word %in% elp_words) %>% 
  filter(stage == STAGE & model == 'LSTM') %>% #arbitrary because values repeat
  filter(train_test == 'train') %>% 
  mutate(model = '(A) Human - Accuracy') %>% 
  left_join(descriptives_elp_acc) %>% 
  mutate(acc = -(elp_acc-accMEAN)/accSD) %>% 
  select(word, model, acc, consistency)


descriptives_elp_rt = mono %>% 
  filter(word %in% elp_words) %>% 
  filter(stage == STAGE & model == 'LSTM') %>% #arbitrary because values repeat
  filter(train_test == 'train') %>% 
  summarise(accMEAN = mean(elp_acc),
            accSD = sd(elp_acc),
            rtMEAN = mean(elp_rt),
            rtSD = sd(elp_rt)) %>% 
  mutate(model = '(B) Human - RT')

d_elp_rt = mono %>% 
  filter(word %in% elp_words) %>% 
  filter(stage == STAGE & model == 'LSTM') %>% #arbitrary because values repeat
  filter(train_test == 'train') %>% 
  mutate(model = '(B) Human - RT') %>% 
  left_join(descriptives_elp_rt) %>% 
  mutate(acc = (elp_rt-rtMEAN)/rtSD) %>% 
  select(word, model, acc, consistency)



mono %>% 
  filter(stage == STAGE & train_test == 'train') %>% 
  left_join(descriptives, by = c('model')) %>% 
  mutate(acc = (accuracy-MEAN)/SD) %>% 
  filter(word %in% elp_words) %>% 
  select(word, model, acc, consistency) %>% 
  rbind(d_elp_acc) %>%
  rbind(d_elp_rt) %>% 
  mutate(model = case_when(model == 'LSTM' ~ '(D) Time-varying',
                           model == 'Feedforward' ~ '(C) Feedforward',
                           TRUE ~ model)) %>% 
  ggplot(aes(consistency, acc, color = model)) +
  geom_point(size = .2) +
  geom_smooth(method = 'lm', color = 'grey32', span = 1/3) +
  scale_color_manual(values = COLORS) +
  facet_grid(~model) +
  labs(x = 'Consistency (body-rime)', y = 'Processing difficulty') +
  theme_bw() +
  theme(legend.position = 'none') +
  ylim(c(-1, 5))




mono %>% 
  filter(model == 'LSTM' & train_test == 'train') %>% 
  filter(freq == 55 | freq == 260269) %>% 
  select(word, freq, freq_scaled)


mono %>% 
  filter(model == 'LSTM' & train_test == 'train') %>% 
  ggplot(aes(freq_scaled)) +
  geom_histogram() +
  geom_vline(xintercept = .708) +
  geom_vline(xintercept = .266)

mono %>% 
  filter(word == 'park') %>% 
  select(word, freq, freq_scaled)


