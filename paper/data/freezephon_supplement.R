


# possibly useful later if you want to calculate error over epoch as a functin of freeze time:
CUTOFFS = data.frame(unique(freezephon$when_freeze), MIN, MAX) %>% 
  rename(when_freeze = unique.freezephon.when_freeze.) %>% 
  replace_na(list(MIN = 0, MAX = 0)) %>% 
  mutate(SPAN = MAX - MIN)

descriptives = freezephon %>% 
  group_by(which_freeze, when_freeze) %>% 
  summarise(accuracy_m = mean(binary_acc),
            accuracy_sd = sd(binary_acc))


freezephon %>% 
  left_join(CUTOFFS) %>% 
  mutate(when_freeze_num = case_when(when_freeze == 'never' ~ 63,
                                     when_freeze == 'early' ~ 18,
                                     when_freeze == 'middle' ~ 36,
                                     when_freeze == 'late' ~ 54,
                                     when_freeze == 'always' ~ 9)) %>% 
  mutate(phon_frozen = case_when(SPAN == 0 ~ FALSE,
                                 epoch < when_freeze_num ~ FALSE,
                                 epoch >= when_freeze_num ~ TRUE)) %>% 
  filter(phon_frozen == TRUE) %>% 
  filter(epoch == MIN | epoch == MAX) %>% 
  left_join(descriptives) %>% 
  group_by(which_freeze, when_freeze, epoch) %>% 
  summarise(MIN = first(MIN),
            MAX = first(MAX),
            how_long_frozen = first(SPAN), 
            accuracy_m = first(accuracy_m),
            accuracy_sd = first(accuracy_sd),
            accuracy = mean(binary_acc),
            accuracy_z = (accuracy-accuracy_m)/accuracy_sd) %>% 
  ungroup() %>% 
  group_by(which_freeze, when_freeze) %>% 
  arrange(desc(accuracy)) %>% 
  summarise(how_long_frozen = first(how_long_frozen),
            accuracy_increase = difference(accuracy_z)) %>% 
  ungroup() %>% 
  group_by(which_freeze, when_freeze) %>% 
  summarize(growth_per_epoch = accuracy_increase/how_long_frozen) %>% 
  mutate(which_freeze = case_when(which_freeze == 'all' ~ 'Stopped all phonology',
                                  which_freeze == 'lstm' ~ 'Stopped phon LSTM only'),
         condition = case_when(when_freeze == 'always' ~ WHENS[1],
                               when_freeze == 'early' ~ WHENS[2],
                               when_freeze == 'middle' ~ WHENS[3],
                               when_freeze == 'late' ~ WHENS[4],
                               when_freeze == 'never' ~ WHENS[5])) %>% 
  ggplot(aes(condition, growth_per_epoch)) +
  geom_bar(stat = 'identity', fill = 'grey29', color = 'black') +
  facet_grid(~which_freeze) +
  scale_y_continuous(labels = comma) +
  labs(x = 'Period of training when learning stops',
       y = 'Accuracy growth per epoch after stop (standardized)') +
  theme(legend.position = 'none') +
  theme_apa()
