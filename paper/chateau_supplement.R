
# This script compiles the visuals for the two-by-three panel figure on the Chateau & Jared (2003) experiments





## Experiment 2 ----
# Model data
N = multi %>% filter(epoch == 45) %>% filter(!is.na(chateauB_consistency)) %>% nrow()

d = multi %>% 
  filter(epoch == 45) %>% 
  filter(!is.na(chateauB_consistency)) %>% 
  mutate(consistency = case_when(chateauB_consistency == 'high' ~ 'High',
                                 chateauB_consistency == 'low' ~ 'Low'),
         frequency = case_when(chateauB_frequency == 'high' ~ 'High',
                               chateauB_frequency == 'low' ~ 'Low'))

multi %>% 
  filter(epoch == 45) %>% 
  filter(!is.na(chateauB_consistency)) %>% 
  mutate(consistency = case_when(chateauB_consistency == 'low' ~ -.5,
                                 chateauB_consistency == 'high' ~ .5),
         frequency = case_when(chateauB_frequency == 'low' ~ -.5,
                               chateauB_frequency == 'high' ~ .5)) %>% 
  lm(mse ~ consistency*frequency, data = .) %>% 
  summary()

COLORS = c('High' = 'grey86', 'Low' = 'black')

multi %>% 
  filter(epoch == 45) %>% 
  filter(!is.na(chateauB_consistency)) %>% 
  group_by(chateauB_consistency, chateauB_frequency) %>% 
  summarise(SD = sd(mse),
            mse = mean(mse),
            SEM = SD/sqrt(N)) %>% 
  mutate(consistency = case_when(chateauB_consistency == 'high' ~ 'High',
                                 chateauB_consistency == 'low' ~ 'Low'),
         frequency = case_when(chateauB_frequency == 'high' ~ 'High',
                               chateauB_frequency == 'low' ~ 'Low')) %>% 
  ggplot(aes(frequency, mse, fill = consistency)) +
  geom_bar(stat = 'identity', position = position_dodge(), color = 'black') +
  geom_errorbar(aes(ymin=mse-SEM, ymax=mse+SEM), position = position_dodge(.9), width = .1, color = 'grey23') +
  geom_point(data = d, position = position_jitterdodge(dodge.width = .9, jitter.width = .05, jitter.height = .0005), color = 'grey') +
  scale_fill_manual(values = COLORS) +
  labs(x = 'Frequency', y = 'Mean squared error', fill = 'Consistency') +
  theme_apa()
