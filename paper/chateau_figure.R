
# This script compiles the visuals for the two-by-three panel figure on the Chateau & Jared (2003) experiments
COLORS = c('High' = 'grey86', 'Low' = 'black')



## Experiment 3 (appendix B) ----
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
  geom_errorbar(aes(ymin=mse-SEM, ymax=mse+SEM), position = position_dodge(.9), width = .2, color = 'grey38') +
  geom_point(data = d, position = position_jitterdodge(dodge.width = .9, jitter.width = .05, jitter.height = .0005), color = 'grey') +
  scale_fill_manual(values = COLORS) +
  labs(x = 'Frequency', y = 'Mean squared error', fill = 'Consistency', title = 'Experiment 3') +
  theme_apa() +
  theme(plot.title = element_text(hjust = .5, size = 34),
        axis.title = element_text(size = 20),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 15),
        axis.text = element_text(size = 16))




# human data ----
d = cj2003_means %>% 
  filter(!is.na(chateauB_consistency)) %>% 
  mutate(consistency = case_when(chateauB_consistency == 'high' ~ 'High',
                                 chateauB_consistency == 'low' ~ 'Low'),
         frequency = case_when(chateauB_frequency == 'high' ~ 'High',
                               chateauB_frequency == 'low' ~ 'Low'))

cj2003_means %>% 
  filter(!is.na(chateauB_consistency)) %>% 
  group_by(chateauB_consistency, chateauB_frequency) %>% 
  summarise(SD = sd(rt),
            rt = mean(rt),
            SEM = SD/sqrt(N)) %>% 
  mutate(consistency = case_when(chateauB_consistency == 'high' ~ 'High',
                                 chateauB_consistency == 'low' ~ 'Low'),
         frequency = case_when(chateauB_frequency == 'high' ~ 'High',
                               chateauB_frequency == 'low' ~ 'Low')) %>% 
  ggplot(aes(frequency, rt, fill = consistency)) +
  geom_bar(stat = 'identity', position = position_dodge(), color = 'black') +
  geom_errorbar(aes(ymin=rt-SEM, ymax=rt+SEM), position = position_dodge(.9), width = .2, color = 'grey38') +
  geom_point(data = d, position = position_jitterdodge(dodge.width = .9, jitter.width = .05, jitter.height = .0005), color = 'grey') +
  scale_fill_manual(values = COLORS) +
  labs(x = 'Frequency', y = 'RT (msec)', fill = 'Consistency') +
  coord_cartesian(ylim = c(550, 700)) +
  theme_apa() +
  theme(axis.title = element_text(size = 20),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 15),
        axis.text = element_text(size = 16))


# Experiment 4 (appendix C) ----
# model data ----
N = multi %>% filter(epoch == 45) %>% filter(!is.na(chateauC_consistency)) %>% nrow()


d = multi %>% 
  filter(epoch == 45) %>% 
  filter(!is.na(chateauC_consistency)) %>% 
  mutate(consistency = case_when(chateauC_consistency == 'high' ~ 'High',
                                 chateauC_consistency == 'low' ~ 'Low'))


multi %>% 
  filter(epoch == 45) %>% 
  filter(!is.na(chateauC_consistency)) %>% 
  group_by(chateauC_consistency) %>% 
  summarise(SD = sd(mse),
            mse = mean(mse),
            SEM = SD/sqrt(N)) %>% 
  mutate(consistency = case_when(chateauC_consistency == 'high' ~ 'High',
                                 chateauC_consistency == 'low' ~ 'Low')) %>% 
  ggplot(aes(consistency, mse, fill = consistency)) +
  geom_bar(stat = 'identity', color = 'black') +
  geom_errorbar(aes(ymin=mse-SEM, ymax=mse+SEM), position = position_dodge(.9), width = .2, color = 'grey38') +
  geom_point(data = d, color = 'grey', position = position_jitter(.02)) +
  scale_fill_manual(values = COLORS) +
  labs(x = 'Consistency', y = 'Mean squared error', fill = 'Consistency', title = 'Experiment 4') +
  theme_apa() +
  theme(plot.title = element_text(hjust = .5, size = 34),
        axis.title = element_text(size = 20),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 15),
        axis.text = element_text(size = 16))

# human data ----
d = cj2003_means %>% 
  filter(!is.na(chateauC_consistency)) %>% 
  mutate(consistency = case_when(chateauC_consistency == 'high' ~ 'High',
                                 chateauC_consistency == 'low' ~ 'Low'))


cj2003_means %>% 
  filter(!is.na(chateauC_consistency)) %>% 
  group_by(chateauC_consistency) %>% 
  summarise(SD = sd(rt),
            rt = mean(rt),
            SEM = SD/sqrt(N)) %>% 
  mutate(consistency = case_when(chateauC_consistency == 'high' ~ 'High',
                                 chateauC_consistency == 'low' ~ 'Low')) %>% 
  ggplot(aes(consistency, rt, fill = consistency)) +
  geom_bar(stat = 'identity', color = 'black') +
  geom_errorbar(aes(ymin=rt-SEM, ymax=rt+SEM), position = position_dodge(.9), width = .2, color = 'grey38') +
  geom_point(data = d, color = 'grey', position = position_jitter(.02)) +
  scale_fill_manual(values = COLORS) +
  labs(x = 'Consistency', y = 'RT (msec)', fill = 'Consistency') +
  coord_cartesian(ylim = c(550, 700)) +
  theme_apa() +
  theme(axis.title = element_text(size = 20),
        legend.title = element_text(size = 16),
        legend.text = element_text(size = 15),
        axis.text = element_text(size = 16))


