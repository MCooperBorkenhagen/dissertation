
N = length(unique(nullphon$word))

nullphon %>% 
  mutate(when_null = as.factor(when_null)) %>% 
  group_by(when_null, null_test, epoch) %>% 
  summarise(accuracy_m = mean(binary_acc),
            accuracy_sd = sd(binary_acc),
            accuracy_sem = accuracy_sd/sqrt(N),
            error_m = mean(mse),
            error_sd = sd(mse),
            error_sem = error_sd/sqrt(N)) %>% 
  mutate(condition = fct_relevel(when_null, c('early', 'middle', 'late', 'never'))) %>% 
  mutate(condition = case_when(condition == 'early' ~ 'Epoch 18',
                               condition == 'middle' ~ 'Epoch 36',
                               condition == 'late' ~ 'Epoch 54',
                               condition == 'never' ~ 'Never')) %>% 
  ggplot(aes(factor(epoch), accuracy_m, color = condition, group = condition)) +
  geom_point() +
  geom_errorbar(aes(ymin = accuracy_m - accuracy_sem, ymax = accuracy_m + accuracy_sem), width = .1) +
  geom_line() +
  facet_grid(~null_test) +
  geom_vline(xintercept = 18)  %>% 
  labs(x = 'Epoch', y = 'Accuracy (proportion over items)', color = 'Onset of transition') +
  theme_apa()



max(str_length(unique(multi$word)))

