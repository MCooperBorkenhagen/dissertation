s %>% 
  ggplot(aes(learntime)) +
  geom_histogram(binwidth = .1, color = 'black') +
  labs(x = 'Learntime (in mins)') +
  theme_classic()

tuning %>% 
  group_by(run_id) %>% 
  summarize(lt = first(learntime), hlsz = first(hidden_size), bs = first(batch_size)) %>% 
  filter(hlsz == 900 & bs == 250)


tuning %>% 
  filter(run_id == 19) %>% 
  ggplot(aes(epoch, acc_test)) +
  geom_smooth(n = 4)
