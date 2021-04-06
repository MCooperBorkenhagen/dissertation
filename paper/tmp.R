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
  ggplot(aes(epoch, acc_test, colour = factor(hidden_size))) +
  geom_smooth(method = 'loess') +
  facet_grid(~batch_size) +
  labs(title = 'Accuracy on test set throughout training', subtitle = '90/10 split for validation',
       x = 'Epoch', y = 'Accuracy', colour = '# hidden units') +
  theme(plot.title = element_text(hjust = .5, size = 22),
        plot.subtitle = element_text(hjust = .5, size = 18))
