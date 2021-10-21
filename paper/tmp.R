

multi %>% 
  group_by(epoch, train_test) %>% 
  summarise(mse = mean(binary_acc)) %>% 
  ggplot(aes(factor(epoch), mse, color = train_test, group = train_test)) +
  geom_point() +
  geom_line()



multi %>% 
  #filter(epoch == EPOCH) %>% 
  filter(!is.na(elp_rt)) %>% 
  ggplot(aes(phonlen, mse)) +
  geom_bar(stat='summary') + 
  facet_grid(~epoch)
  

