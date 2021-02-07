


 aoa = aoa %>% 
  select(word = Word, aoa = Rating.Mean) %>% 
  mutate(aoa = round(as.numeric(aoa), digits = 2),
         word = str_replace(word, '[[:punct:]]', '')) %>% 
  drop_na(aoa) %>% 
  arrange(-desc(aoa)) %>% 
  mutate(rank = seq_len(n()))

 
 
# wcbc
wcbc = wcbc %>% 
  mutate(text = as.character(text)) %>% 
  unnest_tokens(word, text) %>% 
  left_join(wcbc_metadata) %>% 
  mutate(word = tolower(word)) %>%
  mutate(word = str_replace(word, '[[:punct:]]', '')) %>% 
  filter(age <= 60) %>% 
  group_by(word) %>% 
  summarise(freq = n()) %>% 
  arrange(desc(freq)) %>% 
  mutate(rank = seq_len(n()))




rm(wcbc_metadata)
