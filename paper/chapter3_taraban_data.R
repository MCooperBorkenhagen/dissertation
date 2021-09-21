



t18 = read_csv('../outputs/taraban/item-data-taraban-18.csv') %>% 
  mutate(epoch = '18') %>% 
  select(-cycle)
t36 = read_csv('../outputs/taraban/item-data-taraban-36.csv') %>% 
  mutate(epoch = '36') %>% 
  select(-cycle)
t54 = read_csv('../outputs/taraban/item-data-taraban-54.csv') %>% 
  mutate(epoch = '54') %>% 
  select(-cycle)
t72 = read_csv('../outputs/taraban/item-data-taraban-72.csv') %>% 
  mutate(epoch = '72') %>% 
  select(-cycle)


taraban = t18 %>% 
  full_join(t36) %>% 
  full_join(t54) %>% 
  full_join(t72)

syllabics = read_csv('../inputs/taraban/syllabics.csv')
