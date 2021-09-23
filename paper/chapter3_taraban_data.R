




t9 = read_csv('../outputs/taraban/item-data-taraban-9.csv') %>% 
  mutate(epoch = '9') %>% 
  select(-cycle)

t18 = read_csv('../outputs/taraban/item-data-taraban-18.csv') %>% 
  mutate(epoch = '18') %>% 
  select(-cycle)

t27 = read_csv('../outputs/taraban/item-data-taraban-27.csv') %>% 
  mutate(epoch = '27') %>% 
  select(-cycle)

t36 = read_csv('../outputs/taraban/item-data-taraban-36.csv') %>% 
  mutate(epoch = '36') %>% 
  select(-cycle)

t45 = read_csv('../outputs/taraban/item-data-taraban-45.csv') %>% 
  mutate(epoch = '45') %>% 
  select(-cycle)

t54 = read_csv('../outputs/taraban/item-data-taraban-54.csv') %>% 
  mutate(epoch = '54') %>% 
  select(-cycle)

t63 = read_csv('../outputs/taraban/item-data-taraban-63.csv') %>% 
  mutate(epoch = '63') %>% 
  select(-cycle)

t72 = read_csv('../outputs/taraban/item-data-taraban-72.csv') %>% 
  mutate(epoch = '72') %>% 
  select(-cycle)




syllabics = read_csv('../inputs/taraban/syllabics.csv') %>% 
  group_by(body) %>% 
  mutate(body_neighbors = n()) %>% 
  ungroup() %>% 
  group_by(rime) %>% 
  mutate(rime_neighbors = n()) %>% 
  ungroup() %>% 
  group_by(nucleus) %>% 
  mutate(nucleus_neighbors = n()) %>% 
  ungroup() %>% 
  group_by(core) %>% 
  mutate(core_neighbors = n()) %>% 
  ungroup() %>% 
  group_by(body, rime) %>% 
  mutate(body_rime = n()) %>% 
  ungroup() %>% 
  mutate(consistency = body_rime/body_neighbors)

# experimental words

taraban_conditions = read_csv('../inputs/raw/taraban_etal_1987_words.csv') %>% 
  rename(taraban = condition,
         freq_taraban = frequency) %>% 
  filter(word %in% syllabics$word)

jaredA_conditions = read_csv('../inputs/raw/jared_1997_appendixA.csv') %>% 
  rename(jaredA = condition,
         freq_jaredA = frequency) %>% 
  filter(word %in% syllabics$word)

#jaredC_conditions = read_csv('../inputs/raw/jared_1997_appendixC.csv')



freq_train = read_csv('../models/data/taraban-train.csv') %>% 
  mutate(train_test = 'train')



freq_test = read_csv('../models/data/taraban-test.csv', col_names = 'word') %>% 
  mutate(freq = NA,
         freq_scaled = NA,
         train_test = 'test')


frequency = rbind(freq_train, freq_test)

tarabanK = read_csv('../models/data/taraban-K.txt', col_names = F)[[1]]

taraban = t9 %>% 
  full_join(t18) %>% 
  full_join(t27) %>% 
  full_join(t36) %>% 
  full_join(t45) %>% 
  full_join(t54) %>% 
  full_join(t63) %>% 
  full_join(t72) %>% 
  left_join(syllabics) %>% 
  left_join(taraban_conditions) %>% 
  left_join(jaredA_conditions) %>% 
  left_join(frequency)  


taraban_testmode = read_csv('../outputs/taraban/taraban-generalization-epoch72.csv')




rm(t18, t36, t54, t72, frequency, syllabics, taraban_conditions, jaredA_conditions)
