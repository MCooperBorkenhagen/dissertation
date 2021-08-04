
# performance data
require(tidyverse)
source('utilities.R')

c1 = read_csv('../models/item-data-monosyllabic-pre-1.csv') %>% 
  select(-c(cycle)) %>% 
  rename(acc_c1 = accuracy, loss_c1 = loss)

c2 = read_csv('../models/item-data-monosyllabic-pre-2.csv') %>% 
  select(-c(phonlength, cycle)) %>% 
  rename(acc_c2 = accuracy, loss_c2 = loss)

c3 = read_csv('../models/item-data-monosyllabic-pre-3.csv') %>% 
  select(-c(phonlength, cycle)) %>% 
  rename(acc_c3 = accuracy, loss_c3 = loss)

c4 = read_csv('../models/item-data-monosyllabic-advanced-1.csv') %>% 
  select(-c(phonlength, cycle)) %>% 
  rename(acc_c4 = accuracy, loss_c4 = loss)


c5 = read_csv('../models/item-data-monosyllabic-advanced-2.csv') %>% 
  select(-c(phonlength, cycle)) %>% 
  rename(acc_c5 = accuracy, loss_c5 = loss)



posttests = read_csv('../models/posttest-trainwords.csv') %>% 
  full_join(read_csv('../models/posttest-holdout-words.csv'))

consistency_ = read_csv('../models/train-test-items.csv') %>% 
  filter(`train-test` == 'train') %>% 
  left_join(read_csv('../inputs/syllabics.csv')) %>% 
  group_by(body) %>% 
  mutate(community = n()) %>% 
  ungroup() %>% 
  group_by(body, rime) %>% 
  mutate(friends = n()) %>% 
  ungroup() %>% 
  mutate(consistency = friends/community) %>% 
  select(word, consistency)

mono = read_csv('../models/train-test-items.csv') %>% 
  rename(train_test = `train-test`) %>% 
  left_join(posttests)  %>% 
  left_join(read_csv('../inputs/syllabics.csv')) %>% 
  left_join(consistency_) %>% 
  left_join(c1) %>% 
  left_join(c2) %>% 
  left_join(c3) %>% 
  left_join(c4) %>% 
  left_join(c5)

rm(c1, c2, c3, c4, c5)

nw = read.csv('data/monosyllabic-wordwise-distances.csv')

mono = mono %>% 
  left_join(nw)


elp = read_csv('../inputs/raw/elp_5.27.16.csv') %>% 
  select(word = Word,
         elp_acc = I_NMG_Mean_Accuracy,
         elp_rt = I_Mean_RT) %>% 
  mutate(elp_rt = as.numeric(elp_rt))

mono = mono %>% 
  left_join(elp)

monosyllabic_k = read_csv('../models/monosyllabic-K.csv', col_names = F)[[1]]


# final cleans:
rm(elp, nw, consistency_, posttests)
