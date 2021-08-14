
# performance data
require(tidyverse)
source('utilities.R')

# get feedforward item data
ff = read_csv('../outputs/mono/feedforward/item-data.csv') %>% 
  filter(epoch != 320) %>% 
  mutate(stage = factor(case_when(epoch == 80 ~ 'Early',
                           epoch == 160 ~ 'Middle',
                           epoch == 240 ~ 'Late'))) %>% 
  select(-c(train_test, epoch)) %>% 
  mutate(model = 'Feedforward')

# relevel factors
ff$stage = ordered(ff$stage, c('Early', 'Middle', 'Late'))

# data generated from distance matrices (see chapter3_data_preprocess.R)
nearest_f1 = read_csv('data/monosyllabic-feedforward-wordwise-distances-1.csv')
nearest_f2 = read_csv('data/monosyllabic-feedforward-wordwise-distances-2.csv')




# get lstm items (their order is different than the order of feedforward matrices, when order isn't included in dataset)
items = read_csv('../outputs/mono/lstm/train-test-items.csv') %>% 
  select(-freq_scaled)

c1 = read_csv('../outputs/mono/lstm/item-data-monosyllabic-pre-1.csv') %>% 
  mutate(stage = 'Early') %>% 
  select(-c(cycle, phonlength)) %>% 
  full_join(items)

c3 = read_csv('../outputs/mono/lstm/item-data-monosyllabic-pre-3.csv') %>% 
  mutate(stage = 'Middle') %>% 
  select(-c(cycle, phonlength)) %>% 
  full_join(items)

c5 = read_csv('../outputs/mono/lstm/item-data-monosyllabic-advanced-2.csv') %>% 
  mutate(stage = 'Late') %>% 
  select(-c(cycle, phonlength)) %>% 
  full_join(items)

C = rbind(c1, c3, c5)

posttests = read_csv('../outputs/mono/lstm/posttest-trainwords.csv') %>% 
  full_join(read_csv('../outputs/mono/lstm/posttest-holdout-words.csv')) %>% 
  select(-freq) %>% 
  mutate(stage = factor('Late'))

C = C %>% 
  left_join(posttests, by = c('word', 'stage'))

# wordwise distances for lstm
# data generated from distance matrices (see chapter3_data_preprocess.R)
nearest_l1 = read_csv('data/monosyllabic-lstm-wordwise-distances-1.csv')
nearest_l2 = read_csv('data/monosyllabic-lstm-wordwise-distances-2.csv')


lstm = read_csv('../outputs/mono/lstm/train-test-items.csv') %>% 
  select(word, freq_scaled) %>% 
  left_join(C) %>% 
  mutate(model = 'LSTM')

rm(c1, c2, c3, c4, c5)



elp = read_csv('../inputs/raw/elp_5.27.16.csv') %>% 
  select(word = Word,
         elp_acc = I_NMG_Mean_Accuracy,
         elp_rt = I_Mean_RT) %>% 
  mutate(elp_rt = as.numeric(elp_rt))




nw = read.csv('data/monosyllabic-wordwise-distances.csv')

mono = mono %>% 
  left_join(nw)


mono = mono %>% 
  left_join(ff) %>% 
  left_join(read_csv('../inputs/3k/syllabics.csv')) %>% 
  left_join(consistency_)
  left_join(elp)

monosyllabic_k = read_csv('../models/monosyllabic-K.csv', col_names = F)[[1]]


# final cleans:
rm(elp, nw, consistency_, posttests)
