
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
nearest_f = rbind(nearest_f1, nearest_f2)


ff = ff %>% 
  left_join(nearest_f)



# get lstm items (their order is different than the order of feedforward matrices, when order isn't included in dataset)
items = read_csv('../outputs/mono/lstm/train-test-items.csv') %>% 
  select(-freq_scaled) %>% 
  rename(train_test = `train-test`)

# c is for cycle
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

# all cycle data
C = rbind(c1, c3, c5) %>% 
  select(-`train_test`)

# these are the post test data unique to the lstm implementation (because they are generated in offline test mode)
posttests = read_csv('../outputs/mono/lstm/posttest-trainwords.csv') %>% 
  full_join(read_csv('../outputs/mono/lstm/posttest-holdout-words.csv')) %>% 
  mutate(stage = factor('Late'))

C = C %>% 
  left_join(posttests, by = c('word', 'stage')) 

# wordwise distances for lstm
# data generated from distance matrices (see chapter3_data_preprocess.R)
nearest_l1 = read_csv('data/monosyllabic-lstm-wordwise-distances-1.csv')
nearest_l2 = read_csv('data/monosyllabic-lstm-wordwise-distances-2.csv')
nearest_l = rbind(nearest_l1, nearest_l2)

lstm = C %>% 
  mutate(model = 'LSTM') %>% 
  left_join(nearest_l)

rm(c1, c3, c5)

frequency = posttests %>% 
  select(word, freq) %>% 
  left_join(read_csv('../outputs/mono/lstm/train-test-items.csv')) %>% 
  select(word, freq, freq_scaled)

elp = read_csv('../inputs/raw/elp_5.27.16.csv') %>% 
  mutate(word = tolower(Word)) %>% 
  filter(word %in% frequency$word) %>% 
  select(word, elp_acc = I_NMG_Mean_Accuracy,
         elp_rt = I_Mean_RT) %>% 
  mutate(elp_rt = as.numeric(elp_rt))

syllabics = read_csv('../inputs/3k/syllabics.csv') %>% 
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

mono = rbind(lstm, ff) %>% 
  left_join(syllabics) %>% 
  left_join(elp) %>% 
  left_join(frequency)


# final cleans:
rm(C, c1, c3, c5, elp, ff, frequency, items, lstm, nearest_f, nearest_f1, nearest_f2, nearest_l, nearest_l1, nearest_l2, posttests, syllabics)
