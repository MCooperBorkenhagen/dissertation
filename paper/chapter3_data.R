
# performance data
require(tidyverse)
source('utilities.R')

# get feedforward item data
ff = read_csv('../outputs/mono/feedforward/item-data.csv') %>% 
#  filter(epoch != 320) %>% 
  filter(epoch != 240) %>% 
  mutate(stage = factor(case_when(epoch == 80 ~ 'Early',
                           epoch == 160 ~ 'Middle',
                           epoch == 320 ~ 'Late'))) %>% 
  select(-epoch) %>% 
  mutate(model = 'Feedforward')


# relevel factors
ff$stage = ordered(ff$stage, c('Early', 'Middle', 'Late'))




# data generated from distance matrices (see chapter3_data_preprocess.R)
nearest_f = read_csv('data/monosyllabic-feedforward-wordwise-distances-late.csv')



ff = ff %>% 
  left_join(nearest_f)


# get lstm items (their order is different than the order of feedforward matrices, when order isn't included in dataset)
items = read_csv('../outputs/mono/train-test-items.csv') %>% 
  select(-freq_scaled) %>% 
  rename(train_test = `train-test`)

# c is for cycle
c1 = read_csv('../outputs/mono/lstm/item-data-monosyllabic-early-1.csv') %>% 
  mutate(stage = 'Early') %>% 
  select(-c(cycle, phonlength)) %>% 
  full_join(items)

c3 = read_csv('../outputs/mono/lstm/item-data-monosyllabic-early-3.csv') %>% 
  mutate(stage = 'Middle') %>% 
  select(-c(cycle, phonlength)) %>% 
  full_join(items)

c5 = read_csv('../outputs/mono/lstm/item-data-monosyllabic-late-2.csv') %>% 
  mutate(stage = 'Late') %>% 
  select(-c(cycle, phonlength)) %>% 
  full_join(items)

# all cycle data
C = rbind(c1, c3, c5)

# these are the post test data unique to the lstm implementation (because they are generated in offline test mode)
mono_lstm_post = read_csv('../outputs/mono/lstm/posttest-trainwords.csv') %>% # these are the production trials for train words
  full_join(read_csv('../outputs/mono/lstm/posttest-holdout-words.csv')) %>% # these are the production trials for test words
  mutate(stage = factor('Late')) %>% 
  select(-freq)


# wordwise distances for lstm
# data generated from distance matrices (see chapter3_data_preprocess.R)
nearest_l = read_csv('data/monosyllabic-lstm-wordwise-distances-late.csv')

lstm = C %>% 
  mutate(model = 'LSTM') %>% 
  left_join(nearest_l)

rm(c1, c3, c5)


# word ("lexical") data
frequency = read_csv('../outputs/mono/lstm/posttest-trainwords.csv') %>% 
  full_join(read_csv('../outputs/mono/lstm/posttest-holdout-words.csv')) %>% 
  select(word, freq) %>% 
  left_join(read_csv('../outputs/mono/train-test-items.csv')) %>% 
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

mono_lstm_testmode = mono_lstm_post %>% 
  left_join(syllabics) %>% 
  left_join(frequency) %>% 
  left_join(elp) %>% 
  left_join(items)

mono = rbind(lstm, ff) %>% 
  left_join(syllabics) %>% 
  left_join(elp) %>% 
  left_join(frequency) %>% 
  left_join(items)

# reorder stage factor
mono$stage = ordered(mono$stage, c('Early', 'Middle', 'Late'))

monosyllabic_k = read_csv('../outputs/mono/monosyllabic_k.csv', col_names = F)[[1]]

# final cleans:
rm(C, elp, ff, frequency, items, lstm, nearest_f, nearest_l, syllabics, mono_lstm_post)
