
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

d = read_csv('../models/train-test-items.csv') %>% 
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


# nearest phon data (all end of training)
rn = read_csv('../models/train-test-items.csv')$word
dm = as.matrix(read.csv('../models/posttest-outputs-targets-distance-matrix.csv', sep = ' ', header = F))
rownames(dm) = rn
colnames(dm) = rn



nw = data.frame(matrix(nrow = length(rn), ncol = 3))
colnames(nw) = c('word', 'nearest_phon_rank', 'nearest_phon')

row = 1
for (word in rn){
  rank = nearest_word(dm, word, return_rank = T)
  nearest = nearest_word(dm, word, return_rank = F)
  
  nw$word[row] = word
  nw$nearest_phon_rank[row] = rank
  nw$nearest_phon[row] = nearest
  
  row = row + 1
}

d = d %>% 
  left_join(nw)


elp = read_csv('../inputs/raw/elp_5.27.16.csv') %>% 
  select(word = Word,
         elp_acc = I_NMG_Mean_Accuracy,
         elp_rt = I_Mean_RT) %>% 
  mutate(elp_rt = as.numeric(elp_rt))

d = d %>% 
  left_join(elp)


# final cleans:
rm(elp, nw, consistency_, dm, posttests)
