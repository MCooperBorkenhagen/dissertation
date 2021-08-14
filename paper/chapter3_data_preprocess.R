source('utilities.R')

# the location in which the words for the feedforward models are stored, and the order in which they are
# stored are different than for the lstm. Here are the feedforward model words:
train = read_csv('../models/data/mono-train.csv')$word
test = read_csv('../models/data/mono-test.csv')$word
ff_words = c(train, test)

# nearest phon data (all end of training)
# feedforward, early in training

nwf1 = nearest_word_df(as.matrix(read.csv('../outputs/mono/feedforward/posttest-outputs-targets-distance-matrix-1.csv', sep = ' ', header = F)), ff_words) %>% 
  mutate(stage = 'Early')

nwf2 = nearest_word_df(as.matrix(read.csv('../outputs/mono/feedforward/posttest-outputs-targets-distance-matrix-3.csv', sep = ' ', header = F)), ff_words) %>% 
  mutate(stage = 'Late')

# save feedforward nearest word data
# write to file so it can be read in rather than executed:
write.csv(nwf1, 'data/monosyllabic-feedforward-wordwise-distances-1.csv', row.names=F)
write.csv(nwf2, 'data/monosyllabic-feedforward-wordwise-distances-2.csv', row.names=F)



lstm_words = read_csv('../outputs/mono/lstm/train-test-items.csv')$word

nwl1 = nearest_word_df(as.matrix(read.csv('../outputs/mono/lstm/posttest-outputs-targets-distance-matrix-premodel.csv', sep = ' ', header = F)), lstm_words) %>% 
  mutate(stage = 'Early')

nwl2 = nearest_word_df(as.matrix(read.csv('../outputs/mono/lstm/posttest-outputs-targets-distance-matrix-advanced.csv', sep = ' ', header = F)), lstm_words) %>% 
  mutate(stage = 'Late')


write.csv(nwl1, 'data/monosyllabic-lstm-wordwise-distances-1.csv', row.names=F)
write.csv(nwl2, 'data/monosyllabic-lstm-wordwise-distances-2.csv', row.names=F)

