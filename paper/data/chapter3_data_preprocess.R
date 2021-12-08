source('../utilities.R')

# the location in which the words for the feedforward models are stored, 
# and the order in which they are stored are different than for the lstm,
# hence the differences in how the ff and lstm words are aggregated
# Here are the feedforward model words.

train = read_csv('../../models/data/mono-train.csv')$word
test = read_csv('../../models/data/mono-test.csv')$word
ff_words = c(train, test)

# nearest phon data
nwf = nearest_word_df(as.matrix(read.csv('../../outputs/mono/feedforward/posttest-outputs-targets-distance-matrix-4.csv', sep = ' ', header = F)), ff_words) %>% 
  mutate(stage = 'Late')
# write to file so it can be read in rather than executed:
write.csv(nwf, 'monosyllabic-feedforward-wordwise-distances-late.csv', row.names=F)

# now the LSTM words at end of training:
lstm_words = read_csv('../../outputs/mono/train-test-items.csv')$word

nwl = nearest_word_df(as.matrix(read.csv('../../outputs/mono/lstm/posttest-outputs-targets-distance-matrix-late.csv', sep = ' ', header = F)), lstm_words) %>% 
  mutate(stage = 'Late')


write.csv(nwl, 'monosyllabic-lstm-wordwise-distances-late.csv', row.names=F)


