source('utilities.R')


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

# write to file so it can be read in rather than executed:
write.csv(nw, 'data/monosyllabic-wordwise-distances.csv', row.names=F)

