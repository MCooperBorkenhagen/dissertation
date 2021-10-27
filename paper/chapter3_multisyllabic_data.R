
  

cols_ = c(names(read_csv('../outputs/js1990/item-data-js1990-9.csv')), 'epoch')
trd = data.frame(matrix(nrow = 0, ncol = length(cols_)))
colnames(trd) = cols_


PATH = '../outputs/js1990/'
FNAME = 'item-data-js1990-'

for (epoch in c(9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99)){
  tmp = read_csv(paste(PATH, FNAME, epoch, '.csv', sep = '')) %>% 
    mutate(epoch = epoch)
  trd = rbind(trd, tmp)}

# frequency

trainwords = read_csv('../models/data/js1990/train.csv')
testwords = read_csv('../models/data/js1990/test.csv', col_names = F) %>% 
  rename(word = X1) %>% 
  mutate(freq = NA,
         freq_scaled = NA)

frequency = rbind(trainwords, testwords)

# data to merge:
syllabics = read_csv('../inputs/js1990/syllabics.csv') %>% 
  mutate(phonlen = phonlen(phon),
         orthlen = str_length(word))

# Jared et al. (1990)
js1990 = read_csv('../inputs/raw/jared_etal_1990_e1.csv') %>% 
  select(word, js1990_coding = original, js1990_syll = syllable, js1990_condition = condition, js1990_rt = rt, js1990_group = group_id)

# reconcile with chateau experiments
ca = read_csv('../inputs/raw/chateau_etal_2003_a.csv') %>% 
  rename(chateauA = condition,
         chateauA_consistency = consistency)

cb = read_csv('../inputs/raw/chateau_etal_2003_b.csv') %>% 
  rename(chateauB_consistency = consistency,
         chateauB_frequency = frequency)

cc = read_csv('../inputs/raw/chateau_etal_2003_c.csv') %>% 
  rename(chateauC_consistency = consistency)

# elp data
elp = read_csv('../inputs/raw/elp_5.27.16.csv') %>% 
  mutate(word = tolower(Word)) %>% 
  filter(word %in% syllabics$word) %>% 
  select(word, elp_acc = I_NMG_Mean_Accuracy,
         elp_rt = I_NMG_Mean_RT, orth_n = Ortho_N, phon_n = Phono_N, orth_lev = OLD, phon_lev = PLD) %>% 
  mutate(elp_rt = as.numeric(elp_rt), orth_lev = as.numeric(orth_lev), phon_lev = as.numeric(phon_lev))




multi_testmode = read_csv('../outputs/js1990/generalization-epoch99.csv') %>% 
  select(-c(train_test, freq)) %>% 
  mutate(epoch = as.factor(epoch)) %>% 
  left_join(syllabics) %>% 
  left_join(js1990) %>% 
  left_join(frequency) %>% 
  left_join(elp) %>% 
  left_join(ca) %>% 
  left_join(cb) %>% 
  left_join(cc)


multi = trd %>% 
  select(-c(cycle, phonlength)) %>% 
  mutate(epoch = as.factor(epoch)) %>% 
  left_join(syllabics) %>% 
  left_join(js1990) %>% 
  left_join(frequency) %>% 
  left_join(elp) %>% 
  left_join(ca) %>% 
  left_join(cb) %>% 
  left_join(cc)


js1990_means = read_csv('data/jared_etal_1990_e1_means.csv')

cj2003_means = read_csv('data/chateau_etal_2003_means.csv') %>% 
  rename(word = Word, rt = Latency, errors_tot = `# errors`, errors_perc = `% error`) %>% 
  left_join(ca) %>% 
  left_join(cb) %>% 
  left_join(cc)
  


rm(syllabics, ca, cb, cc, PATH, FNAME, trd, tmp, cols_, trainwords, testwords, frequency)


