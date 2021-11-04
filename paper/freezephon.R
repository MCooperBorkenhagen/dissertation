

# all frozen phon weights:
cols_ = c(names(read_csv('../outputs/freezephon-all/0/item-data--9.csv')), 'epoch', 'dir')
df = data.frame(matrix(nrow = 0, ncol = length(cols_)))
colnames(df) = cols_


PATH = '../outputs/freezephon-all/'
FNAME = 'item-data-freezephon-all-'
DIRS = c('0', '18', '36', '54', 'never')

EPOCHS = c(9, 18, 27, 36, 45, 54, 63)

conditions = data.frame(DIRS, c('always', 'early', 'middle', 'late', 'never')) %>% 
  data_frame()
colnames(conditions) = c('dir', 'when_freeze')


for (dir_ in DIRS){
  
  for (epoch in EPOCHS){
    
    tmp = read_csv(paste(PATH, dir_, '/', FNAME, epoch, '.csv', sep = '')) %>% 
      mutate(epoch = epoch, dir = dir_)
    
    df = rbind(df, tmp)    
    
  }
  
}

freezephon_all = df %>% 
  left_join(conditions) %>% 
  select(-c(cycle, dir)) %>% 
  mutate(which_freeze = 'all')

# frozen LSTM only
cols_ = c(names(read_csv('../outputs/freezephon-lstm/0/item-data-freezephon-lstm-9.csv')), 'epoch', 'dir')
df = data.frame(matrix(nrow = 0, ncol = length(cols_)))
colnames(df) = cols_


FNAME = 'item-data-freezephon-lstm-'
PATH = '../outputs/freezephon-lstm/'


for (dir_ in DIRS){
  
  for (epoch in EPOCHS){
    
    tmp = read_csv(paste(PATH, dir_, '/', FNAME, epoch, '.csv', sep = '')) %>% 
      mutate(epoch = epoch, dir = dir_)
    
    df = rbind(df, tmp)    
    
  }
  
}

freezephon_lstm = df %>% 
  left_join(conditions) %>% 
  select(-c(cycle, dir)) %>% 
  mutate(which_freeze = 'lstm')


freezephon = rbind(freezephon_all, freezephon_lstm)

rm(df, conditions, PATH, FNAME, DIRS, dir_, epoch, freezephon_all, freezephon_lstm)
