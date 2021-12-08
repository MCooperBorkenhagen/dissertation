

cols_ = c(names(read_csv('../outputs/nullphon/0/item-data-null-input-9.csv')), 'epoch', 'dir', 'null_test')
df = data.frame(matrix(nrow = 0, ncol = length(cols_)))
colnames(df) = cols_


PATH = '../outputs/nullphon/'
FNAME1 = 'item-data-nullphon-'
FNAME2 = 'item-data-null-input-'
DIRS = c('0', '18', '36', '54', 'never')


conditions = data.frame(DIRS, c('always', 'early', 'middle', 'late', 'never')) %>% 
  data_frame()
colnames(conditions) = c('dir', 'when_null')


for (dir_ in DIRS){
  
  for (epoch in c(9, 18, 27, 36, 45, 54, 63)){
    
    tmp1 = read_csv(paste(PATH, dir_, '/', FNAME1, epoch, '.csv', sep = '')) %>% 
      mutate(epoch = epoch, dir = dir_, null_test = FALSE)
    
    
    tmp2 = read_csv(paste(PATH, dir_, '/', FNAME2, epoch, '.csv', sep = '')) %>% 
      mutate(epoch = epoch, dir = dir_, null_test = TRUE)
    
    df = rbind(df, tmp1)
    df = rbind(df, tmp2)
    
  }
  
}


nullphon = df %>% 
  left_join(conditions) %>% 
  select(-c(cycle, dir))



rm(df, conditions, PATH, FNAME1, FNAME2, DIRS, dir_, epoch)
