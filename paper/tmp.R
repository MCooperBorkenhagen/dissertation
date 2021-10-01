taraban_testmode %>% 
  filter(taraban == 'nonword') %>% nrow()
  filter(phonemes_proportion < 1) %>% 
  select(word, phon_read)  %>% 
  pull(word) -> pronounced_wrong


syllabics = read_csv('../inputs/taraban/syllabics.csv')

syllabics %>% 
  filter(word %in% pronounced_wrong) %>% 
  select(word, phon)

taraban_testmode %>% 
  filter(word == 'toad')

