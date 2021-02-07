

options(scipen = 100)


aoa = readxl::read_xlsx('../inputs/raw/AoA_ratings_Kuperman_et_al_BRM.xlsx')
wcbc = read.csv('~/Documents/words/american_child_texts/data/clean/corpustable.csv')
wcbc_metadata = read_xlsx('~/Documents/words/american_child_texts/data/raw/book/metadata.xlsx') %>% 
  select(doc_id = isbn, age = earliest_advert_level_months)
