
# get data from learnability study
load(file = '~/Box Sync/box_mcb/research/projects/learnability/analysis/1_items/data/clean/data.rda')

lbty = d
rm(d)

load('~/Box Sync/box_mcb/research/projects/american_child_texts/data/clean/tidycorpus.rda')
d = tidycorpus

aoa = readxl::read_xlsx('~/Box Sync/box_mcb/research/words/aoa/AoA_ratings_Kuperman_et_al_BRM.xlsx') %>% 
  select(lemma = Word,
         aoa = Rating.Mean)

d = d %>% 
  left_join(aoa, by = 'lemma')


rm(tidycorpus)

