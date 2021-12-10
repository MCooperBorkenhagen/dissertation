#\includepdf[pages=-]{5_appendix1.pdf}


V = 'ZH'

multi_testmode %>% 
  filter(str_detect(phon, V)) %>% 
  select(word, phon)  

multi_testmode %>% 
  filter(str_detect(phon, V)) %>% 
  select(word, phon)  %>% 
  View()
