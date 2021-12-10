taraban_crossval %>% 
  filter(!is.na(freq_plaut)) %>% 
  filter(epoch == 27) %>% 
  filter(plaut != 'nonword') %>%
  mutate(plaut_c = case_when(plaut == 'exception' ~ -.5,
                               plaut == 'reg_inconsistent' ~ .5,
                             plaut == 'reg_consistent' ~ .5),
         freq_plaut = case_when(freq_plaut == 'low' ~ -.5,
                                  freq_plaut =='high' ~ .5)) %>%
  lmer(loss ~ plaut_c*freq_plaut + (1|word) + (1|run_id), data = .) -> m_plaut2

summary(m_plaut2)

#runtime is about 25 minutes:
m_plaut2_confint = confint(m_plaut2)
m_plaut2_anova = car::Anova(m_plaut2, type = 3, test = 'F')
save(m_plaut2_confint, file = 'data/m_plaut2_confint.Rda')
save(m_plaut2_anova, file = 'data/m_plaut2_anova.Rda')

#load(file = 'data/m_plaut2_anova.Rda')