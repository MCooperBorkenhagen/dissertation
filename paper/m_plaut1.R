

taraban_crossval %>% 
  filter(!is.na(freq_plaut)) %>% 
  filter(epoch == 27) %>% 
  filter(plaut != 'nonword') %>%
  mutate(plaut = case_when(plaut == 'ambiguous' ~ 'Ambiguous',
                             plaut == 'exception' ~ 'Exception',
                             plaut == 'reg_consistent' ~ 'Regular consistent',
                             plaut == 'reg_inconsistent' ~ 'Regular inconsistent'),
         freq_plaut = case_when(freq_plaut == 'low' ~ -.5,
                                  freq_plaut =='high' ~ .5),
         plaut = fct_relevel(plaut, c('Exception', 'Ambiguous', 'Regular inconsistent', 'Regular consistent')),
         plaut_num = as.numeric(plaut)) %>%
  mutate(plaut_c = plaut_num-mean(c(1, 2, 3, 4))) %>% 
  lmer(loss ~ plaut_c*freq_plaut + (1|word) + (1|run_id), data = .) -> m_plaut1

save(m_plaut1, file = 'data/m_plaut1.Rda')

#load(file = 'data/m_plaut1.Rda')

#recompute confints:
m_plaut1_confint = confint(m_plaut1)
save(m_plaut1_confint, file = 'data/m_plaut1_confint.Rda')
#load('data/m_plaut1_confint.Rda')

m_plaut1_summary = summary(m_plaut1)
save(m_plaut1_summary, file = 'data/m_plaut1_summary.Rda')
#load(file = 'data/m_plaut1_summary.Rda')

# inspect
m_plaut1_confint

#runtime is about 25 minutes:
m_plaut1_anova = car::Anova(m_plaut1, type = 3, test = 'F')
# note that the anova table reflects the following usage:
# Analysis of Deviance Table (Type III Wald F tests with Kenward-Roger df)
save(m_plaut1_anova, 'data/m_plaut1_anova.Rda')
#load(file = 'data/m_plaut1_anova.Rda')
m_plaut1_anova