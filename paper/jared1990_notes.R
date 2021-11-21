
## Jared & Seidenberg (1990)
#Consistency effects from @Jared1990 at first try weren't successfully replicated by the model. I moved to the Chateau data after hearing back from Debra

#@Jared1990 sought to extend the findings of @Glushko1979 to multisyllabic words, and their study predated @Chateau2003 using consistency categories over multisyllabic words which were more preliminary than those used in @Chateau2003 given that it was the first study of its kind for multisyllabic words. Their first experiment examined the effect of consistency on naming using a very similar paradigm to @Glushko1979 (and @Taraban1987). Prior to their study, the concept of "consistency" hadn't been applied to words greater than one syllable. They operationalized it as variability in the pronunciation of an orthographically identical syllable, with the syllable identified in the American Heritage Dictionary of the English Language. The minimal pairs provided in their description of the stimuli are `r scaps('rigor')` versus `r scaps('rigid')` (with the first syllable in each being consistent) and `r scaps('divine')` versus `r scaps('ravine')` (with the second syllable in each being inconsistent). The pairs were eligible for inclusion if they contained the same stress pattern. They varied the consistency of stimuli either in the first or the last syllable of the word. The target pattern from their study pertaining to the manipulation of consistency across first and last syllable identified words is in Figure XX.


N = multi_testmode %>%
  filter(!is.na(js1990_rt)) %>% 
  nrow()



multi_testmode %>%
  filter(!is.na(js1990_rt)) %>% 
  mutate(condition = case_when(js1990_condition == 'reg_inconsistent' ~ 'Inconsistent',
                               js1990_condition == 'exception' ~ 'Inconsistent',
                               js1990_condition == 'regular' ~ 'Consistent'),
         syllable = case_when(js1990_syll == 'first' ~ 'First',
                              js1990_syll == 'last' ~ 'Last')) %>% 
  group_by(condition, syllable) %>% 
  summarise(M = mean(js1990_rt),
            SD = sd(js1990_rt),
            SEM = SD/sqrt(N),
            condition = first(condition), 
            syllable = first(syllable)) %>% 
  ungroup() %>% 
  ggplot(aes(syllable, M, fill = condition)) +
  geom_bar(stat = 'summary', position = position_dodge(), color = 'black') +
  geom_errorbar(width = .2, color = 'grey25', aes(ymin = M-SEM, ymax = M+SEM), position = position_dodge(.9)) +
  scale_fill_manual(values = COLORS_jared) +
  labs(x = 'Syllabic position', y = 'RT') +
  theme_apa() +
  theme(legend.title = element_blank()) +
  coord_cartesian(ylim=c(500, 650))

