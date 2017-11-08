rm(list = ls())

library(dplyr)
library(ggplot2)
library(stringr)
library(readr)
library(reshape2)


df_raw <- readr::read_delim("c:/users/peterd/Desktop/GitHub/PriceParser/2017-11-06_Price-Comparison.csv", delim=';')


df <- df_raw %>%
  select(ArtikelId,Category_Level_1, Category_Level_2, Farbe, ObjectRate,
         Lieferantenname, Artikelserie,Sales_LTM,
         grep("Preis.+", names(df_raw))) %>%
  select(-Preis_EAN)


df_boxplot <- df_raw %>%
  select(ArtikelId,Category_Level_1, Category_Level_2, Farbe, ObjectRate,
         Lieferantenname, Artikelserie,Sales_LTM,
         grep("Preis.*", names(df_raw))) %>%
  select(-Preis_EAN,-Preis_Pos) %>%
  reshape2::melt(id.vars=1:8, variable.name="Company", value.name="Preis") %>%
  mutate(Company = Company %>% as.character,
         Company = ifelse(Company=="Preis", "CRH", str_sub(Company,7,-1)),
         Company = str_replace(Company, "Team", "")) %>%
  mutate(Color = ifelse(Company=="CRH", TRUE, FALSE))


median <- df_boxplot %>%
  filter(Preis<1000) %>%
  filter(Sales_LTM>0) %>%
  filter(Company == 'CRH') %>%
  filter(Category_Level_1 != "Ersatzteile") %>%
  tidyr::drop_na() %>%
  group_by(Category_Level_1) %>%
  summarise(Median = median(Preis))



df_boxplot_reshape <- df_boxplot %>%
  select(-Color) %>%
  reshape2::dcast(ArtikelId ~ Company, value.var="Preis", fun.aggregate = max) %>%
  mutate(Sanitas = ifelse(CRH > Sanitas, 'Ja', "Nein"),
         Hug = ifelse(CRH > Hug, 'Ja', "Nein"),
         Saneo = ifelse(CRH > Samep, 'Ja', "Nein"),
         SaniDusch = ifelse(CRH > SaniDusch, 'Ja', "Nein"),
         Sabag = ifelse(CRH > Sabag, 'Ja', "Nein"),
         CRH = "") %>%
  reshape2::melt(id.vars=1, variable.name = "Company", value.name="PriceCheck")


df_boxplot <- df_boxplot %>%
  left_join(df_boxplot_reshape, by=c('ArtikelId', 'Company'))


df_boxplot %>%
  filter(Preis<1000) %>%
  filter(Sales_LTM>0) %>%
  filter(Category_Level_1 != "Ersatzteile") %>%
  tidyr::drop_na() %>%
  ggplot(aes(x=Company, y=Preis, fill=Color)) +
  geom_boxplot(color="black", notch = TRUE) +
  geom_jitter(aes(color=PriceCheck), alpha=0.2) +
  geom_hline(data = median, aes(yintercept = Median),
             size=0.6, alpha=0.95, linetype = 1, color='#34495e') +
  facet_wrap(~Category_Level_1, scales = "free") +
  scale_fill_manual(values=c("#95A5A6","#1abc9c"), labels = c("", "")) +
  theme_minimal() +
  theme(axis.line = element_line(colour = "black"),
        plot.title = element_text(size=22),
        strip.text.x = element_text(size = 10, face="bold"),
        legend.title = element_blank(),
        panel.border = element_blank(),
        panel.background = element_rect('#F0F1F5'),
        panel.grid = element_blank(),
        legend.position = "bottom",
        axis.title.y.right = element_blank()) +
  labs(fill="")
