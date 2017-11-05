

library(dplyr)
library(ggplot2)
library(stringr)
library(readr)
library(reshape2)


df_raw <- readr::read_delim("c:/users/peterd/Desktop/IGH/2017-11-03_Price-Comparison.csv", delim=';')


df <- df_raw %>% 
  select(ArtikelId,Category_Level_1, Category_Level_2, Farbe, ObjectRate,
         Lieferantenname, Artikelserie,Sales_LTM,
         grep("Preis.+", names(df_raw))) %>% 
  select(-Preis_EAN)



dd <- reshape2::melt(df, id.vars=1:9, variable.name='Company', value.name="Preis") %>% 
  as_data_frame() %>%
  mutate(Company = stringr::str_sub(Company, 7, -1),
         Preis_Check = ifelse(Preis_Pos==Preis,"Preis gleich",ifelse(Preis_Pos>Preis,"Preis höher", "Preis tiefer")),
         Preis_Check = Preis_Check %>% as.factor) %>% 
  filter(!is.na(Preis_Check))



dd %>% 
  filter(Company %in% c("Sabag")) %>% 
  # filter(Preis < 7000) %>% 
  # filter(Preis_Pos < 7000) %>% 
  ggplot(aes(x=Preis_Pos, y=Preis, size=Sales_LTM, color=Preis_Check)) +
  geom_point(alpha=0.7) +
  scale_x_log10()+
  scale_y_log10()+SSSS
  facet_wrap(~Category_Level_1, ncol=2)+
  scale_color_manual(values=c("#34495e","#1abc9c", "#e74c3c")) +
  theme_minimal() +
  theme(axis.line = element_line(colour = "black"),
        plot.title = element_text(size=22),
        strip.text.x = element_text(size = 10, face="bold"),
        legend.title = element_blank(),
        panel.border = element_blank(),
        panel.background = element_rect('#F0F1F5'),
        panel.grid = element_blank(),
        legend.position = "bottom",
        axis.title.y.right = element_blank())

dd %>% 
  filter(Company %in% c("Sanitas")) %>% 
  # filter(Preis < 7000) %>% 
  # filter(Preis_Pos < 7000) %>% 
  ggplot(aes(x=Preis_Pos, y=Preis, size=Sales_LTM, color=Preis_Check)) +
  geom_point(alpha=0.7) +
  scale_x_log10()+
  scale_y_log10()+
  facet_wrap(~Category_Level_1, ncol=2)+
  scale_color_manual(values=c("#34495e","#1abc9c", "#e74c3c")) +
  theme_minimal() +
  theme(axis.line = element_line(colour = "black"),
        plot.title = element_text(size=22),
        strip.text.x = element_text(size = 10, face="bold"),
        legend.title = element_blank(),
        panel.border = element_blank(),
        panel.background = element_rect('#F0F1F5'),
        panel.grid = element_blank(),
        legend.position = "bottom",
        axis.title.y.right = element_blank())




ddd <- reshape2::melt(df, id.vars=1:8, variable.name='Company', value.name="Preis") %>% 
  as_data_frame() %>%
  mutate(Company = stringr::str_sub(Company, 7, -1),
         Preis_Check = ifelse(Preis_Pos==Preis,"Preis gleich",ifelse(Preis_Pos>Preis,"Preis höher", "Preis tiefer")),
         Preis_Check = Preis_Check %>% as.factor) %>% 
  filter(!is.na(Preis_Check))



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





df_boxplot %>% 
  filter(Preis<1000) %>% 
  filter(Sales_LTM>0) %>% 
  filter(Category_Level_1 != "Ersatzteile") %>% 
  tidyr::drop_na() %>% 
  ggplot(aes(x=Company, y=Preis, fill=Color)) +
  geom_boxplot(color="black", notch = TRUE) +
  facet_wrap(~Category_Level_1) +
  scale_fill_manual(values=c("#95A5A6","#1abc9c")) +
  theme_minimal() +
  theme(axis.line = element_line(colour = "black"),
        plot.title = element_text(size=22),
        strip.text.x = element_text(size = 10, face="bold"),
        legend.title = element_blank(),
        panel.border = element_blank(),
        panel.background = element_rect('#F0F1F5'),
        panel.grid = element_blank(),
        legend.position = "bottom",
        axis.title.y.right = element_blank())









