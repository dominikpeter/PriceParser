SELECT ArtikelId = Artikelnummer
,FarbId = ''
,AusführungsId = ''
,Preis_Pos = [Bruttoverkauf]
,Preis_EAN = [Barcode]
,Art_Nr_Hersteller = [Lieferantennummer]
,Art_Nr_Hersteller_Firma = [Lieferantenname]
,Art_Nr_EAN = [Barcode]
,Art_Nr_Nachfolge = ''
,Art_Nr_Synonym = ''
,Art_Nr_Synonym_Firma = ''
,Art_Valid_Von = ''
,Art_Valid_Bis = ''
,Art_Txt_Kurz = [Artikel Beschreibung]
,Art_Txt_Lang = [Artikel Beschreibung]
,Art_Menge = ''
,BM_Einheit_Code = ''
,BM_Einheit_Code_BM_Einheit = ''
,AF_Nr = ''
,AF_Txt = ''
,AFZ_Txt = ''
,AFZ_Nr = ''
,Preis = [Bruttoverkauf]
,Category_Level_1 = [Artikelgruppe Beschreibung]
,Category_Level_2 = [Rabattgruppe Beschreibung]
,Category_Level_3 = [Rabattgruppe Beschreibung]
,Category_Level_4 = [Rabattgruppe Beschreibung]
  FROM [AnalystCM].[dbo].[LOOKUP_ArtikelstammCRHT]
WHERE Artikelgruppe in ('5620','5625','5630','5635','5640'
                        ,'5645','5650','5651','5655','5656'
                        ,'5660','5661','5665','5670','5671'
                        ,'5675','5700','5725')
and [Artikel Status] = '02' and [Einmalartikel (J/N)] = 0
