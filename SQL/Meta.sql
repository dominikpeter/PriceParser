
SELECT UniqueId = [SGVSB]
      ,[Lieferantenname]
      ,Erstellt_Am = [Erstellt Am]
      ,[Artikelserie]
FROM [AnalystCM].[dbo].[LOOKUP_ArtikelstammCRHT]
WHERE [Einmalartikel (J/N)] = 0 and (Artikelgruppe BETWEEN 5000 AND 6000 or
                                    Artikelgruppe like '03%')
AND SGVSB > '' AND Lieferantenname <> 'Loosli Badmöbel AG'
