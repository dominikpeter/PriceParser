

set nocount on;

IF OBJECT_ID('tempdb..#dch') IS NOT NULL DROP TABLE #dch
IF OBJECT_ID('tempdb..#sales') IS NOT NULL DROP TABLE #sales

SELECT [Artikelnummer]
      ,[Artikelgruppe]
      ,[Rabattgruppe]
      ,[Lieferantenname]
      ,[Erstellt Am]
      ,[Artikelserie]
      ,[MovexNr]
      ,[SGVSB]
      ,[Bruttoverkauf]
      ,[Standardkosten]
into #dch
FROM [AnalystCM].[dbo].[LOOKUP_ArtikelstammCRHT]
where sgvsb > '' and [Einmalartikel (J/N)] = 0 and Lieferant <> '00008537'


SELECT UniqueId
  ,GrossSales_LTM = sum(GrossSales)
  ,Sales_LTM = sum(Sales)
  ,Margin_LTM = sum(Margin)
  ,Quantity_LTM = sum(Quantity)
  ,ObjectRate = max(ObjectRate)
  ,CountOfOrders = sum(CountOfOrders)
  ,CountOfCustomers = sum(CountOfCustomers)
into #sales
FROM (
SELECT UniqueId = idItemOld
  ,GrossSales = sum(GrossSales)
  ,Sales = sum(Sales)
  ,Margin = sum(Margin)
  ,Quantity = sum(Quantity)
  ,ObjectRate = avg(case when PricingLong IN ('Offerte', 'Baustelle', 'Aktionspreise') then 1.0 else 0.0 end)
  ,CountOfOrders = count(distinct OrderNo)
  ,CountOfCustomers = count(distinct idCustomer)
FROM CRHBUSADWH01.InfoPool.FACT.V_Sales s
  inner join CRHBUSADWH01.infopool.dim.v_item i on i.iditem = s.iditem and i.idbusinesssection = s.idbusinesssection
  inner join [CRHBUSADWH01].[InfoPool].[DIM].[V_Pricing] p on p.idpricing = s.idpricing
where itemgroupgrouphierarchyname_l1 = '05-Sanitär'
and Date > dateadd(month, -12, getdate())
and Sales > 0
group by idItemOld

UNION ALL

SELECT UniqueId = substring(i.iditemorigin, 2, 500)
  ,GrossSales = sum(GrossSales)
  ,Sales = sum(Sales)
  ,Margin = sum(Margin)
  ,Quantity = sum(Quantity)
  ,ObjectRate = 0
  ,CountOfOrders = count(distinct OrderNo)
  ,CountOfCustomers = count(distinct idCustomer)
FROM InfoPool_MOV.FACT.Sales s
  inner join InfoPool_MOV.DIM.v_item i on i.iditem = s.iditem
where itemgroupgrouphierarchyname_l1 = '05-Sanitär'
and Date > dateadd(month, -12, getdate())
and Sales > 0
group by substring(i.iditemorigin, 2, 500), Supplier
) x
GROUP BY UniqueId
HAVING sum(Sales) > 0


SELECT UniqueId = SGVSB
      ,[Article_group] = [Artikelgruppe]
      ,[Discount_group] = [Rabattgruppe]
      ,[Suppliername] = [Lieferantenname]
      ,[Creation_Date] = [Erstellt Am]
      ,[Brand Name] = [Artikelserie]
      ,[Purchaseprice] = [Standardkosten]
	  ,GrossSales_LTM
	  ,Sales_LTM
	  ,Margin_LTM
	  ,Quantity_LTM
	  ,ObjectRate
	  ,CountOfOrders
	  ,CountOfCustomers
FROM #dch x
	left outer join #sales s on s.UniqueId = x.SGVSB collate SQL_Latin1_General_CP1_CI_AI


IF OBJECT_ID('tempdb..#dch') IS NOT NULL DROP TABLE #dch
IF OBJECT_ID('tempdb..#sales') IS NOT NULL DROP TABLE #sales
