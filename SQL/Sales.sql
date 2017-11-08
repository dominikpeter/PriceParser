
SELECT UniqueId
  ,Sales_LTM = sum(Sales)
  ,Margin_LTM = sum(Margin)
  ,Quantity_LTM = sum(Quantity)
  ,ObjectRate = max(ObjectRate)

FROM (

SELECT UniqueId = idItemOld
  ,Sales = sum(Sales)
  ,Margin = sum(Margin)
  ,Quantity = sum(Quantity)
  ,ObjectRate = avg(case when PricingLong IN ('Offerte', 'Baustelle', 'Aktionspreise') then 1.0 else 0.0 end)
FROM CRHBUSADWH01.InfoPool.FACT.V_Sales s
  inner join CRHBUSADWH01.infopool.dim.v_item i on i.iditem = s.iditem and i.idbusinesssection = s.idbusinesssection
  inner join [CRHBUSADWH01].[InfoPool].[DIM].[V_Pricing] p on p.idpricing = s.idpricing
where itemgroupgrouphierarchyname_l1 = '05-Sanitär'
and Date > dateadd(month, -12, getdate())
and Sales > 0
group by idItemOld

UNION ALL

SELECT UniqueId = substring(i.iditemorigin, 2, 500)
  ,Sales = sum(Sales)
  ,Margin = sum(Margin)
  ,Quantity = sum(Quantity)
  ,ObjectRate = 0
FROM InfoPool_MOV.FACT.Sales s
  inner join InfoPool_MOV.DIM.v_item i on i.iditem = s.iditem
where itemgroupgrouphierarchyname_l1 = '05-Sanitär'
and Date > dateadd(month, -12, getdate())
and Sales > 0
group by substring(i.iditemorigin, 2, 500)

) x
GROUP BY UniqueId
