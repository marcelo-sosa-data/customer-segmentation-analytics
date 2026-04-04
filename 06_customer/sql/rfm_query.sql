-- rfm_query.sql — Segmentación RFM en PostgreSQL

WITH rfm_base AS (
    SELECT
        customer_id,
        CURRENT_DATE - MAX(purchase_date)::DATE  AS recency_days,
        COUNT(*)                                  AS frequency,
        SUM(revenue)                              AS monetary
    FROM transactions
    WHERE purchase_date >= CURRENT_DATE - INTERVAL '2 years'
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency)         AS f_score,
        NTILE(5) OVER (ORDER BY monetary)          AS m_score
    FROM rfm_base
)
SELECT *,
    r_score + f_score + m_score AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Campeones'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Leales'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'Nuevos Prometedores'
        WHEN r_score <= 2 AND f_score >= 3 AND m_score >= 3 THEN 'En Riesgo'
        WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 4 THEN 'No Perder'
        WHEN r_score <= 1 AND f_score <= 2                  THEN 'Hibernando'
        ELSE 'Regulares'
    END AS segment
FROM rfm_scores
ORDER BY rfm_total DESC;
