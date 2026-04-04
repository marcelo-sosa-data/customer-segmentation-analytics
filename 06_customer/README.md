# 👥 Customer Segmentation Analytics
> Segmentación de clientes con **K-Means + RFM Analysis** para estrategia comercial accionable

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](.)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](.)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)](.)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)](.)

---

## 📌 Contexto de Negocio

No todos los clientes son iguales. Tratar a todos igual desperdicia presupuesto de marketing y pierde oportunidades. Este proyecto segmenta la base de clientes usando **análisis RFM** (Recencia, Frecuencia, Monto) y **clustering K-Means** para identificar grupos con comportamiento similar y definir estrategias específicas por segmento.

**Preguntas que responde:**
- ¿Quiénes son mis clientes más valiosos?
- ¿Quiénes están a punto de abandonar?
- ¿A quiénes debo hacerles upselling?
- ¿Cómo personalizo mis campañas por segmento?

---

## 📂 Estructura

```
customer-segmentation-analytics/
├── 📁 data/
│   └── sample/
│       └── transactions_sample.csv     # Dataset simulado de muestra
├── 📁 notebooks/
│   ├── 01_EDA_customers.ipynb          # Exploración de base de clientes
│   ├── 02_RFM_analysis.ipynb           # Construcción de métricas RFM
│   ├── 03_clustering_kmeans.ipynb      # Segmentación con K-Means
│   └── 04_segment_profiling.ipynb      # Perfil e insights por segmento
├── 📁 src/
│   ├── rfm_calculator.py               # Cálculo de métricas RFM
│   ├── segmentation_model.py           # Pipeline de clustering
│   ├── segment_profiler.py             # Análisis de perfiles
│   └── visualizations.py              # Gráficos interactivos
├── 📁 dashboards/
│   └── segmentation_dashboard.pbix    # Dashboard Power BI
├── requirements.txt
└── README.md
```

---

## 🔑 Código Core

### `src/rfm_calculator.py`
```python
"""
Cálculo de métricas RFM (Recencia, Frecuencia, Monto).
Base para segmentación de clientes.
"""
import pandas as pd
import numpy as np
from datetime import datetime


class RFMCalculator:
    """
    Calcula métricas RFM y asigna scores del 1 al 5 por quintiles.

    R (Recencia)   → ¿Cuándo compró por última vez? (menor = mejor)
    F (Frecuencia) → ¿Cuántas veces compró? (mayor = mejor)
    M (Monto)      → ¿Cuánto gastó en total? (mayor = mejor)
    """

    def __init__(self, customer_col: str = "customer_id",
                 date_col: str = "purchase_date",
                 revenue_col: str = "revenue",
                 reference_date: datetime = None):
        self.customer_col   = customer_col
        self.date_col       = date_col
        self.revenue_col    = revenue_col
        self.reference_date = reference_date or datetime.today()

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas RFM y scores por cliente."""
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # ── Métricas base ────────────────────────────────────────────────────
        rfm = df.groupby(self.customer_col).agg(
            recency   = (self.date_col,    lambda x: (self.reference_date - x.max()).days),
            frequency = (self.date_col,    "count"),
            monetary  = (self.revenue_col, "sum"),
        ).reset_index()

        # ── Scores 1-5 por quintiles ─────────────────────────────────────────
        # Recencia: menor días = mejor = score 5
        rfm["R_score"] = pd.qcut(rfm["recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
        rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
        rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)

        # ── Score compuesto ──────────────────────────────────────────────────
        rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
        rfm["RFM_total"] = rfm[["R_score","F_score","M_score"]].sum(axis=1)

        # ── Segmento por reglas de negocio ───────────────────────────────────
        rfm["segment"] = rfm.apply(self._assign_segment, axis=1)

        print(f"✅ RFM calculado para {len(rfm):,} clientes")
        print(f"\n📊 Distribución de segmentos:\n{rfm['segment'].value_counts().to_string()}")
        return rfm

    @staticmethod
    def _assign_segment(row) -> str:
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:   return "🥇 Campeones"
        if r >= 3 and f >= 3 and m >= 3:   return "💎 Leales"
        if r >= 4 and f <= 2:              return "🌱 Nuevos Prometedores"
        if r >= 3 and f >= 2 and m >= 3:   return "⬆️  Potencial Alto"
        if r <= 2 and f >= 3 and m >= 3:   return "😴 En Riesgo"
        if r <= 2 and f >= 4 and m >= 4:   return "🚨 No Perder"
        if r <= 1 and f <= 2:              return "💤 Hibernando"
        return "🔵 Regulares"
```

### `src/segmentation_model.py`
```python
"""
Pipeline de clustering K-Means con selección óptima de K.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px


class CustomerSegmentationModel:
    """K-Means con selección automática de K usando Elbow + Silhouette."""

    def __init__(self, features: list[str] = None, max_k: int = 10, random_state: int = 42):
        self.features     = features or ["recency","frequency","monetary"]
        self.max_k        = max_k
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.best_k_      = None
        self.model_       = None
        self.labels_      = None

    def find_optimal_k(self, df: pd.DataFrame) -> int:
        """Método del codo + Silhouette para encontrar K óptimo."""
        X = self.scaler.fit_transform(df[self.features])
        inertias, silhouettes = [], []

        for k in range(2, self.max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, labels))

        # K óptimo = mayor Silhouette score
        self.best_k_ = silhouettes.index(max(silhouettes)) + 2
        print(f"✅ K óptimo: {self.best_k_} (Silhouette={max(silhouettes):.3f})")
        return self.best_k_

    def fit_predict(self, df: pd.DataFrame, k: int = None) -> pd.DataFrame:
        """Entrena K-Means y agrega etiquetas al DataFrame."""
        if k is None:
            k = self.find_optimal_k(df)

        X = self.scaler.fit_transform(df[self.features])
        self.model_  = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        self.labels_ = self.model_.fit_predict(X)

        df = df.copy()
        df["cluster"] = self.labels_
        df["cluster_label"] = df["cluster"].map(
            {i: f"Segmento {i+1}" for i in range(k)}
        )
        return df

    def plot_clusters_3d(self, df: pd.DataFrame) -> go.Figure:
        """Visualización 3D de clusters en espacio RFM."""
        fig = px.scatter_3d(
            df, x="recency", y="frequency", z="monetary",
            color="cluster_label",
            hover_data=["customer_id"] if "customer_id" in df.columns else None,
            title="🎯 Segmentación de Clientes — Espacio RFM 3D",
            labels={"recency":"Recencia (días)","frequency":"Frecuencia","monetary":"Monto ($)"},
            template="plotly_dark", opacity=0.7,
        )
        return fig

    def plot_segment_radar(self, df: pd.DataFrame) -> go.Figure:
        """Gráfico radar con el perfil promedio de cada segmento."""
        profile = df.groupby("cluster_label")[self.features].mean()
        # Normalizar 0-1 para comparar en misma escala
        profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

        fig = go.Figure()
        colors = px.colors.qualitative.Set2

        for i, (seg, row) in enumerate(profile_norm.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=row.tolist() + [row.tolist()[0]],
                theta=self.features + [self.features[0]],
                fill="toself", name=seg,
                line_color=colors[i % len(colors)],
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            title="📊 Perfil de Segmentos — Radar Chart",
            template="plotly_dark",
        )
        return fig
```

### `src/segment_profiler.py`
```python
"""
Genera insights accionables y estrategias por segmento.
"""
import pandas as pd


SEGMENT_STRATEGIES = {
    "🥇 Campeones": {
        "descripcion": "Compraron recientemente, con alta frecuencia y alto gasto.",
        "estrategia":  "Recompensar con programa VIP. Pedir reviews y referidos.",
        "accion":      "Descuento exclusivo + acceso anticipado a nuevos productos.",
        "riesgo":      "Bajo",
        "prioridad":   1,
    },
    "💎 Leales": {
        "descripcion": "Compran regularmente con buen monto. Base sólida.",
        "estrategia":  "Upselling y cross-selling. Membresías premium.",
        "accion":      "Oferta de upgrade de plan con beneficio claro.",
        "riesgo":      "Bajo",
        "prioridad":   2,
    },
    "😴 En Riesgo": {
        "descripcion": "Compraron bien antes pero llevan tiempo sin actividad.",
        "estrategia":  "Campaña de reactivación urgente con incentivo.",
        "accion":      "Email personalizado: '¡Te extrañamos! 20% de descuento'.",
        "riesgo":      "Alto",
        "prioridad":   1,
    },
    "🚨 No Perder": {
        "descripcion": "Clientes de alto valor histórico en riesgo de churn.",
        "estrategia":  "Contacto directo del equipo comercial. Oferta especial.",
        "accion":      "Llamada o WhatsApp personalizado. Descuento de retención.",
        "riesgo":      "Muy Alto",
        "prioridad":   1,
    },
    "🌱 Nuevos Prometedores": {
        "descripcion": "Compraron recientemente pero poca frecuencia aún.",
        "estrategia":  "Onboarding y educación. Segunda compra con descuento.",
        "accion":      "Secuencia de emails educativos + oferta segunda compra.",
        "riesgo":      "Medio",
        "prioridad":   2,
    },
    "💤 Hibernando": {
        "descripcion": "Sin actividad reciente ni historial sólido.",
        "estrategia":  "Campaña masiva de bajo costo. Win-back o dar de baja.",
        "accion":      "Email automatizado con oferta agresiva. Si no responde → baja.",
        "riesgo":      "Muy Alto",
        "prioridad":   3,
    },
}

def generate_segment_report(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Genera reporte ejecutivo con métricas y estrategias por segmento."""
    summary = rfm_df.groupby("segment").agg(
        total_clientes = ("customer_id", "count"),
        recencia_prom  = ("recency",     "mean"),
        frecuencia_prom= ("frequency",   "mean"),
        monto_prom     = ("monetary",    "mean"),
        monto_total    = ("monetary",    "sum"),
    ).round(1).reset_index()

    summary["% clientes"] = (summary["total_clientes"] / summary["total_clientes"].sum() * 100).round(1)
    summary["% revenue"]  = (summary["monto_total"]    / summary["monto_total"].sum()    * 100).round(1)

    # Agregar estrategias
    summary["estrategia"] = summary["segment"].map(
        {k: v["estrategia"] for k, v in SEGMENT_STRATEGIES.items()}
    )
    summary["prioridad"] = summary["segment"].map(
        {k: v["prioridad"] for k, v in SEGMENT_STRATEGIES.items()}
    )

    summary = summary.sort_values(["prioridad","% revenue"], ascending=[True, False])
    print(f"\n{'═'*70}\n📊 REPORTE DE SEGMENTACIÓN\n{'═'*70}")
    print(summary[["segment","total_clientes","% clientes","% revenue","estrategia"]].to_string(index=False))
    return summary
```

---

## 📊 SQL — Segmentación en PostgreSQL

```sql
-- Calcular RFM directamente en SQL
WITH rfm_base AS (
    SELECT
        customer_id,
        CURRENT_DATE - MAX(purchase_date)::DATE     AS recency_days,
        COUNT(*)                                     AS frequency,
        SUM(revenue)                                 AS monetary
    FROM transactions
    WHERE purchase_date >= CURRENT_DATE - INTERVAL '2 years'
    GROUP BY customer_id
),
rfm_scores AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days DESC)  AS r_score,
        NTILE(5) OVER (ORDER BY frequency)          AS f_score,
        NTILE(5) OVER (ORDER BY monetary)           AS m_score
    FROM rfm_base
)
SELECT *,
    r_score + f_score + m_score                     AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN '🥇 Campeones'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN '💎 Leales'
        WHEN r_score >= 4 AND f_score <= 2           THEN '🌱 Nuevos Prometedores'
        WHEN r_score <= 2 AND f_score >= 3           THEN '😴 En Riesgo'
        WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 4 THEN '🚨 No Perder'
        WHEN r_score <= 1 AND f_score <= 2           THEN '💤 Hibernando'
        ELSE '🔵 Regulares'
    END AS segment
FROM rfm_scores
ORDER BY rfm_total DESC;
```

---

## 📈 Power BI — KPIs del Dashboard

| Visual | Métrica |
|---|---|
| 🎯 Treemap | Clientes y revenue por segmento |
| 📊 Scatter | Frecuencia vs. Monto por segmento |
| 🗺️ Mapa | Distribución geográfica por segmento |
| 📈 Tendencia | Evolución de segmentos mes a mes |
| 🏆 Ranking | Top 20 clientes por RFM score |

---

## 💼 Insights de Negocio Típicos

```
Los "Campeones" (8% de clientes) generan el 38% del revenue
Los "En Riesgo" representan $45K en revenue potencial a recuperar
Los "Nuevos Prometedores" tienen 3x más probabilidad de volverse "Leales"
Campaña de reactivación "En Riesgo" → ROI estimado 4:1
```

---
*Stack: Python · Scikit-learn · Plotly · PostgreSQL · Power BI*
