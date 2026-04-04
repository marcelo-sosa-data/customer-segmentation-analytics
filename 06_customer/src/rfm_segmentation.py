"""
rfm_segmentation.py
Segmentación de clientes usando análisis RFM + K-Means.
Ejecutar: python src/rfm_segmentation.py
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go


def calculate_rfm(df: pd.DataFrame,
                  customer_col="customer_id",
                  date_col="purchase_date",
                  revenue_col="revenue",
                  ref_date=None) -> pd.DataFrame:
    """Calcula métricas RFM y asigna scores 1-5."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    ref = ref_date or df[date_col].max()

    rfm = df.groupby(customer_col).agg(
        recency   = (date_col,    lambda x: (ref - x.max()).days),
        frequency = (date_col,    "count"),
        monetary  = (revenue_col, "sum"),
    ).reset_index()

    rfm["R"] = pd.qcut(rfm["recency"],   q=5, labels=[5,4,3,2,1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_score"] = rfm["R"] + rfm["F"] + rfm["M"]

    def segment(row):
        r, f, m = row["R"], row["F"], row["M"]
        if r>=4 and f>=4 and m>=4: return "🥇 Campeones"
        if r>=3 and f>=3 and m>=3: return "💎 Leales"
        if r>=4 and f<=2:          return "🌱 Nuevos Prometedores"
        if r<=2 and f>=3 and m>=3: return "😴 En Riesgo"
        if r<=2 and f>=4 and m>=4: return "🚨 No Perder"
        if r<=1 and f<=2:          return "💤 Hibernando"
        return "🔵 Regulares"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


def cluster_kmeans(rfm: pd.DataFrame, k: int = None) -> pd.DataFrame:
    """Clustering K-Means sobre métricas RFM normalizadas."""
    features = ["recency", "frequency", "monetary"]
    X = StandardScaler().fit_transform(rfm[features])

    if k is None:
        # Selección automática por Silhouette
        scores = {}
        for ki in range(2, 9):
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            scores[ki] = silhouette_score(X, km.fit_predict(X))
        k = max(scores, key=scores.get)
        print(f"✅ K óptimo: {k} (Silhouette={scores[k]:.3f})")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["cluster"] = km.fit_predict(X)
    return rfm


def segment_report(rfm: pd.DataFrame) -> pd.DataFrame:
    """Reporte ejecutivo con métricas por segmento."""
    summary = rfm.groupby("segment").agg(
        clientes       = ("customer_id", "count"),
        recencia_prom  = ("recency",     "mean"),
        frecuencia_prom= ("frequency",   "mean"),
        monto_prom     = ("monetary",    "mean"),
        monto_total    = ("monetary",    "sum"),
    ).round(1).reset_index()

    summary["pct_clientes"] = (summary["clientes"] / summary["clientes"].sum() * 100).round(1)
    summary["pct_revenue"]  = (summary["monto_total"] / summary["monto_total"].sum() * 100).round(1)
    summary = summary.sort_values("pct_revenue", ascending=False)

    print(f"\n{'═'*70}")
    print("📊 REPORTE DE SEGMENTACIÓN RFM")
    print(f"{'═'*70}")
    print(summary[["segment","clientes","pct_clientes","pct_revenue","recencia_prom","monto_prom"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    df = pd.read_csv("data/sample/transactions_sample.csv")

    print("🔄 Calculando métricas RFM...")
    rfm = calculate_rfm(df)

    print("🔄 Aplicando K-Means...")
    rfm = cluster_kmeans(rfm)

    report = segment_report(rfm)

    # Visualización 3D
    fig = px.scatter_3d(
        rfm, x="recency", y="frequency", z="monetary",
        color="segment", title="Segmentación de Clientes — Espacio RFM 3D",
        template="plotly_dark", opacity=0.7,
    )
    fig.show()

    rfm.to_csv("data/sample/rfm_results.csv", index=False)
    print("\n✅ Resultados guardados en data/sample/rfm_results.csv")
