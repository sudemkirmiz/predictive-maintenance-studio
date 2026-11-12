"""Predictive Maintenance Studio - Streamlit dashboard.

Run from the project root:
    streamlit run app.py

The app only loads the saved model; it never trains. Train first with:
    python -m src.train_model
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.predict import (
    TRAIN_COMMAND,
    InputValidationError,
    ModelLoadError,
    load_model_bundle,
    predict_failure,
)

# Chart colors (dark surface): categorical slots + muted text/grid.
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
TEXT_MUTED, GRID = "#c3c2b7", "#383835"

MODEL_LABELS = {
    "Logistic Regression": "Logistic Regression (eşik 0.50)",
    "Random Forest baseline": "Random Forest (eşik 0.50)",
    "Random Forest @ 0.40": "Random Forest iyileştirilmiş (eşik 0.40)",
}

st.set_page_config(page_title="Predictive Maintenance Studio", page_icon="⚙️", layout="wide")


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_model_bundle():
    # Loaded once per server process and reused for every interaction.
    return load_model_bundle()


def style_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, gridColor=GRID, domain=False, tickColor=GRID)
        .configure_legend(labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, orient="top")
    )


# ---------------------------------------------------------------- header
st.title("Predictive Maintenance Studio")
st.caption(
    "UCI AI4I 2020 sentetik veri seti üzerinde makine arızası sınıflandırması. "
    "Eğitsel bir demodur; endüstriyel kullanım için doğrulanmış bir sistem değildir."
)

try:
    pipeline, metadata = get_model_bundle()
except ModelLoadError as exc:
    st.error(f"Model yüklenemedi: {exc}")
    st.code(TRAIN_COMMAND, language="bash")
    st.info("Proje klasöründe, .venv etkinken yukarıdaki komutu çalıştırın ve sayfayı yenileyin.")
    st.stop()

dataset = metadata["dataset"]
evaluation = metadata["test_evaluation"]
threshold = metadata["decision_threshold"]

tab_overview, tab_performance, tab_predict = st.tabs(["Genel Bakış", "Model Performansı", "Arıza Tahmini"])

# ---------------------------------------------------------------- A. Overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kayıt sayısı", f"{dataset['rows']:,}")
    c2.metric("Girdi özelliği", len(metadata["features"]))
    c3.metric("Arızasız kayıt", f"{dataset['non_failures']:,}", f"%{dataset['non_failures'] / dataset['rows'] * 100:.2f}", delta_color="off")
    c4.metric("Arızalı kayıt", f"{dataset['failures']:,}", f"%{dataset['failures'] / dataset['rows'] * 100:.2f}", delta_color="off")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Hedef sınıf dağılımı")
        dist = pd.DataFrame({
            "Sınıf": ["Arıza yok (0)", "Arıza (1)"],
            "Kayıt": [dataset["non_failures"], dataset["failures"]],
        })
        dist["Oran"] = dist["Kayıt"] / dataset["rows"]
        bars = alt.Chart(dist).mark_bar(cornerRadiusEnd=4, size=36).encode(
            x=alt.X("Kayıt:Q", title="Kayıt sayısı"),
            y=alt.Y("Sınıf:N", title=None, sort=None),
            color=alt.Color("Sınıf:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=None),
            tooltip=["Sınıf", alt.Tooltip("Kayıt:Q", format=","), alt.Tooltip("Oran:Q", format=".2%")],
        )
        labels = bars.mark_text(align="left", dx=6, color=TEXT_MUTED).encode(
            text=alt.Text("Oran:Q", format=".1%"), color=alt.value(TEXT_MUTED)
        )
        st.altair_chart(style_chart((bars + labels).properties(height=180)), width="stretch")
        st.caption(
            f"Sınıflar ciddi şekilde dengesiz: her arızalı kayda yaklaşık "
            f"{dataset['non_failures'] / dataset['failures']:.1f} arızasız kayıt düşüyor."
        )

    with right:
        st.subheader("Proje")
        st.markdown(
            "- **Amaç:** Sensör ölçümlerinden bir kaydın *arıza* sınıfına girip girmediğini tahmin etmek "
            "(ikili sınıflandırma).\n"
            "- **Veri:** [UCI AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), "
            "10.000 sentetik kayıt.\n"
            "- **Hedef:** `Machine failure` (0 / 1).\n"
            "- **Kullanılmayan sütunlar:** `UDI`, `Product ID` (kimlik) ve `TWF`, `HDF`, `PWF`, `OSF`, `RNF` "
            "(arıza tipleri; arızadan sonra bilinir, veri sızıntısı yaratır)."
        )

    st.subheader("Girdi özellikleri")
    ranges = metadata["feature_summary"]["numeric"]
    st.dataframe(
        pd.DataFrame([
            {"Özellik": "Type", "Tür": "Kategorik", "Eğitim verisi aralığı": "L / M / H"},
            *[
                {"Özellik": name, "Tür": "Sayısal",
                 "Eğitim verisi aralığı": f"{r['min']:g} – {r['max']:g} (medyan {r['median']:g})"}
                for name, r in ranges.items()
            ],
        ]),
        hide_index=True,
        width="stretch",
    )

# ---------------------------------------------------------------- B. Model performance
with tab_performance:
    st.info(
        "Bu sayfadaki değerler, orijinal **2.000 satırlık ayrılmış test kümesinde** yapılmış deney sonuçlarıdır "
        "(notebook 02–04 ile birebir aynı; `src/train_model.py` tarafından yeniden üretildi). "
        "Canlı tahmin değildir. 0.40 eşiği test kümesine bakılmadan, eğitim verisi içindeki "
        "validation kümesiyle seçildi."
    )

    table = pd.DataFrame(evaluation).T.rename(index=MODEL_LABELS)
    table.index.name = "Model"
    st.dataframe(
        table[["accuracy", "precision", "recall", "f1", "TP", "FP", "TN", "FN"]],
        column_config={
            "accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "f1": st.column_config.NumberColumn("F1", format="%.4f"),
        },
        width="stretch",
    )
    st.caption("Precision, Recall ve F1 arıza sınıfı (1) için hesaplanmıştır. TP: yakalanan arıza, FN: kaçırılan arıza, FP: yanlış alarm.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Arıza sınıfı metrikleri")
        long = (
            table[["precision", "recall", "f1"]]
            .reset_index()
            .melt(id_vars="Model", var_name="Metrik", value_name="Değer")
        )
        long["Metrik"] = long["Metrik"].map({"precision": "Precision", "recall": "Recall", "f1": "F1"})
        grouped = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Model:N", title=None, sort=list(MODEL_LABELS.values()), axis=alt.Axis(labelAngle=0, labelLimit=220)),
            xOffset=alt.XOffset("Metrik:N", sort=["Precision", "Recall", "F1"]),
            y=alt.Y("Değer:Q", title="Skor", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Metrik:N", sort=["Precision", "Recall", "F1"],
                            scale=alt.Scale(domain=["Precision", "Recall", "F1"], range=[BLUE, ORANGE, AQUA]),
                            legend=alt.Legend(title=None)),
            tooltip=["Model", "Metrik", alt.Tooltip("Değer:Q", format=".4f")],
        )
        st.altair_chart(style_chart(grouped.properties(height=320)), width="stretch")

    with right:
        st.subheader("Final confusion matrix")
        final = evaluation["Random Forest @ 0.40"]
        cm = pd.DataFrame([
            {"Gerçek": "Arıza yok (0)", "Tahmin": "Arıza yok (0)", "Kayıt": final["TN"], "Tür": "TN"},
            {"Gerçek": "Arıza yok (0)", "Tahmin": "Arıza (1)", "Kayıt": final["FP"], "Tür": "FP - yanlış alarm"},
            {"Gerçek": "Arıza (1)", "Tahmin": "Arıza yok (0)", "Kayıt": final["FN"], "Tür": "FN - kaçırılan arıza"},
            {"Gerçek": "Arıza (1)", "Tahmin": "Arıza (1)", "Kayıt": final["TP"], "Tür": "TP - yakalanan arıza"},
        ])
        order = ["Arıza yok (0)", "Arıza (1)"]
        base = alt.Chart(cm).encode(
            x=alt.X("Tahmin:N", sort=order, title="Tahmin", axis=alt.Axis(labelAngle=0, orient="top")),
            y=alt.Y("Gerçek:N", sort=order, title="Gerçek"),
        )
        cells = base.mark_rect(stroke="#141413", strokeWidth=2).encode(
            # Log scale so the small cells stay visible next to 1923 TN.
            color=alt.Color("Kayıt:Q", scale=alt.Scale(type="log", range=["#1c2a3d", "#2a78d6"]), legend=None),
            tooltip=["Tür", "Gerçek", "Tahmin", "Kayıt"],
        )
        text = base.mark_text(fontSize=20, fontWeight="bold", color="#f0efec").encode(text="Kayıt:Q")
        st.altair_chart(style_chart((cells + text).properties(height=280)), width="stretch")
        st.caption(f"Random Forest, eşik {threshold:.2f}. Test kümesindeki {final['TP'] + final['FN']} arızanın {final['TP']}'ü yakalandı, {final['FN']}'i kaçırıldı.")

# ---------------------------------------------------------------- C. Failure prediction
with tab_predict:
    st.warning(
        "Canlı tahmin - sentetik veriyle eğitilmiş demo modeldir. Model skoru doğrulanmış gerçek bir arıza "
        "olasılığı değildir ve arızanın ne zaman olacağını tahmin etmez.",
        icon="⚠️",
    )

    summary = metadata["feature_summary"]
    med = {name: r["median"] for name, r in summary["numeric"].items()}
    type_options = summary["type_categories"]

    with st.form("prediction_form"):
        st.caption("Varsayılan değerler eğitim verisinin medyanlarıdır; Type için en sık görülen tip seçilidir.")
        col1, col2, col3 = st.columns(3)
        with col1:
            product_type = st.selectbox(
                "Type", type_options, index=type_options.index(summary["type_most_frequent"]),
                help="Ürün kalite tipi: L (düşük), M (orta), H (yüksek).",
            )
            air_temp = st.number_input("Air temperature [K]", value=med["Air temperature [K]"], step=0.1, format="%.1f")
        with col2:
            process_temp = st.number_input("Process temperature [K]", value=med["Process temperature [K]"], step=0.1, format="%.1f")
            rpm = st.number_input("Rotational speed [rpm]", value=int(med["Rotational speed [rpm]"]), step=10)
        with col3:
            torque = st.number_input("Torque [Nm]", value=med["Torque [Nm]"], step=0.1, format="%.1f")
            tool_wear = st.number_input("Tool wear [min]", value=int(med["Tool wear [min]"]), step=1)
        submitted = st.form_submit_button("Tahmin et", type="primary", width="stretch")

    if submitted:
        inputs = {
            "Type": product_type,
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }
        try:
            result = predict_failure(pipeline, metadata, inputs)
        except InputValidationError as exc:
            st.error("Girdiler geçersiz:\n\n" + "\n".join(f"- {e}" for e in exc.errors))
        else:
            for message in result["warnings"]:
                st.warning(message)

            r1, r2, r3 = st.columns(3)
            r1.metric("Model skoru (arıza sınıfı)", f"%{result['score'] * 100:.1f}")
            r2.metric("Karar eşiği", f"%{result['threshold'] * 100:.0f}")
            r3.metric("Tahmin edilen sınıf", "Arıza" if result["prediction"] else "Arıza yok")

            if result["prediction"] == 1:
                st.error(
                    f"**Arıza (1)** - skor %{result['score'] * 100:.1f}, eşik %{result['threshold'] * 100:.0f} "
                    "değerine eşit veya üzerinde.", icon="🛑",
                )
            else:
                st.success(
                    f"**Arıza yok (0)** - skor %{result['score'] * 100:.1f}, eşik %{result['threshold'] * 100:.0f} "
                    "değerinin altında.", icon="✅",
                )
            st.progress(min(result["score"], 1.0), text=f"Model skoru %{result['score'] * 100:.1f} (eşik %{result['threshold'] * 100:.0f})")
            st.caption(
                "Skor, Random Forest ağaçlarının bu kaydı arıza sınıfına atama oranına dayanır. "
                "Eşik, kaçırılan arızaları azaltmak için 0.50 yerine 0.40 olarak seçildi."
            )

st.divider()
st.caption(
    f"Model: {metadata['model']} · eğitim: {metadata['train_rows']:,} kayıt · "
    f"scikit-learn {metadata['sklearn_version']} · Eğitsel demo, sentetik veri."
)
