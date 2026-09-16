from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from climate_advice import CROP_RULES, fetch_open_meteo, generate_climate_advice, summarize_weather
from gradcam import GradCAM, overlay_heatmap_on_image
from reliability import assess_image_quality, safe_predict
from train_plant_disease import build_model
from yield_estimation import estimate_yield, health_factor_from_diagnosis

# Configuration de la page
st.set_page_config(
    page_title="AgriIA - Diagnostic Phytosanitaire & Conseils Agricoles",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS pour une interface soignée
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4a5568;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-high {
        background-color: #d1fae5;
        color: #065f46;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-med {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-rej {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint["architecture"],
        checkpoint["num_classes"],
        freeze_backbone=checkpoint.get("freeze_backbone", False),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, device


def find_available_checkpoints() -> list[Path]:
    return sorted(Path("outputs").rglob("best_model.pt"))


# Barre latérale
st.sidebar.title("⚙️ Paramètres AgriIA")

available_models = find_available_checkpoints()
if not available_models:
    st.sidebar.error("Aucun modèle 'best_model.pt' trouvé dans le dossier 'outputs'.")
    st.stop()

selected_model_path = st.sidebar.selectbox(
    "Modèle IA actif :",
    options=available_models,
    format_func=lambda p: f"{p.parent.name} ({p})",
)

labels_path = selected_model_path.parent / "label_map.json"
if not labels_path.exists():
    st.sidebar.error(f"Fichier label_map.json introuvable dans {selected_model_path.parent}")
    st.stop()

with open(labels_path, "r", encoding="utf-8") as f:
    labels_map = json.load(f)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Garde-fous & Fiabilité")
min_conf = st.sidebar.slider("Seuil de certitude minimale :", min_value=0.50, max_value=0.95, value=0.75, step=0.05)
min_sharp = st.sidebar.slider("Seuil de netteté minimale (anti-flou) :", min_value=10.0, max_value=120.0, value=45.0, step=5.0)
use_tta = st.sidebar.checkbox("Activer le Test-Time Augmentation (TTA)", value=True, help="Teste la photo sous 5 angles pour vérifier la stabilité du diagnostic.")

model, checkpoint, device = load_cached_model(str(selected_model_path))
image_size = checkpoint.get("image_size", 224)
normalize_mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
normalize_std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])

eval_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ]
)


# En-tête principal
st.markdown('<div class="main-title">🌿 AgriIA — Diagnostic & Décision Agronomique</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Vision par ordinateur certifiée, explicabilité Grad-CAM, alertes climatiques et prévision de rendement.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔬 Diagnostic & Explicabilité", "🌦️ Météo & Risques Fongiques", "📈 Estimation de Rendement"])

# -------------------------------------------------------------
# ONGLET 1 : DIAGNOSTIC & GRAD-CAM
# -------------------------------------------------------------
with tab1:
    st.subheader("Analyse de feuille et vérification de fiabilité")
    col_input, col_result = st.columns([1, 1])

    with col_input:
        uploaded_file = st.file_uploader("Téléchargez une photo de feuille :", type=["jpg", "jpeg", "png"])
        
        # Option de démonstration si aucune image chargée
        sample_images = sorted(Path("data").rglob("*.jpg"))
        selected_sample = None
        if not uploaded_file and sample_images:
            use_sample = st.checkbox("Ou tester avec une photo d'exemple locale")
            if use_sample:
                selected_sample = st.selectbox("Choisir un échantillon :", sample_images, format_func=lambda p: p.name)

        target_image = None
        if uploaded_file is not None:
            target_image = Image.open(uploaded_file).convert("RGB")
        elif selected_sample is not None:
            target_image = Image.open(selected_sample).convert("RGB")

        if target_image is not None:
            st.image(target_image, caption="Photo soumise", use_column_width=True)

    with col_result:
        if target_image is not None:
            with st.spinner("Analyse de la netteté et inférence en cours..."):
                result = safe_predict(
                    model=model,
                    image=target_image,
                    labels=labels_map,
                    transform=eval_transform,
                    device=device,
                    min_confidence=min_conf,
                    min_sharpness=min_sharp,
                    use_tta=use_tta,
                )

            # Métriques de qualité
            q_col1, q_col2, q_col3 = st.columns(3)
            q_col1.metric("Netteté", f"{result.quality['sharpness']:.1f}", delta="OK" if result.quality["is_sharp"] else "Floue", delta_color="normal" if result.quality["is_sharp"] else "inverse")
            q_col2.metric("Luminosité", f"{result.quality['brightness']:.1f}/255")
            q_col3.metric("Stabilité TTA", f"{result.stability_score:.0%}")

            # Badge de décision
            st.markdown(f"### Statut du diagnostic :")
            if result.status == "HIGH_CONFIDENCE":
                st.markdown(f'<span class="badge-high">{result.badge_label}</span>', unsafe_allow_html=True)
            elif result.status == "MODERATE_CONFIDENCE":
                st.markdown(f'<span class="badge-med">{result.badge_label}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-rej">{result.badge_label}</span>', unsafe_allow_html=True)

            st.info(result.message)

            if result.status != "REJECTED":
                st.markdown(f"**Diagnostic proposé :** `{result.top_label}` ({result.top_confidence:.1%})")

                # Probabilités
                st.write("**Top prédictions :**")
                for lbl, score in result.top_predictions:
                    st.progress(score, text=f"{lbl} : {score:.1%}")

                # Explicabilité Grad-CAM
                st.markdown("---")
                st.subheader("🔍 Explicabilité Visuelle (Grad-CAM)")
                alpha_slider = st.slider("Opacité de la carte thermique sur la feuille :", 0.0, 1.0, 0.45, 0.05)
                cmap_choice = st.selectbox("Palette de chaleur :", ["jet", "viridis", "magma", "plasma"])

                tensor = eval_transform(target_image).unsqueeze(0).to(device)
                cam = GradCAM(model, architecture=checkpoint["architecture"])
                top_class_id = int([k for k, v in labels_map.items() if v == result.top_label][0])
                heatmap = cam.generate_heatmap(tensor, target_class=top_class_id)
                cam.remove_hooks()

                overlay_img, _ = overlay_heatmap_on_image(target_image, heatmap, alpha=alpha_slider, colormap_name=cmap_choice)
                st.image(overlay_img, caption=f"Zone d'infection ciblée par l'IA pour '{result.top_label}'", use_column_width=True)
        else:
            st.write("👈 Veuillez charger une photo de feuille pour lancer le diagnostic sécurisé.")

# -------------------------------------------------------------
# ONGLET 2 : MÉTÉO & RISQUES FONGIQUES
# -------------------------------------------------------------
with tab2:
    st.subheader("Suivi Agroclimatique & Risques en temps réel")
    m_col1, m_col2 = st.columns([1, 2])

    with m_col1:
        crop_name = st.selectbox("Culture suivie :", list(CROP_RULES.keys()), format_func=lambda c: {"maize": "Maïs", "tomato": "Tomate", "cassava": "Manioc"}.get(c, c))
        city_presets = {
            "Cotonou (Bénin)": (6.37, 2.43),
            "Abidjan (Côte d'Ivoire)": (5.36, -4.01),
            "Accra (Ghana)": (5.60, -0.19),
            "Lomé (Togo)": (6.13, 1.22),
            "Dakar (Sénégal)": (14.69, -17.44),
            "Personnalisé": (0.0, 0.0),
        }
        selected_city = st.selectbox("Localisation géographique :", list(city_presets.keys()))
        if selected_city == "Personnalisé":
            lat = st.number_input("Latitude :", value=6.37, format="%.4f")
            lon = st.number_input("Longitude :", value=2.43, format="%.4f")
        else:
            lat, lon = city_presets[selected_city]

        btn_refresh_weather = st.button("📡 Actualiser la météo (Open-Meteo)")

    with m_col2:
        if btn_refresh_weather or "weather_data" not in st.session_state:
            try:
                with st.spinner("Récupération des prévisions Open-Meteo..."):
                    raw_weather = fetch_open_meteo(lat, lon)
                    st.session_state["weather_data"] = summarize_weather(raw_weather)
            except Exception as e:
                st.error(f"Erreur lors de la récupération météo : {e}")

        if "weather_data" in st.session_state:
            w_sum = st.session_state["weather_data"]
            advice = generate_climate_advice(crop_name, w_sum)

            w1, w2, w3 = st.columns(3)
            w1.metric("Température 7j", f"{w_sum['avg_temperature_7d_c']:.1f} °C")
            w2.metric("Humidité relative", f"{w_sum['current_humidity_pct']} %")
            w3.metric("Pluie prévue (7j)", f"{w_sum['rain_7d_mm']:.1f} mm")

            st.markdown("#### Alertes & Recommandations Agronomiques :")
            if advice["alerts"]:
                for alert in advice["alerts"]:
                    st.warning(f"⚠️ {alert}")
            else:
                st.success("✅ Aucune alerte critique sur les 7 prochains jours.")

            for rec in advice["recommendations"]:
                st.info(f"💡 {rec}")

# -------------------------------------------------------------
# ONGLET 3 : SIMULATEUR DE RENDEMENT
# -------------------------------------------------------------
with tab3:
    st.subheader("Projection de Rendement & Pertes Évitables")
    y_col1, y_col2 = st.columns([1, 1])

    with y_col1:
        y_crop = st.selectbox("Culture pour l'estimation :", ["maize", "tomato", "cassava"], format_func=lambda c: {"maize": "Maïs", "tomato": "Tomate", "cassava": "Manioc"}.get(c, c), key="yield_crop")
        soil_factor = st.slider("Facteur de fertilité du sol :", 0.5, 1.2, 0.95, 0.05)
        climate_factor = st.slider("Facteur climatique (stress hydrique/chaleur) :", 0.5, 1.0, 0.88, 0.02)
        health_status = st.selectbox("État sanitaire observé :", ["Saine (Healthy)", "Maladie légère", "Maladie modérée", "Attaque sévère"])
        severity_map = {"Saine (Healthy)": "healthy", "Maladie légère": "low", "Maladie modérée": "medium", "Attaque sévère": "high"}
        health_factor = health_factor_from_diagnosis(health_status, confidence=0.90, severity=severity_map[health_status])

    with y_col2:
        yield_res = estimate_yield(y_crop, climate_factor=climate_factor, health_factor=health_factor, soil_factor=soil_factor)
        st.markdown("#### Résultats de simulation :")
        res1, res2 = st.columns(2)
        res1.metric("Rendement potentiel", f"{yield_res['potential_t_per_ha']} t/ha")
        res2.metric("Rendement estimé", f"{yield_res['estimated_t_per_ha']} t/ha", delta=f"{yield_res['estimated_t_per_ha'] - yield_res['potential_t_per_ha']:.2f} t/ha")

        st.info(f"Fourchette de sécurité : entre **{yield_res['min_t_per_ha']}** et **{yield_res['max_t_per_ha']}** t/ha.")
