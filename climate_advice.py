import argparse
import json
from dataclasses import dataclass
from datetime import date
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class CropRules:
    optimal_temp_min: float
    optimal_temp_max: float
    fungal_temp_min: float
    fungal_temp_max: float
    fungal_humidity_min: float
    weekly_water_need_mm: float
    sowing_months: List[int]


CROP_RULES: Dict[str, CropRules] = {
    "maize": CropRules(20, 32, 20, 30, 80, 35, [4, 5, 6, 7]),
    "tomato": CropRules(18, 30, 18, 28, 75, 30, [9, 10, 11, 12, 1, 2]),
    "cassava": CropRules(24, 32, 22, 30, 80, 25, [4, 5, 6, 7]),
}


def fetch_open_meteo(latitude: float, longitude: float) -> dict:
    """Recupere les donnees actuelles et la prevision 7 jours via Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 7,
        "timezone": "auto",
    }
    with urlopen(f"{url}?{urlencode(params)}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_weather(weather: dict) -> dict:
    current = weather["current"]
    daily = weather["daily"]
    avg_temp_7d = sum((hi + lo) / 2 for hi, lo in zip(daily["temperature_2m_max"], daily["temperature_2m_min"])) / 7
    rain_7d = sum(daily["precipitation_sum"])
    return {
        "current_temperature_c": current["temperature_2m"],
        "current_humidity_pct": current["relative_humidity_2m"],
        "current_precipitation_mm": current["precipitation"],
        "avg_temperature_7d_c": avg_temp_7d,
        "rain_7d_mm": rain_7d,
    }


def generate_climate_advice(crop: str, summary: dict, month: int | None = None) -> dict:
    if crop not in CROP_RULES:
        raise ValueError(f"Culture inconnue: {crop}. Choisir parmi {', '.join(CROP_RULES)}")

    rules = CROP_RULES[crop]
    month = month or date.today().month
    alerts = []
    recommendations = []

    temp = summary["avg_temperature_7d_c"]
    humidity = summary["current_humidity_pct"]
    rain = summary["rain_7d_mm"]

    temp_ok = rules.optimal_temp_min <= temp <= rules.optimal_temp_max
    if not temp_ok:
        alerts.append(
            f"Temperature moyenne prevue ({temp:.1f} C) hors plage optimale "
            f"({rules.optimal_temp_min}-{rules.optimal_temp_max} C)."
        )

    fungal_risk = rules.fungal_temp_min <= temp <= rules.fungal_temp_max and humidity >= rules.fungal_humidity_min
    if fungal_risk:
        alerts.append("Risque accru de maladie fongique: humidite elevee et temperature favorable.")
        recommendations.append("Inspecter les feuilles, reduire l'exces d'humidite si possible et agir selon les recommandations locales.")

    if rain < rules.weekly_water_need_mm * 0.5:
        alerts.append(f"Secheresse possible: pluie prevue {rain:.1f} mm sur 7 jours.")
        recommendations.append("Prevoir une irrigation complementaire si disponible.")
    elif rain > rules.weekly_water_need_mm * 1.8:
        alerts.append(f"Exces d'eau possible: pluie prevue {rain:.1f} mm sur 7 jours.")
        recommendations.append("Verifier le drainage et surveiller les symptomes fongiques.")

    if month in rules.sowing_months:
        recommendations.append("Fenetre de semis potentiellement favorable, a confirmer avec le calendrier local.")
    else:
        recommendations.append("Mois moins favorable au semis selon cette regle simple.")

    return {
        "crop": crop,
        "weather_summary": summary,
        "alerts": alerts,
        "recommendations": recommendations,
        "climate_factor_for_yield": estimate_climate_factor(crop, summary),
    }


def estimate_climate_factor(crop: str, summary: dict) -> float:
    rules = CROP_RULES[crop]
    temp = summary["avg_temperature_7d_c"]
    rain = summary["rain_7d_mm"]

    temp_penalty = 0.0
    if temp < rules.optimal_temp_min:
        temp_penalty = min(0.25, (rules.optimal_temp_min - temp) * 0.03)
    elif temp > rules.optimal_temp_max:
        temp_penalty = min(0.25, (temp - rules.optimal_temp_max) * 0.03)

    water_ratio = rain / max(rules.weekly_water_need_mm, 1)
    water_penalty = min(0.30, abs(1 - water_ratio) * 0.20)
    return round(max(0.45, 1 - temp_penalty - water_penalty), 2)


def combine_photo_and_climate(photo_diagnosis: dict, climate_result: dict) -> str:
    disease = photo_diagnosis.get("label", "inconnu")
    confidence = photo_diagnosis.get("confidence", 0.0)
    alerts = " ".join(climate_result["alerts"]) or "Pas d'alerte climatique majeure."
    recs = " ".join(climate_result["recommendations"])
    return (
        f"Diagnostic photo: {disease} (confiance {confidence:.0%}). "
        f"Conditions climatiques: {alerts} "
        f"Recommandation finale: {recs}"
    )


def main():
    parser = argparse.ArgumentParser(description="Conseils agronomiques simples pilotes par la meteo.")
    parser.add_argument("--crop", choices=list(CROP_RULES), required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    weather = fetch_open_meteo(args.lat, args.lon)
    summary = summarize_weather(weather)
    result = generate_climate_advice(args.crop, summary)
    print(result)


if __name__ == "__main__":
    main()
