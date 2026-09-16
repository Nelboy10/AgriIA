import argparse
from dataclasses import dataclass
from typing import Dict


@dataclass
class YieldReference:
    potential_t_per_ha: float
    uncertainty_pct: float


YIELD_REFERENCES: Dict[str, YieldReference] = {
    "maize": YieldReference(potential_t_per_ha=4.0, uncertainty_pct=0.25),
    "tomato": YieldReference(potential_t_per_ha=25.0, uncertainty_pct=0.30),
    "cassava": YieldReference(potential_t_per_ha=18.0, uncertainty_pct=0.30),
}


DISEASE_SEVERITY_FACTORS = {
    "healthy": 1.00,
    "low": 0.90,
    "medium": 0.75,
    "high": 0.55,
}


def health_factor_from_diagnosis(label: str, confidence: float, severity: str | None = None) -> float:
    """Convertit un diagnostic image en facteur sanitaire simplifie."""
    lower_label = label.lower()
    if "healthy" in lower_label or "saine" in lower_label:
        return 1.0
    if severity in DISEASE_SEVERITY_FACTORS:
        base = DISEASE_SEVERITY_FACTORS[severity]
    else:
        base = 0.80
    confidence_adjustment = 1 - confidence * (1 - base)
    return round(max(0.45, confidence_adjustment), 2)


def estimate_yield(crop: str, climate_factor: float, health_factor: float, soil_factor: float = 1.0) -> dict:
    if crop not in YIELD_REFERENCES:
        raise ValueError(f"Culture inconnue: {crop}. Choisir parmi {', '.join(YIELD_REFERENCES)}")
    ref = YIELD_REFERENCES[crop]
    central = ref.potential_t_per_ha * climate_factor * health_factor * soil_factor
    return {
        "crop": crop,
        "potential_t_per_ha": ref.potential_t_per_ha,
        "climate_factor": climate_factor,
        "health_factor": health_factor,
        "soil_factor": soil_factor,
        "estimated_t_per_ha": round(central, 2),
        "min_t_per_ha": round(central * (1 - ref.uncertainty_pct), 2),
        "max_t_per_ha": round(central * (1 + ref.uncertainty_pct), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Estimation simplifiee du rendement par facteurs de stress.")
    parser.add_argument("--crop", choices=list(YIELD_REFERENCES), required=True)
    parser.add_argument("--climate-factor", type=float, required=True)
    parser.add_argument("--health-factor", type=float, required=True)
    parser.add_argument("--soil-factor", type=float, default=1.0)
    args = parser.parse_args()

    print(estimate_yield(args.crop, args.climate_factor, args.health_factor, args.soil_factor))


if __name__ == "__main__":
    main()
