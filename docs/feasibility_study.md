# Étude de faisabilité - IA de diagnostic phytosanitaire, conseils climatiques et rendement

## 1. Modèle IA de diagnostic par photo

### Objectif

Entrée: photo RGB d'une feuille.  
Sortie: classe de maladie/ravageur ou état sain, plus score de confiance.

Dataset de départ: PlantVillage via `load_dataset("mohanty/PlantVillage", "color")`. Ce dataset est utile pour prototyper, mais il contient surtout des photos prises dans des conditions contrôlées. Il faudra donc vérifier la performance sur des photos réelles d'Afrique de l'Ouest avant usage opérationnel.

### Datasets terrain à ajouter à l'étude

PlantVillage reste utile pour démarrer vite, mais il ne doit pas être le seul jeu de données de référence. Deux sources terrain sont plus proches du contexte visé:

| Dataset | Culture | Zone | Type d'annotation | Intérêt pour le projet | Limite principale |
|---|---|---|---|---|---|
| iCassava / Makerere | Manioc | Ouganda | Label image-level, une classe par image | Photos collectées en conditions réelles avec smartphone, chez des agriculteurs; meilleur test de robustesse terrain que PlantVillage | Ne modélise pas plusieurs maladies ou niveaux de sévérité simultanés sur une même feuille |
| KaraAgro AI Maize | Maïs | Ghana | Bounding boxes + classes maladie/dégât | Très pertinent pour l'Afrique de l'Ouest; permet de localiser les symptômes, pas seulement classifier l'image entière | Demande un modèle de détection type YOLO/Faster R-CNN ou une conversion vers classification |
| KaraAgro AI Cocoa | Cacao | Ghana | Bounding boxes sur feuilles, cabosses et tiges | Ajoute une culture importante régionalement et des annotations fines pour maladie/état sain | Plus complexe à intégrer si le prototype initial reste limité aux feuilles |

Conséquence méthodologique: utiliser PlantVillage pour le premier prototype, puis évaluer et fine-tuner avec iCassava pour le manioc et KaraAgro pour le maïs/cacao. Pour KaraAgro, deux pistes sont possibles: entraîner un détecteur d'objets pour localiser les zones atteintes, ou recadrer les bounding boxes pour créer un dataset de classification compatible avec le pipeline actuel.

### Deux architectures proposées

| Option | Principe | Avantages | Limites | Usage recommandé |
|---|---|---|---|---|
| CNN simple from scratch | Réseau convolutif entraîné uniquement sur PlantVillage | Très pédagogique, architecture lisible, pas de dépendance forte à un modèle pré-entraîné | Plus lent, moins robuste avec peu de données, risque de surapprentissage | Prototype éducatif, mémoire académique |
| Transfer learning MobileNetV2/ResNet18 | Modèle pré-entraîné ImageNet, dernière couche remplacée | Plus rapide, meilleure précision avec peu de données, meilleur point de départ terrain | Demande une explication du pré-entraînement, dépend de poids externes | Prototype sérieux et base de production |

### Prétraitement

| Étape | Méthode | Rôle |
|---|---|---|
| Redimensionnement | 224 x 224 px | Entrée standard pour ResNet/MobileNet |
| Normalisation | Moyenne/écart-type ImageNet | Compatible avec modèles pré-entraînés |
| Rotation | ±25° | Simule orientation variable des feuilles |
| Flip horizontal/vertical | Probabiliste | Réduit la sensibilité à la prise de vue |
| Zoom/crop | `RandomResizedCrop` | Simule distance caméra variable |
| Variation couleur | brightness/contrast/saturation faibles | Simule éclairage modéré |

### Précision attendue

Hypothèses à vérifier: les valeurs ci-dessous supposent des classes bien représentées, peu de bruit d'étiquetage, et un test sur images proches de PlantVillage. Sur terrain réel, la précision peut chuter fortement.

| Taille d'entraînement | Classes | CNN scratch accuracy attendue | Transfer learning accuracy attendue |
|---:|---:|---:|---:|
| 5 000 images | 10-15 | 75-85 % | 85-93 % |
| 20 000 images | 20-30 | 85-92 % | 92-97 % |
| 50 000+ images | 30+ | 90-95 % | 95-98 %+ |

Formule simple pour dossier d'étude:

`accuracy terrain estimée = accuracy PlantVillage - pénalité domaine`

Exemple: `95 % - 15 points = 80 %` si les photos terrain ont éclairage, arrière-plan, variétés et maladies différentes.

### Matériel, temps et coût d'entraînement

| Configuration | CNN scratch | ResNet18/MobileNetV2 fine-tuning | Coût indicatif |
|---|---:|---:|---:|
| CPU local récent | 6-20 h | 3-10 h | 0 $, hors électricité |
| Google Colab gratuit GPU si disponible | 1-4 h | 30-90 min | 0 $, disponibilité non garantie |
| GPU cloud T4/A10 | 1-3 h | 20-60 min | environ 0,20-1,50 $/h selon fournisseur |

Calcul exemple:

`coût entraînement = durée GPU en heures × prix horaire`

Si ResNet18 prend 0,75 h sur GPU à 0,60 $/h: `0,75 × 0,60 = 0,45 $`.

### Métriques à produire

Le script `train_plant_disease.py` sauvegarde:

| Métrique | Utilité |
|---|---|
| Accuracy globale | Vue rapide de performance |
| Précision par classe | Parmi les prédictions d'une classe, part correcte |
| Rappel par classe | Parmi les vrais cas d'une classe, part détectée |
| Matrice de confusion | Identifie les maladies confondues |

### Limites connues

| Risque | Effet | Réduction possible |
|---|---|---|
| Photos PlantVillage trop propres | Mauvaise généralisation terrain | Ajouter iCassava et KaraAgro, puis tester sur photos locales réelles |
| Maladies non présentes dans le dataset | Le modèle force une mauvaise classe connue | Ajouter classe "inconnu"/rejet par seuil de confiance |
| Cultures locales absentes | Diagnostic impossible ou trompeur | Extension dataset maïs, manioc, cacao et tomate locaux |
| Éclairage, flou, arrière-plan | Baisse de confiance | Contrôle qualité image et augmentation plus réaliste |
| Variétés locales différentes | Symptômes visuels variables | Fine-tuning avec données locales |
| Annotation bounding-box non utilisée | Perte d'information sur la localisation des symptômes | Prévoir une phase détection d'objets ou recadrage automatique des zones annotées |

## 2. Étude comparative des approches

| Critère | CNN scratch | MobileNetV2 | ResNet18 |
|---|---:|---:|---:|
| Précision attendue PlantVillage | 85-92 % | 93-97 % | 94-98 % |
| Robustesse avec peu de données | Moyenne | Bonne | Bonne |
| Temps GPU indicatif | 1-4 h | 20-60 min | 30-90 min |
| Taille modèle | Faible à moyenne | Faible | Moyenne |
| Complexité code | Faible | Moyenne | Moyenne |
| Recommandation | Pédagogie | Mobile/embarqué | Prototype principal |

## 3. Impact potentiel

Hypothèse demandée: maladies et ravageurs peuvent causer jusqu'à 30 % de pertes annuelles. Ce chiffre est un ordre de grandeur à vérifier avec données locales par culture, zone et saison.

Formule:

`récolte préservée (%) = perte évitable maximale × taux de détection utile × taux d'action correcte`

Exemple sur 1 ha de maïs:

| Paramètre | Hypothèse |
|---|---:|
| Rendement potentiel | 4 t/ha |
| Prix de vente | 220 $/t |
| Valeur brute | 880 $/ha |
| Perte maladie/ravageur maximale | 30 % |
| Part évitable par détection précoce | 40 % |
| Adoption/action correcte | 70 % |

Calcul:

`gain récolte = 4 × 30 % × 40 % × 70 % = 0,336 t/ha`

`gain économique = 0,336 × 220 = 73,92 $/ha`

## 4. Module de conseils climatiques

### Pourquoi un système à règles plutôt qu'un deep learning

Un modèle deep learning climatique from scratch demanderait beaucoup de données locales fiables: météo historique, pratiques agricoles, dates de semis, maladies observées, rendements, sols, interventions. Au démarrage, ces données sont rarement disponibles. Un système à règles agronomiques piloté par la météo est plus transparent, moins coûteux, plus facile à corriger par des experts locaux, et peut ensuite être enrichi par un LLM pour reformuler les conseils en langage naturel.

### APIs météo utilisables

| API | Usage | Coût/limite vérifiée | Remarque |
|---|---|---|---|
| Open-Meteo | Prévisions actuelles et 7 jours par GPS | Gratuit sans clé pour prototypage non commercial, limite indiquée de 10 000 appels/jour et 300 000/mois; offres commerciales pour volume | Simple pour prototype |
| NASA POWER | Données agroclimatiques historiques par GPS | Gratuit, données quotidiennes disponibles en JSON/CSV; limite de 20 paramètres par requête point | Très utile pour historique/climatologie |
| OpenWeatherMap | Météo actuelle/prévisions | Offre gratuite limitée et offres payantes | Utile si besoin d'écosystème commercial |

Sources consultées le 13 septembre 2026: Open-Meteo pricing et page principale, NASA POWER API documentation, iCassava/Makerere, KaraAgro AI Maize, KaraAgro AI Cocoa.

### Exemple de règles agronomiques

À valider avec un agronome local.

| Culture | Température optimale | Risque fongique simplifié | Besoin eau/semaine | Mois de semis indicatifs |
|---|---:|---:|---:|---|
| Maïs | 20-32 °C | humidité ≥80 % et 20-30 °C | 35 mm | avril-juillet |
| Tomate | 18-30 °C | humidité ≥75 % et 18-28 °C | 30 mm | sept.-févr. selon zone |
| Manioc | 24-32 °C | humidité ≥80 % et 22-30 °C | 25 mm | avril-juillet |

### Fiabilité des prévisions

| Horizon | Fiabilité qualitative | Impact conseil |
|---|---|---|
| 1-3 jours | Bonne | Alertes irrigation/pluie généralement utiles |
| 4-5 jours | Moyenne | Conseils à formuler avec prudence |
| 6-7 jours | Plus incertaine | À utiliser comme tendance, pas comme décision unique |

Méthode de pondération:

`score conseil = score règle × coefficient horizon`

Exemple: coefficient 0,9 à 3 jours, 0,7 à 5 jours, 0,5 à 7 jours.

### Coût à l'échelle

Hypothèse: 5 000 utilisateurs, 1 requête météo/jour.

`requêtes mensuelles = 5 000 × 30 = 150 000`

Ce volume reste sous 300 000 requêtes/mois pour l'offre gratuite Open-Meteo non commerciale. Pour usage commercial, prévoir une offre payante ou cache serveur.

### Valeur ajoutée climatique

Formule:

`gain climatique = valeur récolte × perte liée aux mauvais timings × part évitable × adoption`

Exemple maïs:

| Paramètre | Hypothèse |
|---|---:|
| Valeur brute | 880 $/ha |
| Perte liée à mauvais timing/irrigation | 15 % |
| Part évitable par conseil | 30 % |
| Adoption | 70 % |

`gain = 880 × 15 % × 30 % × 70 % = 27,72 $/ha`

## 5. Estimation de rendement

### Pourquoi un modèle agronomique simplifié au départ

Sans historique local de rendement, entraîner un modèle ML direct serait fragile: il apprendrait surtout les biais du petit jeu de données disponible. Un modèle à facteurs de stress est plus explicable: on part d'un rendement potentiel, puis on applique des réductions liées au climat, à la santé de la plante et éventuellement au sol.

Formule proposée:

`rendement estimé = rendement potentiel × facteur climat × facteur sanitaire × facteur sol`

Fourchette:

`min = estimation centrale × (1 - incertitude)`

`max = estimation centrale × (1 + incertitude)`

Exemple maïs:

| Paramètre | Valeur |
|---|---:|
| Rendement potentiel | 4 t/ha |
| Facteur climat | 0,88 |
| Facteur sanitaire | 0,82 |
| Facteur sol | 0,95 |
| Incertitude | ±25 % |

Calcul:

`central = 4 × 0,88 × 0,82 × 0,95 = 2,74 t/ha`

`min = 2,74 × 0,75 = 2,06 t/ha`

`max = 2,74 × 1,25 = 3,43 t/ha`

### Références initiales ajustables

Ces valeurs sont des placeholders réalistes à remplacer par FAO, statistiques nationales ou données coopératives locales.

| Culture | Rendement potentiel de référence | Incertitude initiale |
|---|---:|---:|
| Maïs | 4 t/ha | ±25 % |
| Tomate | 25 t/ha | ±30 % |
| Manioc | 18 t/ha | ±30 % |

### Marge d'erreur

| Modèle | Données nécessaires | Erreur initiale plausible |
|---|---|---:|
| Facteurs de stress simplifiés | Rendements potentiels + règles agronomiques | ±25-40 % |
| Random Forest/XGBoost local | Historique rendement + météo + maladies + sol + pratiques | ±10-25 % si données suffisantes |

### Valeur économique de l'estimation

Formule:

`bénéfice = valeur récolte × amélioration de décision commerciale`

Exemple:

| Paramètre | Hypothèse |
|---|---:|
| Valeur brute maïs | 880 $/ha |
| Gain par meilleure planification ventes/stockage | 3 % |

`bénéfice = 880 × 3 % = 26,40 $/ha`

### Données à collecter en priorité

| Donnée | Pourquoi |
|---|---|
| Rendements réels par parcelle et saison | Cible du futur modèle ML |
| Coordonnées GPS et dates de semis/récolte | Relier production et météo |
| Photos datées et diagnostics confirmés | Adapter le modèle santé au terrain |
| Bounding boxes de symptômes | Entraîner ou évaluer la localisation des zones atteintes |
| Traitements appliqués | Expliquer les pertes ou récupérations |
| Type de sol, fertilisation, irrigation | Réduire l'incertitude |
| Variété cultivée | Symptômes et rendement dépendent de la variété |

## 6. Hypothèses à vérifier localement

| Hypothèse | À vérifier avec |
|---|---|
| Prix par tonne | Marchés locaux/cooperatives |
| Rendements potentiels | FAO, ministères, instituts agronomiques |
| Calendriers de semis | Services agricoles locaux |
| Seuils maladie/météo | Agronomes et essais terrain |
| Taux d'adoption des conseils | Enquêtes utilisateurs |
| Performance terrain du modèle image | Jeu test local indépendant |
| Gain de KaraAgro par rapport à PlantVillage | Test maïs/cacao séparé avec métriques classification et détection |
| Gain de iCassava par rapport à PlantVillage | Test manioc terrain avec validation agronomique locale |
