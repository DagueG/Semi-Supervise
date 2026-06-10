# BrainScan AI — Phase 1 R&D

Exploration analytique et apprentissage **semi-supervisé** pour la détection de tumeurs cérébrales sur IRM, dans un contexte où la quasi-totalité des images n'est pas étiquetée.

Projet réalisé pour CurelyticsIA (mission fictive). Rôle : Data Scientist junior, Computer Vision.

## Contexte

L'annotation d'IRM par des radiologues experts est rare et coûteuse. L'objectif est d'exploiter un large volume d'images **non étiquetées** pour amplifier la valeur d'un petit ensemble d'images expertisées, et d'évaluer si l'approche est transposable à grande échelle (4 M d'images pour 5 000 €).

## Données

1 506 IRM cérébrales, 512×512, en niveaux de gris (stockées en RGB avec R=G=B).

```
data/
├── avec_labels/
│   ├── cancer/   (50 images)
│   └── normal/   (50 images)
└── sans_label/   (1 406 images)
```

Le jeu « fortement » labellisé (100 vraies étiquettes) et le jeu « faiblement » labellisé (pseudo-étiquettes) sont gardés **strictement séparés** : les vrais labels ne servent qu'à évaluer, jamais à entraîner sur le jeu faible.

## Structure du dépôt

```
brainscan-ai/
├── data/                          # dataset — NON versionné
├── notebooks/
│   ├── 01_exploration.ipynb       # EDA : structure, résolution, équilibre, exemples
│   ├── 02_feature_extraction.ipynb# embeddings ResNet50 gelé (2048-d) -> features.npy
│   ├── 03_clustering.ipynb        # PCA + K-Means/DBSCAN, ARI, génération des labels faibles
│   └── 04_semi_supervised.ipynb   # CNN supervisé vs semi + label propagation
├── outputs/                       # artefacts générés — NON versionné
│   ├── features.npy               #   embeddings (1506, 2048)
│   ├── inventory.csv              #   table maîtresse path / split / label
│   ├── strong_labeled.csv         #   100 vrais labels
│   └── weak_labeled.csv           #   1406 pseudo-labels (clustering)
├── BrainScanAI_Phase1.pptx        # support de présentation
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Placer le dataset décompressé dans `data/`, puis exécuter les notebooks dans l'ordre `01 → 04`. Le notebook `02` produit `features.npy` (réutilisé par `03` et `04`), donc à lancer une fois avant les suivants. Conçu pour tourner sur **CPU**.

## Démarche

1. **Exploration** — vérification de la résolution, des canaux, de l'équilibre des classes (50/50 sur les labellisées).
2. **Extraction de features** — ResNet50 pré-entraîné, couches **gelées**, tête de classification retirée → 1 vecteur de 2 048 dimensions par image, extrait une seule fois.
3. **Clustering & labels faibles** — standardisation + PCA(50) + K-Means (k=2). Évaluation par score **ARI** sur les 100 images expertisées. Les clusters servent de pseudo-étiquettes pour les 1 406 images non labellisées.
4. **Comparaison de modèles** — un CNN entraîné de façon supervisée (vrais labels seuls) puis semi-supervisée (pseudo-labels → affinage), et une **label propagation** appliquée directement sur les embeddings ResNet. Métrique prioritaire : **rappel sur la classe Cancer** (un faux négatif = tumeur manquée).

## Résultats clés

Clustering : ARI = 0,124, accuracy 0,68 sur les labellisées. Le signal tumoral existe (un cluster pur à 91 % en cancer) mais les features ImageNet sont trop génériques pour une séparation nette → pseudo-labels bruités.

Comparaison sur un test de 30 IRM jamais vues (cancer = classe positive) :

| Approche | Rappel cancer | F1 cancer | Accuracy |
|---|---|---|---|
| Supervisé (CNN appris de zéro) | 0,76 | 0,71 | 0,69 |
| Clustering → CNN (semi) | 0,72 | 0,73 | 0,74 |
| **Label propagation** | 0,73 | **0,85** | **0,87** |
| Label propagation (5 découpages) | **0,87** | **0,91** | **0,91** |

**Conclusion.** Le semi-supervisé fonctionne — à condition d'opérer dans un bon espace de features. Les CNN appris de zéro plafonnent (trop peu d'images pour apprendre leurs propres features) ; la label propagation, qui réutilise les embeddings ResNet et propage les vrais labels dans le graphe de similarité, domine nettement. Le verrou n'était pas l'idée semi-supervisée, mais le véhicule.

## Recommandations (passage à l'échelle)

Le coût visé (5 000 € / 4 M images = 0,00125 €/image) est tenable côté compute : la contrainte est la **qualité des features**, pas le budget. Pistes prioritaires :

- Remplacer les features ImageNet par un encodeur **auto-supervisé** (SimCLR/DINO) entraîné sur les images non labellisées, ou un backbone médical.
- **Active learning** : dépenser l'annotation experte sur les images les plus incertaines.
- **Humain dans la boucle** sur les cas flagués « cancer » ; infra GPU et jeu de test conséquent (≫ 30) pour des mesures fiables.

## Limites

Exploration, non déployable cliniquement : test de petite taille (30 images, forte variance), rappel cancer ≈ 0,87 insuffisant pour un usage médical, label propagation transductive (avantage de protocole vs les CNN). Écart constaté avec la documentation : 1 506 images au lieu de 1 500 annoncées.