# [Titre — une ligne, factuel]  (≤ 600 mots, 2-3 figures — À ÉCRIRE À LA MAIN, SANS LLM)

## Le problème (≈ 5 lignes)
- Lenses validés par cas d'étude, jamais caractérisés comme détecteurs en aveugle.
- Text inversion : formalisée pour les activation oracles, jamais testée sur J/R-lens.
- Neel (review J-Lens, 6 juil) : [citation 1 : "concaténer les top-10 tokens... suffirait"] ; [citation 2 : "j'aimerais des données sur le taux de faux positifs"]. URLs.
- Réponse partielle à la question de S. Athena sous le post R-lens ("quelle est la bonne façon d'évaluer ces lenses").

## Takeaways (3-4 puces, un chiffre chacune)
- Sur 40 paires / 4 familles, scan [J/R] aveugle → juge LLM : AUC X [IC] (fausse alerte Y %), vs Z (prompt seul), W (reconstruction seule), V (permuté). Valeur ajoutée non expliquée par la text inversion : X−W.
- J vs R : ...
- Par famille : ...
- Méthodologique (H3) : ...

## Figures
- fig1_triplet.png — Δ1/Δ2/Δ3 par instrument (IC).   - fig2_auc_family.png.   - (fig3 exemple annoté)

## Limites (3 lignes)
Un modèle (Qwen3.6-27B) ; n=40 ; juge LLM (2 juges, κ=…) ; anomalies synthétiques ; couches sous-échantillonnées 1/4 ; top-10.

## Vérification (3 lignes, factuel)
J'ai lu 30 scans bruts (tableau en annexe), recalculé les AUC indépendamment, audité le harness pour les fuites (results/leak_check.csv), testé 2 juges × 2 prompts. L'agent a écrit le pipeline ; j'ai conçu chaque condition avant de coder.
