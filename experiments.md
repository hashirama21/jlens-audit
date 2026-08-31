# Journal d'expériences (append-only)

Format par entrée :
## YYYY-MM-DD HH:MM — <titre court>  [compté: oui/non, durée]
- Fait :
- Vérifié :
- Ce que je crois maintenant :
- Doute / ce qui pourrait être faux :
- Prochaine étape :

---
## 2026-08-15 — Session 0 : cadrage et squelette du repo  [compté: NON — préparation générale]
- Fait : choix du sujet A ; lecture doc de candidature, papier J-Lens (à finir), review de Neel, post R-lens (5 août) ; conception des 7 conditions AVANT tout code ; squelette complet du repo (src/, prompts, checks, metrics) ; 8 paires pilote rédigées ; 40 questions multihop pour la validation.
- Vérifié : le post R-lens ne contient ni audit en scan aveugle, ni FPR, ni contrôle d'inversion → créneau ouvert. R-lens sans avantage sur petits modèles → Qwen3.6-27B obligatoire → GPU 80 Go. Une paire pilote (bug_02) était identique en A/B, attrapée par un check automatique et corrigée.
- Ce que je crois maintenant : le résultat le plus probable est mixte par famille ; la valeur du projet tient aux contrôles (reconstruction, permutation), pas à l'AUC brute.
- Doute / ce qui pourrait être faux : le format des lenses (src/lens.py # ADAPTER) — à résoudre en lisant le README HF avant toute chose ; le juge peut détecter la famille plutôt que l'anomalie (check family) ; les prompts du juge sont en français et le contenu en anglais — à harmoniser en anglais si le juge s'en trouve gêné.
- Prochaine étape : pod 80 Go + download.sh ; lire lenses/README.md ; résoudre # ADAPTER ; smoke test ; puis étape 3.
---
## 2026-08-18 00:30 — Audit du code + refonte (bugs, DRY, anglais)  [compté: NON — fiabilisation du pipeline, aucun résultat de recherche]
- Fait :
  - Audit complet des 13 modules `src/` + prompts + env.
  - Nouveau `src/store.py` : couche IO unique (paires / scans / verdicts) → DRY.
  - 5 bugs critiques corrigés : (B1) `gen_pairs.generate` ne peut plus écraser des paires `human_checked` ; (B2) `load_pairs(validated_only=True)` filtre partout les paires rejetées/non validées ; (B3) `conditions.run` reprenable et sans doublon (done-set + dédup lecture) ; (B4) erreurs API ni cachées ni comptées dans les AUC ; (B5) `upper_bound` en leave-one-**pair**-out + scaler dans le pipeline (double fuite supprimée).
  - Contrôle H3 corrigé : `permute_positions` = une permutation de positions unique partagée entre couches (préserve la cohérence inter-couches). Bootstrap métriques **apparié par paire**. Générateur de corpus distinct des juges (temp 0.9, champ `generator` loggé).
  - Divers : clé de cache juge = modèle+température ; `find_pos` robuste multi-token ; gate leak-check dans `conditions` ; `checks.leak` mesure la séquence contiguë max ; tokenizer sans le 27B (`load_tok`) ; imports lourds (seaborn/openai/matplotlib) rendus lazy.
  - **Harmonisation anglais** de tout le pipeline (code, docstrings, prompts juge, question de capacité, gabarits de génération) — résout le doute de la Session 0.
  - 13 tests unitaires GPU-free (`tests/test_pure.py`), verts en local (py 3.14, sans torch).
- Vérifié : `py_compile` tout `src/` + tests ; `pytest` 13/13 ; imports légers OK ; grep : aucune référence à l'ancienne API.
- Ce que je crois maintenant : pipeline cohérent et testé côté logique pure ; les `# ADAPTER` de `lens.py` restent ouverts (dépendent du README des lenses).
- Doute / ce qui pourrait être faux : rien n'a tourné sur GPU ni contre un vrai juge ; Δ2 confond « lens inversible » et « juge bon reconstructeur » (à ajouter §5.4) ; budget contexte réel = 12 couches (stride 4), à trancher (stride 6 vs top-5).
- Prochaine étape : pod + download ; lire `lenses/README.md` ; résoudre les `# ADAPTER` avec moi ; smoke test `sushi → Japan`.
---
