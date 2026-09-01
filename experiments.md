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
## 2026-08-31 — Résolution des `# ADAPTER` + 4 bugs bloquants (revue externe)  [compté: NON — fiabilisation, aucun résultat de recherche]
- Fait :
  - README HF (camilablank/workspace-lenses) lu et confirmé par WebFetch. Format tranché : UN `lens.pt` par lens (`{j-lens,r-lens}/lens.pt`), dict `['J','n_prompts','source_layers','d_model','provenance']` ; formule `softmax(W_U · norm(J_ℓ · h_ℓ))` → pas de biais, norm NON repliée (on garde `self.final_norm`), orientation `H @ J.T` correcte, ancre identité à `target_layer=62`.
  - **B1** `lens.py` : abandon du safetensors par couche ; `_stack(kind)` charge le `.pt` unique et mappe `source_layers → index` ; `_load_layer_map` renvoie `J[idx[L]]`, KeyError explicite si couche absente (skip_first=4).
  - **B2** `config.py` : archi réelle centralisée — `N_LAYERS=64`, `D_MODEL=5120`, `TARGET_LAYER=62`, `SKIP_FIRST=4` (le commentaire « ~48 couches / ~12 » était faux).
  - **B3** `load_model.layers()` : grille `range(0,64,stride)` filtrée par `source_layers`, avec `62` toujours inclus (ancre). Plus de KeyError à L0, la couche 62 n'est plus ratée.
  - **B4** `validate.identity_check()` : go/no-go binaire — à L62 j/r-lens doivent reproduire exactement le logit lens (top-5). Tourne AVANT smoke dans tous les modes.
  - Mineurs : `env/download.sh` → `hf` (ex-`huggingface-cli`), restreint à `j-lens/r-lens/*`+README (~7 Go au lieu de ~25) ; `load_model` `torch_dtype=`→`dtype=`.
  - Générateur de corpus déplacé vers une **3ᵉ famille** distincte des deux juges : `google/gemini-2.5-pro` (pas un Qwen = famille sous audit). IDs OpenRouter vérifiés (endpoint /models, 396 modèles) : juge A `claude-sonnet-4.5`→`4.6` ; juge B `gpt-5-mini` toujours valide.
- Vérifié : `pytest tests/test_pure.py` 13/13 ; parse AST des 4 modules touchés ; IDs présents dans la liste OpenRouter réelle (pas la synthèse WebFetch, jugée peu fiable).
- Ce que je crois maintenant : les `# ADAPTER` de `lens.py`/`load_model.py` sont résolus et alignés sur le README ; le pipeline est prêt pour le pod. Rien n'a encore tourné sur GPU.
- Doute / ce qui pourrait être faux : `identity_check` étant symétrique à I, il n'attrape pas une transposition PURE de J à L62 (I=Iᵀ) — vrai filet pour indexation/norm, pas pour l'orientation seule ; le décalage `hidden_states[L+1]` reste à confirmer au premier scan. Choix de scope (stride 8, top-5, juges réduits, drop upper_bound) NON appliqués : décisions à toi.
- Prochaine étape : pod + `download.sh` ; `python -m src.validate --smoke` (identity_check d'abord) ; si vert, `python -m src.checks leak` avant tout scan aveugle.
---
## 2026-08-31 (2) — 2ᵉ passe de revue : F1–F6 + orientation  [compté: NON — fiabilisation]
- Fait :
  - **F1 (décision : chat template partout)** : helper unique `load_model.to_input_ids` (honore `USE_CHAT_TEMPLATE`), utilisé par `get_resid` (scan), `capability.ask` et `gen_pairs.span` → positions alignées de bout en bout. Familles injection/conflict désormais lues dans leur cadrage conversationnel réel. `anomaly_token_span` recalculé dans le même cadrage (ordre : décision → `gen_pairs span` → scan).
  - **F2** : `identity_check` ne revendique plus l'orientation (I=Iᵀ l'en empêche) ; ajout d'`orientation_check` (recouvrement à la sous-ancre chargée la plus proche → attrape une transposition). Les deux câblés dans `validate.__main__` et le notebook.
  - **F3** : go/no-go (identity+orientation) remonté en tête d'étape 2 du notebook, avant le smoke.
  - **F4** : `layers()` retombe sur la grille naïve si les lenses ne sont pas téléchargées → la cellule SETUP ne plante plus.
  - **F5** : `flush()` ajouté sur la branche d'erreur de reconstruction (`conditions.py`).
  - **F6** : laissé suivi (règle `data/*.jsonl` du .gitignore commentée par l'humain) — décision ouverte.
- Vérifié : `pytest tests/test_pure.py` 13/13 ; parse AST des 5 modules touchés ; helper partagé unique (aucun `apply_chat_template` en double).
- Ce que je crois maintenant : entrée cohérente scan↔capacité↔span ; go/no-go réellement discriminant (ancre + orientation) et présent là où l'humain l'exécute.
- Doute / ce qui pourrait être faux : le scan inclut désormais les tokens spéciaux du template comme positions (bruit pour le juge, mais fidèle) ; `orientation_check` a un seuil arbitraire (`min_overlap=6/10`) à calibrer au premier run ; rien n'a tourné sur GPU.
- Prochaine étape : pod → `download.sh` → go/no-go → `gen_pairs generate/review/span/stats` → scan.
---
## 2026-09-01 — Audit post-F1 : régression `find_pos` + cohérence des positions  [compté: NON — fiabilisation]
- Fait :
  - **Régression F1 corrigée** : `find_pos` (validate.py) tokenisait le texte BRUT alors que `get_resid` renvoie désormais la séquence templatée → `H[L][pos]` décalé du préfixe template. `smoke`/`pass_at_k` lisaient la mauvaise position. `find_pos` opère maintenant sur `render_input(...)`, même cadrage que le scan. `identity_check`/`orientation_check` n'étaient pas touchés (position-agnostiques).
  - **DRY du cadrage** : `load_model.render_input` (chaîne exacte envoyée au modèle) devient la source unique ; `to_input_ids` et le nouveau `content_span` s'appuient dessus ; `add_special_tokens=False` sous template (le rendu porte déjà les tokens spéciaux, équivaut à `apply_chat_template(tokenize=True)`) → tokenisation identique entre entrée modèle et recherche de position.
  - **Tokens du template exclus du scan** : `scan_text` ne sérialise plus que `[lo, hi)` = les positions du contenu (via `content_span`), en gardant les indices ABSOLUS → `anomaly_token_span` et `checks.evidence` restent alignés, moins de bruit et de budget juge.
  - **Mineurs** : `layers()` fallback exclut désormais 0..3 (`SKIP_FIRST`) et inclut `TARGET_LAYER` ; tag `# ADAPTER` retiré d'`unembed_parts` (noms Qwen confirmés).
- Vérifié : `pytest tests/test_pure.py` 13/13 ; parse AST des 5 modules touchés. Non couvert hors GPU : `find_pos`/`content_span`/`scan_text` (dépendent du tokenizer Qwen) — à valider au premier run sur pod.
- Ce que je crois maintenant : positions cohérentes de bout en bout (scan ↔ span ↔ evidence ↔ validation) ; entrée modèle inchangée en ids (refactor équivalent).
- Doute / ce qui pourrait être faux : `content_span` suppose que `content` apparaît verbatim dans le rendu (fallback = plage complète sinon) ; `orientation_check` garde un seuil arbitraire à calibrer.
- Prochaine étape : pod → go/no-go → vérifier à la main `find_pos`/`content_span` sur une paire avant le scan complet.
---
