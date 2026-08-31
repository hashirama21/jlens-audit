# STATUS — ce qui est dans ce zip, ce qui ne peut pas y être, et pourquoi

| Élément | État | Explication |
|---|---|---|
| src/ (13 modules), prompts, docs, env | ✅ complet | syntaxe vérifiée |
| data/pairs_pilot.jsonl | ✅ 8 paires (2/famille), rédigées à la main | `human_checked=false` : à relire (`python -m src.gen_pairs review`) — c'est une exigence de Neel, pas une option |
| data/pairs.jsonl (40) | ⚙️ produit par `python -m src.gen_pairs generate` sur le pod | les 32 restantes sont générées via OpenRouter avec les gabarits, pilotes en tête ; puis review humaine intégrale, `span`, `stats` |
| data/multihop.jsonl | ✅ 40 items | filtrer sur le pod : garder ceux où le modèle répond `answer` correctement |
| notebooks/main.ipynb | ✅ créé | cellules SETUP / lenses / smoke / validation / capacité / test de temps |
| experiments.md | ✅ session 0 remplie | les sessions suivantes sont à écrire par vous, à chaque fin de séance |
| lenses/, modèle Qwen3.6-27B | ❌ impossible dans un zip | 54 Go + ~10 Go, téléchargés sur le pod par `env/download.sh` (~10-20 min) |
| scans/, results/, judge/outputs/ | ❌ vides par nature | ce sont les SORTIES des 20 heures ; les fabriquer serait exactement ce que Neel rejette. Ils se remplissent avec `scan`, `conditions`, `metrics`, `checks` |
| src/lens.py `# ADAPTER` | ✅ résolu | format lu dans le README HF (un `lens.pt` par lens, `J`+`source_layers`) ; go/no-go `identity_check` + `orientation_check` |
| config.py JUDGE_A/JUDGE_B, GENERATOR | ✅ fixés | juges Anthropic/OpenAI (IDs vérifiés OpenRouter) ; générateur 3ᵉ famille `gemini-2.5-pro` ; env-overridables |
| Framing entrée (F1) | ✅ chat template | `USE_CHAT_TEMPLATE=True` ; scan/capacité/span via `to_input_ids` ; relancer `gen_pairs span` si on bascule le flag |

Ordre des opérations sur le pod : `env/download.sh` → lire `lenses/README.md` → corriger `src/lens.py` → `python -m src.validate --smoke` → notebook cellules 3-6 → go/no-go → `gen_pairs generate/review/span/stats` → le compteur démarre.
