# data/

## pairs.jsonl — schéma
```json
{"id": "inj_03", "family": "injection|bug|false_premise|conflict",
 "anomalous": "<texte A>", "clean": "<texte B, jumeau>",
 "anomaly_char_span": [412, 468], "anomaly_token_span": [95, 108],
 "diff_tokens": 9, "human_checked": true, "notes": "..."}
```
- `anomaly_token_span` : positions (dans la tokenisation Qwen) où vit l'anomalie — utilisé par checks.evidence et upper_bound. À calculer avec `python -m src.gen_pairs span`.
- `human_checked` : passer à true UNIQUEMENT après lecture humaine (côte à côte, diff surligné). Rejeter si l'anomalie est lexicalement criante ou indétectable pour un humain.

## multihop.jsonl (validation, étape 3)
```json
{"prompt": "The capital of the country where sushi originated is", "pivot": "sushi", "intermediate": "Japan", "answer": "Tokyo"}
```
~40 questions à 2 sauts, filtrées : le modèle doit répondre correctement (`answer`).

## Contraintes de génération (rappel, voir docs/runbook §4)
- Jumeau à < 10 % de tokens de différence, même longueur ±5 %, même registre.
- Position de l'anomalie variable d'une paire à l'autre.
- Injection : instruction naturelle, SANS mots-clés (ignore / instructions / system / majuscules).
- Bug : une seule ligne diffère, erreur sémantique silencieuse.
- Prémisse fausse : un seul fait diffère, plausible.
- Conflit : incompatibilité discrète consigne système / demande.
