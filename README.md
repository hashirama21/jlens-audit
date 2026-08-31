# jlens-audit

Audit en aveugle des lens de lecture d'activations (J-Lens / R-lens / logit lens) sur Qwen3.6-27B :
un scan (toutes positions × couches) passé à un juge LLM détecte-t-il des anomalies dans le texte lu par le modèle,
et le signal survit-il aux contrôles de text inversion (H2) et d'apophénie (H3) ?

## Reproduction (10 lignes)
```bash
pip install -r env/requirements.txt
bash env/download.sh                      # modèle + lenses (Qwen3.6-27B, camilablank/workspace-lenses)
export OPENROUTER_API_KEY=...
python -m src.validate --smoke            # étape 2 : conformité 'sushi -> Japan' (après résolution des # ADAPTER)
python -m src.validate                    # étape 3 : pass@10 multihop (data/multihop.jsonl), J vs R -> results/, figs/
python -m src.gen_pairs generate          # étape 4a : corpus (modèle générateur ≠ juges)
python -m src.gen_pairs review            # étape 4b : validation humaine (human_checked) — NON délégable
python -m src.gen_pairs span              # étape 4c : anomaly_token_span (tokenizer Qwen)
python -m src.capability                  # étape 4d : le modèle voit-il les anomalies en clair ?
python -m src.scan                        # étape 5 : scans des 80 textes x 3 instruments -> scans/
python -m src.checks leak                 # étape 6 : audit de fuite (obligatoire avant les conditions)
python -m src.conditions                  # étape 7 : les 7 conditions x juges -> judge/outputs/ (reprenable)
python -m src.metrics                     # étape 9 : AUC/IC, triplet, figures -> results/, figs/
python -m src.checks sample --n 30        # étape 8 : tirage des 30 scans à lire à la main
```
Tests locaux (sans GPU) : `python -m pytest` — fonctions pures + dry-run des étapes 6-9 avec juge simulé.
Voir docs/ pour le plan et le runbook complets.

## Statut des points `# ADAPTER`
Voir src/lens.py — le format exact des lenses (une matrice par couche ? biais ? norm ?) doit être lu dans lenses/README.md.
Le test de conformité `python -m src.validate --smoke` (« sushi → Japan ») doit passer avant tout.
