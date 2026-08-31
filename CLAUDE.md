# CLAUDE.md — règles pour l'agent

Projet : audit en aveugle J-Lens / R-lens / logit lens sur Qwen3.6-27B.
Question : le signal d'un lens en scan aveugle survit-il quand on neutralise le canal de reconstruction du texte ?
Hypothèses à discriminer : H1 lecture interne / H2 text inversion / H3 apophénie du juge.

## Règles non négociables
- Le modèle et les lenses sont chargés UNE fois dans la cellule "SETUP" du kernel persistant (notebooks/main.ipynb).
  Ne jamais recharger, ne jamais redémarrer le kernel sans me demander.
- Toute expérience écrit ses résultats dans results/<nom>.json et ses figures dans figs/<nom>.png. Jamais de résultat uniquement affiché.
- Avant de conclure qu'une expérience "marche", me montrer 3 exemples bruts.
- Le design des conditions est fixé dans src/conditions.py ; ne pas ajouter de condition sans me demander.
- Ne jamais toucher à data/pairs.jsonl après validation humaine (human_checked=true).
- Format des lenses : suivre lenses/README.md (le README du repo HF camilablank/workspace-lenses) à la lettre.
  Les points marqués `# ADAPTER` dans src/lens.py doivent être résolus en lisant ce README, pas en devinant. Si ambigu : demander.
- Chaque fin de session : ajouter une entrée dans experiments.md (fait / vérifié / doute / prochaine étape).
- Le prompt du juge ne doit JAMAIS nommer une famille d'anomalie ni un instrument.
- Avant tout run de condition scan-seul : exécuter `python -m src.checks leak` et me montrer le résultat.

## Ce que je fais moi-même (ne pas déléguer)
- Lecture des 40 paires ; lecture des 30 scans ; recalcul indépendant des AUC ; décisions de pivot ; rédaction du doc et de l'exec summary.
