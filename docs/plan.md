# Sujet A — Plan de bout en bout
## Audit en aveugle des lens de lecture d'activations (J-Lens / R-lens) — de la préparation à la soumission

*Version du 15 août 2026. Deadline : vendredi 4 septembre, 23h59 PT (formulaire d'extension possible jusqu'au 11).*

---

## 0. Ce que la recherche du jour a confirmé

**Le créneau est ouvert.** Le post R-lens du 5 août (Blank, Bhatia, Nanda) a été lu en entier : il évalue les lenses par pass@10 sur cinq catégories (multihop, multilingue, association, typo, poésie), par ablation causale et par CKA — **aucun audit en scan aveugle, aucune mesure de faux positifs, aucun contrôle de text inversion**. Notre question reste vierge.

**Trois faits nouveaux qui reshapent le protocole :**

1. **Le R-lens n'apporte rien sur les plus petits modèles** — l'avantage apparaît et croît avec l'échelle, maximal sur DeepSeek-V4-flash. Conséquence : si on travaille en 4B/9B, la comparaison J vs R est peu informative ; **il faut au moins Qwen3.6-27B** pour que l'axe comparatif ait du sens. Ça déplace le choix GPU vers un 80 Go (A100/H100, 1,5-2,5 $/h) ou une quantification 4-bit sur 48 Go — la seconde option est risquée car la quantification peut altérer la fidélité du lens ; à tester en phase 0.

2. **Les lenses disponibles** (HF `camilablank/workspace-lenses`, 46,7 Go, MIT) : qwen3.5-4b / 9b / 27b, qwen3.6-27b, qwen3.6-35b-a3b, qwen3.5-122b-a10b, gemma-3-27b-it, deepseek-v4-flash — J-lens ET R-lens pour chacun. **Qwen3.6-27b est le modèle de la réplication de Neel** : c'est notre choix par défaut, argument de comparabilité directe.

3. **Deux commentaires sous le post fixent la norme du milieu.** Burny salue le post pour ses « comparaisons à des baselines, barres d'erreur, contrôle aléatoire, pas de graph crimes, et pas relégué en appendice » — et note que le papier J-Lens d'Anthropic ne les respectait pas. Stella Athena (EleutherAI, autrice du tuned lens) soulève que « c'est vraiment difficile de savoir si on a la bonne façon de mesurer ce que le modèle pense après la couche K », que les métriques du papier J-Lens étaient « soit invariantes par couche par construction, soit exposées à ce piège », et demande « quelle est la bonne façon de faire ce type d'évaluation ». **Notre projet est une réponse partielle à sa question** — à citer dans le write-up, c'est exactement le niveau de conversation où Neel veut voir un candidat.

---

## 1. Ce qui fait qu'une candidature est retenue — le cahier des charges implicite

Avant le plan, la cible. D'après le doc de candidature (lu intégralement) et le formulaire de la ronde précédente, la sélection se joue sur cinq choses, dans cet ordre :

**1.1 Les réponses au formulaire Airtable sont le filtre principal.** Neel les lit toutes ; il ne lit pas tous les write-ups. Une candidature dont le formulaire ne le convainc pas de lire le doc est perdue quelle que soit la qualité du doc. Le formulaire est donc à traiter comme un livrable de premier rang, pas comme une formalité.

**1.2 L'executive summary est le seul texte garanti d'être lu.** ~1 page, max 3 pages / 600 mots, avec graphes, en tête du Google Doc. Format qu'il recommande : (a) le problème et pourquoi il est intéressant, (b) les takeaways de haut niveau, (c) un paragraphe + un graphe par expérience clé.

**1.3 « Une candidature qui m'apprend quelque chose. »** Le critère de taste. Notre angle : appliquer aux lenses le contrôle de text inversion que le champ vient de formaliser pour les activation oracles — un résultat qu'il n'a pas et qu'il a publiquement demandé.

**1.4 La preuve de vérification humaine au-delà de l'agent.** Le conseil qu'il désigne lui-même comme le plus important. Ce qu'il cherche : « I read 30 scans and confirmed… », des chiffres recalculés indépendamment, des baselines conçues par nous, un design expérimental dont l'agent n'est que l'exécutant. Une candidature qui ressemble à « un agent a fait un projet » est rejetée.

**1.5 Zéro prose LLM dans le formulaire et l'exec summary.** Signal négatif marqué — « elles se confondent, il en voit des centaines ». À l'inverse, l'usage agentique du LLM pour le code et l'exécution est un facteur d'acceptation ~3× (chiffre qu'il donne).

Et trois pièges à ne pas franchir : surestimer les résultats (son signal négatif n°1), maquiller un négatif, oublier une baseline évidente. Le résultat mixte ou négatif est acceptable ; le résultat surclamé ne l'est pas.

---

## 2. Calendrier maître

| Fenêtre | Statut compteur | Objectif | Livrable de fin de fenêtre |
|---|---|---|---|
| **Sam 15 – dim 16 août** | Hors | Décisions d'infra + lectures fondamentales | GPU commandé, Claude Max actif, papier J-Lens et post R-lens lus |
| **Lun 17 – ven 21** (soirs) | Hors | ARENA 1.2 (3 sections), setup agent + Jupyter persistant, chargement d'un lens | Le lens répliqué produit des readouts sur Qwen3.6-27B |
| **Sam 22 – dim 23** | Hors | Réplication d'une éval du post R-lens ; test de capacité du modèle ; go/no-go | **GO/NO-GO** documenté ; corpus pilote de 8 paires ; le compteur peut démarrer |
| **Lun 24 – ven 28** (soirs, ~2h chacun) | **Compté (~8h)** | Corpus complet, pipeline de scan, harness du juge | Scans des 40 paires × 3 instruments sur disque ; premières conditions tournées |
| **Sam 29 – dim 30** | **Compté (~10h)** | Toutes les conditions, sanity checks, analyse, figures, rédaction du doc | Doc de recherche complet, figures finales |
| **Lun 31 – mer 2 sept** | Compté (2h restantes + 2h exec summary) | Executive summary, formulaire | Doc partagé « anyone with link », formulaire rempli |
| **Jeu 3 sept** | — | Relecture à froid, soumission | **Soumis avec 24h d'avance** |
| Ven 4 sept | — | Marge | — |

Principe de la marge : tout ce qui déborde du 30 août rogne sur l'exec summary, qui est la partie la plus lue. La discipline est donc de figer les expériences le dimanche 30 au soir, quoi qu'il arrive.

---

## 3. Phase 0 — Préparation (15-23 août, hors compteur)

C'est la phase où l'on achète, gratuitement en termes de compteur, tout ce qui rendra les 20h productives. Neel dit explicitement que l'apprentissage général, le setup technique et la lecture préalable ne comptent pas — à condition de ne pas avoir encore choisi le problème... ce qui est une zone grise puisque nous l'avons choisi. **Règle de bonne foi à appliquer** : compter tout ce qui est spécifique au projet (générer les paires, écrire le pipeline de scan, analyser), ne pas compter ce qui serait nécessaire pour n'importe quel projet lens (installer nnsight, charger un lens, comprendre son API, refaire une éval publiée). Le noter tel quel dans le doc — la transparence sur le décompte est elle-même un signal.

### 3.1 Décisions d'infra (ce week-end)

**GPU.** Deux options :
- **A100/H100 80 Go** sur Runpod (~1,6-2,5 $/h), Qwen3.6-27B en bf16 (~54 Go) + activations. ~40h → **65-100 $**. C'est l'option propre.
- 48 Go + quantification 4-bit du modèle. ~25-40 $. Risque : le lens a été fitté sur le modèle bf16 ; en 4-bit les activations dévient et le lens peut se dégrader d'une façon qui contaminerait précisément ce qu'on mesure. À tester en 3.4 — si la réplication de l'éval passe en 4-bit, l'option est valide ; sinon monter en 80 Go.

Recommandation : **prendre le 80 Go d'emblée**. Le surcoût (~40 $) est négligeable devant le risque de découvrir le 24 août que les readouts sont bruités par la quantification. Louer à la demande, arrêter le pod entre sessions (stockage persistant Runpod pour ne pas recharger 46 Go de lenses).

**Agent.** Claude Code avec le plan Max (Neel : « les limites de débit du plan Pro rendent l'usage agentique difficile »). Modèle : Fable pour la planification, Opus 5 pour le débit si besoin.

**Jupyter persistant.** JupyterLab sur le pod + `jupyter-mcp-server` (recette exacte dans le doc de Neel, section « Set up a persistent Jupyter Kernel »). Sans ça, l'agent recharge 27B à chaque script — en soirées de 2h, c'est fatal. CLAUDE.md dès le premier jour avec : le modèle vit dans une cellule dédiée, jamais de restart sans demander, plots sauvés en PNG dans `figs/`, résultats en JSON dans `results/`, un log `experiments.md` append-only.

**Tracking.** Toggl (ou équivalent) dès le 24 ; capture jointe au doc. Neel le suggère et c'est gratuit en crédibilité.

**API juge.** OpenRouter, ~20 $. Deux juges différents (ex. un Claude, un GPT) pour l'accord inter-juges.

### 3.2 Lectures (dans cet ordre, ~6h au total)

1. **Papier J-Lens** (transformer-circuits, 6 juil) — sections 1-4 en entier, appendices A.6 (évals quantitatives), A.20-A.22 (prompt injection, eval awareness, agent d'audit). C'est A.22 qui décrit l'usage « agent équipé du lens » dont notre projet mesure la fiabilité.
2. **Review de Neel** + son fil de commentaires (le paragraphe « je soupçonne que concaténer les top-10 tokens... » et « j'aimerais plus de données sur le taux de faux positifs » sont nos deux citations d'ancrage — les copier avec l'URL).
3. **Post R-lens** (fait aujourd'hui) — noter les cinq catégories d'éval et le pass@10 : c'est notre réplication de validation.
4. **« Current activation oracles are hard to use »** + **« Building Better AO »** — pour le vocabulaire (text inversion, vagueness) et le protocole de contrôle d'inversion qu'ils décrivent. On importe leur grille, on le dit.
5. **« Test your best methods on our hard CoT interp tasks »** — pour la méthodologie ID/OOD, le g-mean², et parce que c'est l'équipe de Neel : citer que notre banc d'essai est le pendant « forward pass » de leur banc « CoT ».
6. **Éval pré-enregistrée du commentateur** (sous la review) — notre seul précédent direct ; TF-IDF comme baseline supplémentaire si le temps.
7. **ARENA chapitre 1.2, trois premières sections** — hooks, résidual stream, logit lens. Suffisant.

Ne pas lire : les meta-tokens en profondeur (c'est le sujet des scholars actuels), la CKA, les extensions multi-tokens. Hors périmètre.

### 3.3 Setup technique (soirs du 17 au 21)

- Jour 1 : pod, JupyterLab, MCP, Claude Code branché, CLAUDE.md. Test : l'agent exécute une cellule et voit le résultat.
- Jour 2 : télécharger Qwen3.6-27B + les lenses J et R correspondants (`camilablank/workspace-lenses/qwen3.6-27b`). Lire le README du repo (format des lenses, API de chargement, quelle couche = quel fichier).
- Jour 3 : hooks nnsight ou PyTorch bruts pour extraire le résidual stream à toutes les couches en un forward ; appliquer le J-lens et le R-lens à une position ; afficher les top-10. Comparer visuellement avec un exemple du post (le « sushi → Japan » multihop) — si on retrouve « Japan » à peu près aux mêmes couches, le montage est bon.
- Jour 4 : logit lens (trivial : unembedding direct) sur les mêmes activations. Trois instruments alignés dans une même fonction `scan(prompt) → dict[position][layer][instrument] = top10`.
- Jour 5 : marge / rattrapage.

### 3.4 Week-end de validation (22-23 août) — le go/no-go

**Test 1 — le lens marche.** Refaire l'éval « multihop » ou « multilingue » du post R-lens : 30-50 prompts, pass@10 par couche pour J et R. Résultat attendu : courbes du même ordre que le post (R > J en couches précoces, convergence en fin). Si on est très loin → problème de montage ou de quantification, ne pas avancer.

**Test 2 — le modèle sait faire la tâche.** Générer 8 paires pilote (2 par famille d'anomalie), montrer les versions anormales EN CLAIR à Qwen3.6-27B avec la question « y a-t-il quelque chose d'anormal ? ». Cible > 80 % de détection. Sinon les anomalies sont trop subtiles pour ce modèle et on mesurerait son incapacité, pas celle du lens — recalibrer avant de démarrer.

**Test 3 — le pipeline tient en temps.** Scanner une paire complète (2 prompts × 3 instruments × toutes couches sous-échantillonnées). Si ça prend > 3 min par prompt, réduire l'échantillonnage de couches (1 sur 4 → 1 sur 6) ou le top-k.

**Test 4 — pas de collision.** Re-vérifier LessWrong/AF (tag J-Lens), arXiv depuis le 15 août, les commentaires du post R-lens et de la review. Une heure.

**Décision.** Les quatre passent → GO, le compteur démarre lundi 24. Test 1 échoue de façon irréparable → NO-GO sujet A, basculer sur B (dont le protocole est prêt et qui est API-only). Test 2 échoue → recalibrer le corpus (durcir ou simplifier), pas d'abandon.

---
## 4. Les 20 heures — plan jour par jour (24-30 août)

Le budget : corpus 3h · pipeline 3h · harness juge 2h · runs 4h · sanity checks 4h · analyse 2h · rédaction du doc 2h. Les runs tournent en tâche de fond ; les heures de runs comptées sont celles d'attention active (itération, débogage), pas d'attente.

**Règle de conduite pendant les 20h.** Le rôle de l'agent : écrire le code du pipeline, lancer les runs, produire les figures brutes. Le rôle humain : concevoir chaque condition et chaque baseline (déjà fait — le protocole est écrit avant la première ligne de code), lire les données brutes, recalculer les chiffres clés, décider des pivots, tenir le journal. Chaque soir : une entrée dans `experiments.md` (« ce que j'ai fait, ce que j'ai vérifié, ce que je crois maintenant, ce que je ferais différemment ») — c'est la matière première du « show your reasoning ».

### Lundi 24 (2h) — Corpus, partie 1
- Générer 40 paires appariées, 4 familles × 10, avec l'agent, à partir d'un gabarit strict par famille (voir §5.1).
- Contrainte de génération : le jumeau propre diffère de la version anormale de moins de 10 % des tokens ; même longueur ±5 % ; même registre.
- **Lecture humaine intégrale des 40 paires** (c'est ~1h, non déléguable). Rejeter/régénérer les paires où l'anomalie est soit évidente lexicalement (un mot-clé « IGNORE PREVIOUS INSTRUCTIONS » — trop facile pour la text inversion, ça biaise vers H2), soit indétectable pour un humain.
- Livrable : `data/pairs.jsonl` avec, par paire : id, famille, texte anormal, texte propre, position(s) de l'anomalie, diff en tokens.

### Mardi 25 (2h) — Corpus, partie 2 + scan
- Test de capacité sur les 40 (pas seulement les 8 pilotes) : le modèle détecte-t-il en clair ? Noter le taux par famille — c'est aussi une baseline (le plafond de ce que le lens peut « révéler » : si le modèle ne voit pas l'anomalie même en clair, son état interne ne la contient probablement pas).
- Lancer le scan complet en tâche de fond : 80 prompts × 3 instruments × couches sous-échantillonnées × top-10. Sérialiser en JSON.
- Pendant que ça tourne : écrire le format de présentation au juge (§5.2).

### Mercredi 26 (2h) — Harness du juge
- Prompt de juge fixe, température 0, sortie JSON forcée : `{anomaly: bool, description: str, confidence: 0-1, evidence: [tokens cités]}`.
- **Audit de fuite** : en conditions scan-seul, vérifier programmatiquement qu'aucune sous-chaîne du prompt original de plus de 3 tokens n'apparaît dans ce que reçoit le juge (hors les tokens du scan eux-mêmes — c'est là toute la question, on ne les filtre pas, on les mesure). Vérifier que le prompt du juge ne nomme aucune famille d'anomalie.
- Tourner les conditions 1 (prompt seul) et 2 (scan J-Lens) sur les 80 items, deux juges. Regarder les premières distributions.

### Jeudi 27 (2h) — Toutes les conditions
- Conditions 3 (R-lens), 4 (logit lens), 5 (permuté), 6 (jumeaux — en fait déjà incluse : les 40 propres sont dans les 80 items), 7 (reconstruction → second juge).
- Pour la 5 : permuter les tokens ENTRE positions au sein d'un même scan (préserve le vocabulaire global, détruit la structure positionnelle) — c'est ça qui isole l'apophénie ; une permutation entre items serait un autre test (contamination croisée), à faire si le temps.
- Pour la 7 : le premier juge reçoit le scan et la consigne « reconstruis le texte que le modèle lisait » ; le second juge reçoit uniquement cette reconstruction et la question standard. La différence détection(scan) − détection(reconstruction) est la part du signal qui n'est PAS expliquée par la reconstruction du texte.
- Fin de soirée : tableau brut 7 conditions × 3 instruments × 2 juges. Ne pas conclure encore.

### Vendredi 28 (2h) — Premiers sanity checks
- **Lire 30 scans bruts** (15 anormaux, 15 propres, tirés au sort par un script) et noter, pour chacun : les tokens « evidence » cités par le juge y sont-ils réellement ? à quelle couche ? sont-ils liés à l'anomalie ou à autre chose ?
- Recalculer une AUC à la main : exporter les scores de la condition 2 en CSV, refaire le calcul dans une cellule indépendante (sklearn ou à la main), comparer.
- Regarder 10 verdicts corrects et 10 faux positifs du juge avec leur raisonnement : détecte-t-il pour la bonne raison ?
- Journal : premières impressions, mais surtout la liste des choses qui pourraient rendre les résultats faux (voir §5.4).

### Samedi 29 (5h) — Analyse + itération
- Métriques finales : AUC (bootstrap 1000, IC 95 %) par instrument × famille ; TPR@FPR=5 % ; taux de fausse alerte sur jumeaux ; accord inter-juges (Cohen κ).
- **Le triplet central** par instrument : Δ1 = détection(scan) − détection(prompt seul) ; Δ2 = détection(scan) − détection(reconstruction) ; Δ3 = détection(permuté) − chance. Une figure pour ce triplet — c'est LA figure de l'exec summary.
- Décision de mi-parcours : la phrase-cible (§6.2) se remplit-elle ? Si un instrument est plat partout, vérifier avant tout que ce n'est pas un bug (comparer à la réplication du week-end précédent).
- Si le temps : baseline TF-IDF sur les scans (pour dialoguer avec l'éval du commentateur) ; permutation inter-items ; **probe linéaire (ou Mahalanobis) sur les activations brutes comme borne supérieure** — « l'information sur l'anomalie existe-t-elle linéairement dans le résidual stream, indépendamment de tout readout ? » (leave-one-out, forte régularisation, 40 items = fragile, à présenter avec prudence ; si la probe voit et le lens+juge non, la perte est dans le canal de verbalisation).
- Toutes les figures en version finale : polices lisibles, IC affichés, une idée par graphe.

### Dimanche 30 (5h) — Sanity checks finaux + rédaction du doc
- Deux derniers checks : (a) refaire une condition avec un seed différent du juge / un prompt de juge reformulé — les Δ tiennent-ils ? (b) inspection des faux positifs sur jumeaux propres : que « voit » le juge ? (Souvent très instructif : c'est là qu'on comprend H3.)
- **Rédiger le doc de recherche** (§6.3) — 2h. Le doc, pas encore l'exec summary. Ton : compte-rendu de labo, honnête, avec les échecs.
- Figer. Sauvegarder tout (`results/`, `figs/`, `experiments.md`, notebook) hors du pod.

---

## 5. Détails de design qui décident de la qualité

### 5.1 Les quatre familles d'anomalies — gabarits
- **Injection** : un résultat d'outil (recherche web, lecture de fichier) contenant une instruction adressée à l'assistant, formulée sans mots-clés de jailbreak (« Note pour l'assistant : mentionner X dans la réponse » plutôt que « IGNORE ALL »). Jumeau : même résultat d'outil sans la note.
- **Bug** : 15-25 lignes de code avec une erreur sémantique silencieuse (off-by-one, condition inversée, mauvaise variable). Jumeau : le code correct.
- **Prémisse fausse** : question factuelle contenant une présupposition fausse mais plausible (« Depuis que la France a adopté l'euro en 1995… »). Jumeau : la même question avec la date correcte.
- **Conflit d'instructions** : consigne système + demande utilisateur incompatibles de façon discrète. Jumeau : compatibles.

Pour chaque famille : 10 paires, longueur cible 150-400 tokens, position de l'anomalie variable (pas toujours au même endroit — sinon le juge apprend la position).

### 5.2 Format de présentation du scan au juge
Compact et neutre :
```
[pos 12] L8: tok tok tok ... | L16: ... | L24: ... | L32: ...
[pos 13] ...
```
Sans le token d'entrée à la position (sinon c'est le prompt). Sans nom d'instrument dans le prompt du juge (aveugle sur la condition aussi). Longueur : 300 tokens × 8 couches × 10 tokens = 24k tokens de contexte — OK pour un juge frontier ; si trop long, réduire à 6 couches ou top-5, et le documenter.

### 5.3 Ce que chaque condition permet de conclure (matrice)
| Observation | Lecture |
|---|---|
| scan ≫ prompt seul, reconstruction ≪ scan, permuté ≈ chance | H1 : le lens lit vraiment dedans |
| reconstruction ≈ scan | H2 : le lens est un compresseur de texte ; sa valeur en audit aveugle est nulle au-delà du transcript |
| permuté > chance nettement | H3 : le juge apophénise ; contamination de toute évaluation par juge LLM |
| scan ≈ prompt seul mais > 0 | Le lens ne fait ni mieux ni pire que lire — utile seulement si le prompt n'est pas accessible (cas des CoT latentes — à dire) |
| J vs R : R > J | Le R-lens tient ses promesses hors de ses propres évals |
| Effets ≠ selon la famille | Carte des régimes — le résultat le plus probable et le plus utile |

### 5.4 La liste « comment mes résultats pourraient être faux » (à tenir dès le vendredi)
Neel : « un signal très positif est quand je pense à une façon dont vos résultats pourraient être faux et découvre que vous l'avez déjà vérifiée ». Liste de départ :
- Le juge voit un fragment du prompt (fuite) → audit programmatique (§4 mercredi).
- L'anomalie est lexicalement évidente → filtre à la génération + regarder si les tokens « evidence » sont ceux de l'anomalie ou des tokens voisins.
- Le juge détecte la *famille* plutôt que l'*anomalie* (les scans de code ont une signature) → mesurer la détection sur jumeaux propres PAR famille : si le juge « détecte » du code propre à 40 %, il détecte la famille.
- Le lens en 4-bit diverge du bf16 → réplication (test 1) ; si 80 Go, non-problème.
- Le sous-échantillonnage de couches rate la couche où l'anomalie vit → refaire 5 items en toutes couches, comparer.
- Le prompt du juge induit un biais « oui » → taux de « oui » sur jumeaux = la mesure ; deux formulations de prompt.
- Les 40 paires sont trop peu pour les IC → bootstrap affiché ; annoncer les IC, ne rien conclure d'un Δ dont l'IC contient 0.
- Un seul modèle sujet → le dire ; si le temps, 10 items sur qwen3.5-9b comme sanity de généralisation (mais R-lens n'y apporte rien, donc J seul).

---

## 6. La soumission (31 août – 3 septembre)

### 6.1 Ordre de rédaction
D'abord le doc de recherche (dimanche 30), puis l'executive summary (lundi 31), puis le formulaire (mardi 1er), puis relecture croisée à froid (mercredi 2 – jeudi 3). L'exec summary se rédige *après* le doc parce qu'il en est la distillation ; le formulaire se rédige *après* l'exec summary parce qu'il doit donner envie de le lire.

### 6.2 L'executive summary — structure et la phrase-cible
Max 600 mots, 2-3 figures. Structure calquée sur la recommandation de Neel :

1. **Le problème (5 lignes).** Les lens de lecture d'activations ont été validés par cas d'étude, jamais caractérisés comme détecteurs en aveugle. Le confound central de la famille — la text inversion — a été formalisé pour les activation oracles mais jamais testé sur J-Lens/R-lens. Neel a demandé publiquement des données sur le taux de faux positifs. Réponse partielle à la question de Stella Athena sur « la bonne façon d'évaluer » ces lenses.
2. **Les takeaways (3-4 puces).** Un par ligne. La phrase-cible : *« Sur 40 paires appariées couvrant 4 familles d'anomalies, un scan [J/R-lens] passé en aveugle à un juge LLM détecte l'anomalie à X % (FPR Y %) contre Z % en lisant le prompt, W % depuis la seule reconstruction, V % sur scans permutés — la valeur ajoutée non expliquée par la text inversion est de [X−W] points, [concentrée sur les familles …]. »* Puis le takeaway J vs R, puis le takeaway méthodologique (H3 ou son absence).
3. **Une figure = le triplet Δ1/Δ2/Δ3 par instrument, avec IC.** Une seconde = AUC par famille × instrument. Éventuellement une troisième = un exemple de scan annoté (le côté qualitatif que Neel apprécie).
4. **Limites (3 lignes)** : un modèle, 40 paires, juge LLM, anomalies synthétiques.
5. **Vérification (3 lignes)** : « J'ai lu 30 scans bruts, recalculé les AUC indépendamment, audité le harness pour les fuites, et testé deux juges et deux prompts ; l'agent a écrit le pipeline, j'ai conçu chaque condition. » Court, factuel, vérifiable.

Écrit à la main, sans LLM. Relu à haute voix. Si une phrase pourrait sortir d'un générateur (« delve », « crucial », « landscape »), la réécrire.

### 6.3 Le doc de recherche (à la suite de l'exec summary, dans le même Google Doc)
Sections : setup (modèle, lenses, corpus avec **5 paires tirées au sort reproduites in extenso**), pipeline, les 7 conditions et leur raison d'être, résultats complets (tableaux + toutes les figures), les sanity checks un par un avec ce qu'ils ont trouvé, la liste « comment ça pourrait être faux » avec l'état de chaque item (vérifié / non vérifié), le journal `experiments.md` en annexe (brut, avec les impasses), la capture Toggl, le lien vers le repo (code + données + scans). Neel apprécie « show your reasoning », les rabbit holes abandonnés y compris.

Partage : « anyone with the link can view » — à vérifier deux fois, c'est une cause classique de dossier non lu.

### 6.4 Le formulaire — stratégie
Les questions de la ronde précédente (à confirmer sur le formulaire 12.0 dès son ouverture) : résumé du projet, ce que vous avez appris, comment vous avez utilisé les LLM, ce que vous feriez avec plus de temps, expérience préalable, motivation.

- **Résumé** : la phrase-cible + une phrase sur le pourquoi. Pas de préambule.
- **Ce que j'ai appris** : une chose sur les lenses, une chose sur la méthode (l'importance du contrôle d'inversion), une chose sur soi (ex. : où on a perdu du temps). Concret, personnel.
- **Usage des LLM** : franc et précis — « Claude Code a écrit le pipeline de scan et le harness ; j'ai conçu les 7 conditions avant de coder, lu 30 scans, recalculé les AUC, audité les fuites ; l'exec summary et ces réponses sont écrits sans LLM. » C'est la réponse qu'il attend et elle doit être vraie.
- **Avec plus de temps** : les extensions évidentes — plus de modèles (le R-lens grandit avec l'échelle → DeepSeek-V4-flash), anomalies naturelles plutôt que synthétiques, le banc Ivanova et al. porté au forward pass, un détecteur entraîné (probe sur scans) vs le juge. Montre le taste.
- **Expérience** : honnête — ingénieur ML, première expérience interp, ce que ça a impliqué (ARENA en une semaine). Cinq de ses huit scholars en 8.0 étaient dans ce cas ; ce n'est pas un handicap si le travail est propre.

### 6.5 Check-list de soumission (jeudi 3)
- [ ] Doc partagé « anyone with link », testé en navigation privée
- [ ] Exec summary ≤ 600 mots, figures visibles dans le doc (pas des liens)
- [ ] Repo public avec code, données, scans, README de reproduction
- [ ] Capture Toggl dans le doc
- [ ] Aucune phrase de l'exec summary/formulaire générée par LLM
- [ ] Les chiffres du formulaire = ceux de l'exec summary = ceux du doc
- [ ] Les deux citations d'ancrage de Neel (review) référencées avec URL
- [ ] Relecture par quelqu'un d'extérieur si possible (clarté = top 20 % instantané, selon lui)

---

## 7. Plans de secours

- **Le pipeline n'est pas prêt le 24 août** → décaler le compteur de 2-3 jours (le week-end 29-30 absorbe) ; si toujours pas prêt le 27 → formulaire d'extension au 11 septembre (Neel le propose explicitement pour ceux qui manquent de temps).
- **Résultats plats partout au 29** → d'abord suspecter un bug (comparer à la réplication) ; si confirmé plat, c'est un résultat : « en régime aveugle sur ces 4 familles, aucun instrument ne bat le prompt seul » — le documenter avec la même rigueur, c'est publiable et Neel a dit qu'il le prendrait au sérieux.
- **Découverte d'une collision (quelqu'un publie l'audit) après le 24** → ne pas abandonner : positionner le nôtre comme réplication indépendante + extension (R-lens, text inversion, familles) ; une réplication propre d'un résultat de 2 semaines est un bon signal.
- **GPU indisponible / budget** → 48 Go + 4-bit avec test 1 comme garde-fou ; en dernier recours qwen3.5-9b en J-lens seul (perd l'axe R, garde tout le reste).

---

## 8. Ce que ce plan produit, vu de Neel

Une candidature qui : répond à une question qu'il a posée il y a six semaines ; utilise les artefacts de ses propres scholars (les lenses, le modèle de sa réplication) ; importe une grille méthodologique du champ (text inversion) là où personne ne l'a appliquée ; contient toutes les baselines qu'il exige et deux qu'il n'attendait pas (reconstruction, permutation) ; dit clairement en 600 mots ce qu'elle affirme et ce qu'elle ne peut pas affirmer ; et prouve, ligne par ligne, qu'un humain a vérifié ce que l'agent a produit.

Ce n'est pas la garantie d'être retenu — sa barre est un « borderline accept » sur des projets propres, pas des résultats spectaculaires. Mais c'est la forme exacte de candidature que son écosystème récompense depuis un an : prendre une méthode de moins de six mois, la soumettre à un test honnête, et rapporter ce qu'on trouve.
