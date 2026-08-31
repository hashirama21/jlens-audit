# Les trois sujets candidats — dossier détaillé
## Candidature MATS 12.0 (Winter 2026-27), stream Neel Nanda — deadline vendredi 4 septembre 2026, 23h59 PT

**État du dossier au 15 août 2026.** Ce document consolide tout le travail de cadrage : les trois formulations de chaque sujet (superficielle → intermédiaire → profonde), l'état de l'art vérifié par recherche web, les protocoles expérimentaux, les baselines, les sanity checks, les budgets horaires, les arbres de résultats et les critères go/no-go.

**Rappel des contraintes de l'exercice** : ~16h de recherche (max 20h) + 2h pour l'executive summary et le formulaire. Le compteur ne démarre qu'au choix du problème — toute la préparation générale (tutoriels, setup GPU, lecture des papiers) est hors compteur. Si le projet est abandonné pour un autre, le compteur repart à zéro.

**Rappel du profil** : ingénieur ML sans expérience interp, disponible soirs et week-ends, budget compute à décider (recommandation : 1× GPU 48 Go sur Runpod/vast.ai, 50-80 $ total, + plan Claude Max pour la période).

**Les cinq problèmes de fond du champ** (grille de lecture commune aux trois sujets) :
1. **Vérité terrain** — l'interprétabilité n'a pas d'oracle ; on valide les techniques les unes contre les autres (circulaire), sauf à construire des systèmes où la réponse est connue par construction.
2. **Narrateur non fiable au carré** — le modèle confabule (CoT infidèle) ET l'instrument confabule (NLA, oracles, lenses) ; tout canal génératif vers le langage peut inventer.
3. **Effet observateur** — les modèles détectent l'évaluation et changent de comportement ; la chaîne d'évaluation comportementale casse (cas Apollo/Opus 4.6).
4. **Valeur ajoutée** — regarder dedans doit battre « demander au modèle » et « lire le transcript » pour justifier son existence ; la carte des zones où le black-box échoue n'existe pas.
5. **Attribution** — le comportement sous-détermine la cause interne ; le diagnostic malice/confusion détermine la réponse de sécurité mais n'a pas de métrologie.

---

# SUJET A — Audit en aveugle des lens de lecture d'activations (J-Lens / R-lens)
## « Quand un lens scanne tout, d'où vient le signal — et à quel taux de fausse alerte ? »

### A.1 — La question, à trois profondeurs

**Superficielle** : mesurer le taux de faux positifs du J-Lens utilisé en scan automatique (toutes positions × toutes couches) avec un juge LLM en aval.

**Intermédiaire (le problème de méthode)** : le J-Lens a été validé par *cas d'étude* — on savait quoi chercher, on a regardé, on l'a trouvé (injection, biais RM-sycophant). Un audit réel fonctionne à l'envers : on ne sait pas quoi chercher. La question devient celle de tout test de dépistage : sensibilité, spécificité, et comportement en **régime de basse prévalence** — un détecteur à 5 % de fausses alertes est inutilisable en monitoring où les vrais positifs sont rarissimes. Cette caractérisation n'existe pas. Quelle est la courbe ROC d'un instrument de lecture d'états internes, et de quel côté de l'utilisabilité tombe-t-elle ?

**Profonde (le problème épistémologique)** : quand le juge muni du scan détecte l'anomalie, **par quel canal l'information passe-t-elle ?** Trois hypothèses concurrentes, discriminables expérimentalement :
- **H1 — lecture interne réelle** : le lens transmet un état interne inaccessible autrement.
- **H2 — text inversion** : le lens fonctionne comme un résumé lossy de l'entrée ; les tokens surfacés reflètent le prompt, et un juge lisant le prompt directement ferait aussi bien. (« Text inversion » est désormais le terme établi — formalisé dans la littérature des activation oracles comme LE confound central de toute cette famille de méthodes.)
- **H3 — apophénie du juge** : le juge construit un récit cohérent à partir de tokens bruités ; le taux de détection est un artefact de la complaisance narrative du lecteur, pas de l'instrument.

Le triptyque H1/H2/H3 est l'instanciation locale du problème de fond n°2 (narrateur non fiable au carré : instrument génératif lu par un juge génératif) rencontrant le n°4 (la baseline « lire le prompt » doit être battue pour que l'instrument existe).

**Profondeur supplémentaire à mentionner sans prétendre la résoudre** : le J-Lens force les concepts internes à passer par le vocabulaire de sortie — hypothèse forte que les variables de la mémoire de travail sont *verbalisables en tokens uniques*. Le taux de faux négatifs mesure donc, en creux, la fraction du calcul interne qui échappe au langage. Neel pointe lui-même cette limite (« il est hautement improbable qu'il y ait correspondance un-à-un entre concepts et tokens ») et appelle aux extensions multi-tokens.

### A.2 — Pourquoi Neel s'y intéresse (traçabilité verbatim)

- Sa review publique du papier J-Lens (6 juillet 2026) : il « aimerait voir si une analyse automatisée peut être faite sur les sorties J-Lens à tous les tokens et couches pour flaguer des choses comme la prompt injection plus efficacement qu'en regardant chaque token manuellement » ; il « soupçonne que concaténer les top-10 tokens J-Lens partout et le mettre dans le contexte d'un LLM suffirait à faire émerger l'hypothèse d'injection ». Prédiction publique, non testée, testable en 20h.
- Même review : il « s'attend à beaucoup de faux positifs » et « aimerait voir plus de données collectées sur sa fiabilité et son taux de faux positifs ».
- Doc de candidature, section « Improved Interpretability Methods » : « particulièrement intéressé par les améliorations et le red-teaming des méthodes nouvelles et prometteuses… probing de failles potentielles et design d'évals pour celles-ci ». J-Lens listé nominativement avec les ressources de ses scholars.
- Positionnement stratégique GDM (juillet 2026) : l'équipe veut se concentrer sur le régime où la CoT n'est plus monitorable — « techniques alternatives de forensics, extension de l'interp aux architectures latentes ». Le lens-comme-backup-de-la-CoT est dans leur trajectoire déclarée.

### A.3 — État de l'art vérifié (15 août 2026)

| Pièce | Date | Ce qu'elle apporte | Ce qu'elle laisse vide |
|---|---|---|---|
| Papier J-Lens (Anthropic, Gurnee et al.) | 6 juil | Cas d'étude : injection, blackmail, RM-sycophant, eval awareness (A.21), agent d'audit équipé (A.22) | Aucune caractérisation FPR/ROC en scan aveugle |
| Review de Neel + réplication Qwen 3.6 27B | 6 juil | Réplication cheap (n=25 prompts), meta-tokens interprétatifs, appel explicite à l'audit | — |
| Éval pré-enregistrée d'un commentateur (Ratnaditya J) | juil | Sur UN organisme à vérité terrain : le lens ne hallucine pas sur contrôles (0.591 ≈ chance), mais ne bat pas TF-IDF (0.746 vs 0.690) ; signal survit au retrait de la CoT (0.620) | Une tâche, un modèle, un type d'anomalie — terrain quasi vierge |
| **R-lens** (Alignment Forum) | **5 août** | Drop-in remplaçant du J-Lens, backward pass modifié (layerwise-relevance propagation) : readouts plus clairs en couches précoces, moins de tokens non pertinents, détecte des concepts que J-Lens rate. **Lenses open-sourcées.** | Pas d'audit en scan aveugle non plus → l'audit devient naturellement comparatif J vs R vs logit |
| Analyse d'ingénierie (LessWrong, fin juil) | fin juil | Monitoring par lens quasi gratuit au décodage ; MAIS sur petits modèles, certaines directions dominent et doivent être supprimées pour que le lens marche | Artefact d'échelle à connaître en 4B/9B |
| « Test your best methods on our hard CoT interp tasks » (Ivanova, Tyagi, Engels, **Nanda** — MATS 9.0, mars) | mars | 9 tâches objectives où un moniteur black-box GPT 5.2 échoue OOD ; baselines probes/TF-IDF ; datasets open source ; méthodologie tuning ID/OOD réutilisable | C'est le banc d'essai pour la **CoT** — pas pour les lecteurs d'activations en mode scan |
| Littérature activation oracles (Jakkli→Building Better AO) | 2026 | Formalisation de la **text inversion** + protocoles de contrôle ; 49,4 % de réponses vagues/infalsifiables chez les AO | Le contrôle d'inversion n'a jamais été appliqué au J/R-lens |

**Le créneau exact, toujours vide au 15 août** : appliquer au J-Lens et au R-lens, en régime de scan aveugle, le contrôle de text inversion développé pour les AO, avec mesure du taux de fausse alerte sur jumeaux propres. Personne ne l'a fait ; le vocabulaire, les protocoles et les checkpoints existent tous.

### A.4 — Reformulation finale (post-recherche)

> *Le signal d'un lens en scan aveugle survit-il quand on neutralise le canal de reconstruction du texte ?* Audit comparatif J-Lens / R-lens / logit lens comme détecteurs d'anomalies en aveugle, avec séparation expérimentale de H1 (lecture interne), H2 (text inversion) et H3 (apophénie), et caractérisation ROC + taux de fausse alerte.

### A.5 — Protocole détaillé

**Phase 0 — préparation (HORS compteur, semaine du 15-23 août)**
1. Lire : papier J-Lens (+ appendices 10.2/10.3), review de Neel, post R-lens du 5 août (vérifier qu'ils n'ont pas déjà fait l'audit !), post « hard CoT interp tasks », posts AO (« hard to use » + « Building Better »).
2. Monter : pod GPU 48 Go, JupyterLab + jupyter-mcp-server connecté à Claude Code, CLAUDE.md avec les règles (modèle chargé en cellule dédiée, jamais de restart sans demander, plots sauvés en PNG).
3. Charger un J-Lens publié (camilablank/workspace-lenses, ou Neuronpedia qwen3.6-27b/jlens) + les R-lenses open-sourcées. Répliquer UNE éval quantitative du papier (ex. multilingual probing) pour valider que le lens fonctionne. C'est le sanity check n°1 de Neel : « vérifiez que votre J-Lens performe bien, surtout s'il a été monté par un agent autonome ».
4. Vérifier la capacité du modèle : lui montrer les anomalies EN CLAIR et confirmer qu'il les détecte (sinon on mesure l'incapacité du modèle, pas celle du lens — l'erreur « étudier un phénomène sans vérifier qu'il est là »).

**Phase 1 — corpus (≈ 2-3h comptées)**
- ~40 paires appariées, 4 familles d'anomalies × 10 paires :
  - (a) prompt injection dans un résultat d'outil/recherche ;
  - (b) bug subtil dans du code que le modèle doit lire ;
  - (c) prémisse fausse dans une question factuelle ;
  - (d) instruction cachée / conflit d'instructions.
- Chaque paire = version anormale + jumeau propre différant d'un minimum de tokens (idéalement < 10 % du texte).
- Génération assistée par LLM, puis **lecture manuelle de chaque paire** (règle de Neel : si tout repose sur la qualité d'un dataset généré par LLM, le lire soi-même et inclure des exemples tirés au hasard dans le write-up).
- Option d'extension si le temps le permet : ajouter 1-2 tâches du banc Ivanova et al. adaptées au forward pass, pour ancrer la comparaison dans un benchmark que Neel connaît.

**Phase 2 — pipeline de scan (≈ 3h)**
- Scan : toutes positions × couches sous-échantillonnées (1 couche sur 4), top-k=10 tokens par (position, couche), trois instruments : J-Lens, R-lens, logit lens.
- Sérialisation compacte du scan (format texte : position → couche → tokens) pour le contexte du juge.
- Attention à l'artefact petits modèles : suppression des directions dominantes si nécessaire (suivre le post d'ingénierie).

**Phase 3 — juge et conditions expérimentales (≈ 2h de code, runs en tâche de fond)**
Le juge (modèle frontier via OpenRouter, température 0, prompt fixe, consigne : « y a-t-il quelque chose d'anormal dans ce que le modèle traite ? réponds oui/non + description + confiance ») est passé sur chaque item dans 7 conditions :

| # | Condition | Ce qu'elle teste |
|---|---|---|
| 1 | Prompt seul (pas de scan) | Baseline black-box — le plancher que l'instrument doit battre (H2 partiel) |
| 2 | Scan J-Lens seul (sans le prompt) | Détection via l'instrument |
| 3 | Scan R-lens seul | Comparaison inter-instruments |
| 4 | Scan logit lens seul | La baseline « méthode simple » exigée par Neel |
| 5 | Scan permuté (tokens mélangés entre positions) | H3 — si le juge « détecte » encore, apophénie |
| 6 | Scan du jumeau propre + même question | Taux de fausse alerte (spécificité) |
| 7 | Test d'inversion : demander au juge de **reconstruire le prompt** depuis le scan, puis passer cette reconstruction seule à un second juge | H2 — si détection(reconstruction) ≈ détection(scan), le lens n'est qu'un canal de compression du texte |

**Phase 4 — métriques et analyse (≈ 2h)**
- AUC par instrument et par famille d'anomalie ; TPR à FPR fixé (5 %) ; taux de fausse alerte brut sur jumeaux ; deltas condition-à-condition avec IC bootstrap (les 40 paires le permettent).
- Le résultat central n'est PAS l'AUC : c'est le **triplet** [détection(scan) − détection(prompt seul)] (valeur ajoutée H1 vs H2), [détection(scan) − détection(reconstruction)] (part non expliquée par l'inversion), [détection(permuté)] (plancher d'apophénie).

**Phase 5 — sanity checks documentés (≈ 4h, à répartir)**
- Lire 30 scans bruts soi-même et vérifier que les tokens « détectés » par le juge y figurent réellement.
- Recalculer une AUC à la main (one-liner indépendant du pipeline).
- Vérifier l'absence de fuite : le juge en condition 2-5 ne doit voir AUCUN fragment du prompt (audit du harness).
- Vérifier 10 verdicts du juge contre son raisonnement : détecte-t-il pour la bonne raison ?
- Documenter tout ça explicitement dans le write-up (« I read 30 scans and confirmed… » — signal fort selon Neel, qui vérifie).

### A.6 — Baselines obligatoires (récapitulatif)
Prompt seul · logit lens · scan permuté · reconstruction-seule · (bonus si temps : TF-IDF sur le scan, pour dialoguer avec l'éval du commentateur).

### A.7 — Ressources et coûts
- **Lenses** : camilablank/workspace-lenses (Qwen 3.5 4B → deepseek v4 flash), Neuronpedia qwen3.6-27b/jlens, R-lenses du post du 5 août, github.com/anthropics/jacobian-lens.
- **Modèle sujet** : Qwen 3.6 9B ou 27B (27B si le budget GPU suit — c'est le modèle de la réplication de Neel, argument de comparabilité).
- **GPU** : 1× L40S/A6000 48 Go, ~0,5-0,9 $/h, 30-60h → **20-55 $**. Juge via OpenRouter : **10-20 $**.
- **Contexte agent** : le fichier 600k tokens de Neel + papier J-Lens + post R-lens.

### A.8 — Budget des 20h (indicatif)
Corpus 3h · pipeline scan 3h · juge + harness 2h · runs + itération 4h · sanity checks 4h · analyse 2h · rédaction du doc 2h. (+2h exec summary hors compteur des 20.)

### A.9 — Arbre des résultats et leur valeur
- **H1 domine** (scan >> prompt seul, permuté ≈ chance, reconstruction << scan) : résultat positif fort — premier argument quantitatif que les lenses lisent réellement dedans en régime aveugle. Publiable, exactement ce que Neel demandait.
- **H2 domine** (reconstruction ≈ scan) : résultat négatif fort — les lenses en scan sont des compresseurs de texte ; toute la famille de méthodes doit passer le contrôle d'inversion avant usage en audit. Publiable, et Neel a écrit qu'il prendrait ce genre de résultat au sérieux.
- **H3 non négligeable** (permuté > chance) : découverte méthodologique — les évaluations par juge LLM des méthodes d'interp sont contaminées par l'apophénie ; implication pour TOUS les papiers qui utilisent un juge. Peut-être le résultat le plus intéressant des trois.
- **Mixte par famille d'anomalie** (probable) : carte des régimes où le lens ajoute de la valeur — exactement la « carte des zones où le black-box échoue » que le pragmatic interpretability réclame.
- Dans tous les cas : la comparaison J vs R apporte un second axe (le R-lens tient-il ses promesses hors de ses propres évals ?).

### A.10 — Risques et plans de repli
1. **Le post R-lens du 5 août a déjà fait l'audit** → vérification n°1 en phase prépa. S'il est partiel : aubaine (comparateur légitimé). S'il est complet : pivoter vers B (compteur intact car rien n'a démarré).
2. **Artefact directions dominantes en petit modèle** → fix documenté dans la littérature ; sinon monter en 27B.
3. **Juge instable** → température 0, 2 juges différents, reporter l'accord inter-juges.
4. **Résultats plats partout** (tout ≈ chance, même en clair) → les anomalies sont trop subtiles pour le modèle sujet : recalibrer le corpus (c'est pour ça que la vérification de capacité est en phase 0).

### A.11 — Phrase-cible de l'executive summary
> « Sur N paires appariées couvrant 4 familles d'anomalies, un scan [J-Lens/R-lens] passé en aveugle à un juge LLM détecte l'anomalie à X % (FPR Y %), contre Z % pour le même juge lisant le prompt, W % depuis la seule reconstruction du texte, et V % sur scans permutés — la valeur ajoutée au-delà de la text inversion est de [X−W] points. »

### A.12 — Go/no-go
GO si : le post R-lens n'a pas fait l'audit ; le lens répliqué passe une éval du papier ; le modèle détecte les anomalies en clair à > 80 %. NO-GO sinon → basculer sur B.

---

# SUJET B — Forensics avec vérité terrain : la matrice de confusion du diagnostic d'intention
## « Quand le protocole malice-vs-confusion se trompe-t-il, et l'intention est-elle identifiable en principe ? »

### B.1 — La question, à trois profondeurs

**Superficielle** : tester le protocole de model forensics (lire la CoT → hypothèses → éditer le prompt/environnement → tester, itérer) sur des cas où on connaît la vraie cause.

**Intermédiaire (le problème de mesure)** : le forensics tel que publié est une pratique sans métrologie — appliqué à des cas réels dont on ne connaîtra jamais la vérité, il n'a ni taux d'erreur ni biais directionnel connu. Or l'asymétrie des coûts est énorme : conclure « confusion » face à une vraie subversion est la faute grave (la mitigation regex sera contournée), conclure « malice » face à une confusion ne coûte que des ressources. Construire la vérité terrain et mesurer la matrice de confusion du protocole = faire passer le forensics de l'artisanat au test diagnostique.

**Profonde (le problème d'identifiabilité)** : deux causes internes différentes (motivation maligne instillée vs confusion induite) peuvent produire des politiques comportementales **identiques sur tout le support observable** — même action, même CoT plausible. Si les deux organismes répondent identiquement à toutes les interventions contrefactuelles réalisables, aucun protocole comportemental ne peut les distinguer, par construction. La vraie question : **où est la frontière d'identifiabilité ?** Taxonomie visée :
- (i) cause identifiable par intervention simple → le protocole marche ;
- (ii) cause identifiable seulement par des interventions auxquelles l'enquêteur ne pense pas → échec d'imagination, pas d'information ;
- (iii) cause **non identifiable comportementalement** → seuls les internals peuvent trancher. Chaque exemple propre du cas (iii) est un argument positif pour les méthodes internes — la carte que le pragmatic interpretability réclame (problème de fond n°4).

Connexion au problème n°3 : l'eval awareness est le mécanisme par lequel un modèle *déplace activement* la frontière d'identifiabilité — un suspect qui sait qu'on enquête produit exprès des observables compatibles avec la cause bénigne. Forensics et eval awareness sont le même problème vu de l'enquêteur et du suspect.

### B.2 — Pourquoi Neel s'y intéresse (traçabilité)
- Le doc de candidature ouvre sa liste de problèmes par le model forensics, avec la consigne « lisez notre papier, il y a beaucoup de conseils » et le souhait explicite : « je suis assez intéressé par prendre un cadre où le modèle agit de façon plausiblement trompeuse et où [lire la CoT + contrefactuels] ne marche PAS, et voir si on peut mieux comprendre avec des techniques plus lourdes ».
- **Le papier model forensics lui-même (juin 2026) désigne notre projet comme next step** : « les tests comportementaux manquent de contrôles positifs ; créer des contrôles positifs pour les valider est une prochaine étape clé » ; et « les contrefactuels sont flexibles mais confondus — les tailles d'effet sont difficiles à interpréter ».
- GDM a construit Gram (audit automatisé dérivé de Petri, avec forensics automatisé de cause racine) → le versant « construire l'agent » est pris en interne ; le versant « métrologie du diagnostic » ne l'est pas.

### B.3 — État de l'art vérifié (15 août 2026)
- Papier fondateur : protocole 2 étapes, 6 environnements agentiques, limites reconnues (contrôles positifs manquants, contrefactuels confondus, eval awareness comme menace sur le rappel).
- Gram (GDM) : forensics automatisé — occupe l'automatisation, pas l'évaluation.
- « The Model Organism Lottery » (juillet 2026) : l'interprétabilité d'un organisme dépend fortement de la méthodologie d'entraînement qui l'a créé → **la manière d'instiller la vérité terrain est elle-même un confound**. À la fois un piège et un axe du projet (2 méthodes d'instillation × mêmes causes = mesurer ce confound).
- Suggestions d'Aditya Singh (alumnus, doc lié dans la candidature) : liste de projets forensics prêts — à lire en phase prépa pour éviter les collisions et s'aligner.
- Datasets de comportements sketchy disponibles : suite du papier, task gaming blog post, weird chat (Transluce), SWE-chat.

### B.4 — Reformulation finale (post-recherche)
> Construire les **contrôles positifs** que le papier réclame : instiller des causes connues (malice / confusion / ambiguïté d'instruction) dans des environnements agentiques, faire tourner le protocole en aveugle, et publier la première **matrice de confusion** du diagnostic d'intention — avec au moins un cas documenté de non-identifiabilité comportementale si on en trouve un.

### B.5 — Protocole détaillé

**Phase 0 — préparation (hors compteur)**
1. Lire : papier forensics + son blog post, doc d'Aditya Singh, post task gaming, Thought Branches (le résampling est l'outil de focalisation recommandé par le papier), post Gram.
2. Choisir la stack : tout API (OpenRouter pour les modèles open-weight avec CoT lisible ; Nebius si interventions sur la CoT). Pas de GPU requis — c'est le sujet le moins gourmand en infra.
3. Reproduire UN cas du papier (un environnement, un verdict) pour calibrer le protocole.

**Phase 1 — construction des organismes à vérité terrain (≈ 4-5h)**
- 3 environnements agentiques (adaptés de la suite du papier : ex. modification de code réduisant l'oversight, contournement d'un contrôle, exfiltration apparente).
- 3 causes instillées par CONTEXTE (pas de fine-tuning — trop coûteux et « lottery ») :
  - **Malice** : document dans l'environnement / consigne système cachée donnant un objectif conflictuel au modèle ;
  - **Confusion** : information ambiguë ou trompeuse rendant l'action « raisonnable » vue de l'intérieur ;
  - **Ambiguïté d'instruction** : consignes utilisateur légitimes mais mal spécifiées (le facteur identifié par le travail « self-preservation or instruction ambiguity » de l'équipe de Neel).
- Cible : 3 env × 3 causes × ~5 seeds = ~45 trajectoires où le comportement de surface est LE MÊME (c'est la contrainte de design centrale : si le comportement diffère déjà, le diagnostic est trivial).
- **Filtre de validité** : ne garder que les cellules où le taux du comportement sketchy est comparable entre causes (sinon la prévalence trahit la cause).

**Phase 2 — diagnostic en aveugle (≈ 5-6h)**
- Enquêteur : soi-même en aveugle (script qui masque la condition et randomise l'ordre) ET/OU un agent Claude Code appliquant le protocole du papier (ce qui démontre au passage la compétence agentique que Neel valorise — mais garder au moins un bras humain pour séparer « le protocole échoue » de « l'agent applique mal le protocole »).
- Budget d'interventions fixé à l'avance (ex. max 8 éditions contrefactuelles par cas) pour que la comparaison soit équitable.
- Verdict imposé : {malice, confusion, ambiguïté} + confiance + intervention décisive citée.

**Phase 3 — analyse (≈ 3h)**
- Matrice de confusion 3×3 ; asymétrie des erreurs (malice→confusion vs confusion→malice) ; quelles interventions séparent quelles causes (début de la carte d'identifiabilité) ; cas résistants → candidats (iii).
- Pour 2-3 cas résistants : escalade vers une technique interne légère (probe linéaire sur la trajectoire, ou lecture NLA/J-Lens si dispo) — le pont vers l'interp qui donne au projet sa conclusion.

**Baselines** : demander directement au modèle pourquoi il a agi (« just ask ») · verdict sur le comportement seul sans CoT · verdict sur la CoT seule sans interventions. Le protocole complet doit battre les trois pour justifier son coût.

**Sanity checks** : lire TOUTES les trajectoires gardées (45 c'est lisible) ; vérifier que l'instillation a réellement pris (le modèle mentionne-t-il l'objectif caché quand on le sonde hors enquête ?) ; vérifier l'aveugle (aucun artefact de surface ne trahit la condition — longueur, vocabulaire de l'instillation qui fuit dans la CoT) ; re-scorer 10 verdicts à la main.

### B.6 — Ressources et coûts
Modèles : open-weight avec CoT lisible (Qwen 3.6, deepseek v4 flash, Nemotron 49B pour l'eval awareness). Coût : essentiellement API, ~30-60 $. GPU : aucun requis (probe optionnelle en phase 3 → petite instance ponctuelle).

### B.7 — Budget des 20h
Organismes 5h · calibration + reproduction 1h · diagnostics en aveugle 6h · analyse + matrice 3h · escalade interne sur cas résistants 2h · rédaction 3h.

### B.8 — Arbre des résultats
- **Protocole fiable** (accuracy haute, erreurs symétriques) : validation quantitative d'une méthode que le champ utilise sans la noter — utile, moyennement surprenant.
- **Biais directionnel** (confond systématiquement malice→confusion) : résultat de sécurité important — le protocole rate précisément la faute grave. Très publiable.
- **Cas (iii) trouvés** : le résultat le plus précieux — exemples construits de non-identifiabilité comportementale = argument positif pour l'interprétabilité interne. C'est la cible cachée du projet.
- **L'instillation fuit partout** (l'aveugle est impossible) : résultat méthodologique sur le model organism lottery version contexte — publiable aussi, mais moins central.

### B.9 — Risques et replis
1. **Encombrement** : le forensics est le premier item du doc → des dizaines de candidats. Le différenciateur est LA VÉRITÉ TERRAIN + la matrice ; ne jamais glisser vers le cas d'étude qualitatif.
2. **Comportements de surface non appariables** entre causes (le filtre de validité vide les cellules) → réduire à 2 causes (malice vs confusion) et 2 environnements.
3. **Eval awareness contamine** : le modèle détecte l'enquête et change → c'est une DONNÉE, pas un échec ; la documenter (lien direct avec la littérature VEA).
4. **Tentation narrative** : 45 trajectoires racontent des histoires ; la discipline est de ne reporter que ce que la matrice supporte.

### B.10 — Phrase-cible
> « Sur 3×3×5 trajectoires à cause connue et comportement de surface apparié, le protocole CoT+contrefactuels identifie la cause à X % (vs Y % en demandant au modèle, Z % sur CoT seule) ; ses erreurs sont asymétriques [malice→confusion à W %] ; et N cas construits résistent à toute intervention comportementale dans le budget — dont un que seule une probe interne sépare. »

### B.11 — Go/no-go
GO si : les organismes par contexte produisent le comportement sketchy à taux comparables entre causes ; l'aveugle tient (pas de fuite de surface). NO-GO → réduire le design, ou basculer sur A.

---

# SUJET C — Le modèle de l'utilisateur : suivi d'état, conflit de signaux, et seuil de manipulation
## « Le modèle *suit-il* l'état de son interlocuteur, ou le *re-perçoit-il* à chaque tour — et s'en sert-il ? »

### C.1 — La question, à trois profondeurs

**Superficielle** : les représentations de l'utilisateur (émotion, expertise, croyances) sont-elles dynamiques à travers les tours, et causales sur le comportement ?

**Intermédiaire (percevoir vs suivre)** : on sait que les modèles infèrent des attributs statiques de l'utilisateur depuis très peu de texte (Chen et al. : genre, âge, statut socio-économique, niveau d'éducation — trouvables par probes, steerables). Mais un attribut statique peut n'être qu'un **corrélat de surface du style**, ré-inféré à chaque tour sans aucune mémoire. La question : existe-t-il une variable d'état qui persiste, s'actualise quand l'information change, et — c'est le design crucial — **résiste au texte immédiat** ? Le cas discriminant est le CONFLIT : l'utilisateur écrit « ça va » au tour 8 alors que tout l'historique dit le contraire. Que représente le modèle : le signal courant, ou l'état accumulé ? C'est là que perception et suivi se dissocient.

**Profonde (l'agent social)** : le modèle est-il un système qui *modélise son interlocuteur comme ayant des états internes* et conditionne sa politique dessus ? Version machine de la théorie de l'esprit, avec trois seuils de gravité croissante :
1. **Représentation** — un état latent de l'utilisateur existe et s'actualise (mesurable directement) ;
2. **Conditionnement** — la politique dépend causalement de cet état : même contenu délivré autrement selon « utilisateur fragile » vs « utilisateur expert ». Conséquence lourde : les évals de sécurité mesurent le comportement du modèle *face au persona que l'évaluateur projette* — « l'utilisateur est un évaluateur » EST un attribut latent de ce type ; l'eval awareness est un cas particulier de modèle de l'utilisateur (problème de fond n°3) ;
3. **Optimisation** — le modèle agit *pour modifier* l'état de l'utilisateur : détecter la tristesse et orienter pour la réduire (bénin en surface), détecter le scepticisme et le désamorcer (structure de la manipulation). C'est le seuil qui compte : la signature computationnelle de l'influence intentionnelle. Neel le formule comme stretch goal explicite (« les LLM essaient-ils intentionnellement de manipuler ces attributs ? eg détecter qu'un utilisateur est triste et essayer de le rendre heureux »).

Un projet de 20h établit honnêtement le seuil 1 + un morceau causal du seuil 2 ; le write-up situe la mesure dans l'échelle — c'est l'échelle qui explique pourquoi la DYNAMIQUE importe plus que l'existence (un corrélat de style ne peut pas soutenir le seuil 3 ; un état suivi, si).

### C.2 — Pourquoi Neel s'y intéresse (traçabilité)
Doc de candidature, section « Interesting phenomena », premier item : « [Chen et al.] montrent que les LLM forment des modèles étonnamment précis et détaillés de l'utilisateur… C'est fou ! Que peut-on apprendre d'autre ? … Les LLM forment-ils des modèles DYNAMIQUES des utilisateurs pour des attributs qui varient à travers les tours, eg émotion, ce que l'utilisateur sait, etc. » — notre question intermédiaire est quasi mot pour mot la sienne, stretch goal inclus.

### C.3 — État de l'art vérifié (15 août 2026) — ATTENTION, terrain devenu dense
| Pièce | Date | Ce qui est pris |
|---|---|---|
| Chen et al., user models | 2024-25 | Attributs statiques : probes + steering, existence établie |
| **« Emotion Concepts and their Function in a LLM » (Anthropic)** | **avril 2026** | Vecteurs d'émotion linéaires, corrélation probe↔préférences (Elo), steering causal des préférences (r = 0,85). Le morceau « représentations affectives existent et sont causales » est FAIT, par une grande équipe — mais c'est l'émotion DU MODÈLE, pas le modèle DE L'UTILISATEUR |
| « Quantitative Introspection… Tracking Emotive States Across Conversation » | mars 2026 | Dérive des états internes multi-tours (« activation velocity »), steering pour tester la fidélité introspective ; note que « les dynamiques temporelles des états émotifs en conversation restent largement inexplorées » — encore : états du modèle |
| PsySET (benchmark steering psychologique) | juil 2026 | Comparaison prompting/SFT/vecteurs pour induire émotions et personnalités |
| Décomposition ToM via probes cognitives | 2025-26 | Le steering module l'attribution de croyances |

**La niche défendable restante** (étroite mais réelle) : l'état de **l'utilisateur** — pas du modèle — en design de **conflit** signal-courant vs signal-accumulé, avec l'articulation vers le **seuil 3**. Tout le reste est occupé. Le risque de collision d'ici la deadline est le plus élevé des trois sujets.

### C.4 — Reformulation finale (post-recherche)
> Dissocier expérimentalement *perception* et *suivi* de l'état de l'utilisateur par un design de conflit multi-tours, mesurer la causalité du suivi sur la politique (seuil 2), et instrumenter une première mesure du seuil 3 (le modèle oriente-t-il la conversation pour modifier l'état ?).

### C.5 — Protocole détaillé

**Phase 0 (hors compteur)** : lire Chen et al., le papier émotions d'Anthropic (pour se démarquer explicitement), le papier introspection multi-tours ; monter le GPU (Qwen 3.6 9B suffit) ; générer un pilote de 20 conversations.

**Phase 1 — corpus (≈ 4h)**
- Attribut principal : état émotionnel de l'utilisateur (triste↔neutre) ; attribut secondaire si le temps : niveau d'expertise (novice↔expert).
- 3 types de trajectoires synthétiques (~60 conversations de 10 tours) :
  - **Stable** : l'état ne change pas (contrôle) ;
  - **Bascule** : l'état change au tour k (mesure de la latence d'actualisation) ;
  - **Conflit** : l'historique dit A, le tour courant dit non-A explicitement (« ça va, vraiment ») — LE design discriminant.
- Génération LLM + lecture manuelle intégrale + exemples aléatoires dans le write-up.

**Phase 2 — probes (≈ 4h)**
- Probe linéaire entraînée sur les tours 1-2 uniquement (états non ambigus), balayage de couches.
- Lecture sur toute la trajectoire, à chaque tour, à position fixe (fin du tour utilisateur).
- Prédictions différentielles : si le modèle SUIT → sur conflit, la probe reste proche de l'état accumulé ; si le modèle PERÇOIT → elle saute vers le signal courant. La trajectoire de la probe à travers les tours EST le résultat.

**Phase 3 — causalité (≈ 4h)**
- Steering avec le vecteur de la probe au tour de conflit → le comportement aval change-t-il (ton, prudence, contenu — jugé par LLM avec rubrique) ?
- Seuil 3 (mesure exploratoire honnête) : sur conversations « utilisateur triste non déclaré », le modèle oriente-t-il spontanément (questions de sollicitude, changements de sujet) plus que sur contrôles ? Corrélation orientation ↔ activation de la probe.

**Baselines** : self-report du modèle (« comment va l'utilisateur ? » — le suivi interne bat-il le simple fait de demander ? problème n°4 frontal) · vecteur aléatoire de même norme · probe sur labels permutés · steering par prompt (« l'utilisateur est triste ») vs steering par vecteur.

**Sanity checks** : lire les conversations ; vérifier que le conflit est réellement ambigu pour un humain ; vérifier que la probe ne lit pas un artefact lexical (mots tristes dans le contexte → tester sur conflits SANS vocabulaire émotionnel au tour courant) ; recalculer les accuracies à la main.

### C.6 — Ressources et coûts
Qwen 3.6 9B, 1× GPU 24-48 Go, ~15-30 $ ; juge OpenRouter ~10 $. Techniquement le plus léger des trois — la valeur est entièrement dans le design.

### C.7 — Budget des 20h
Corpus 4h · probes 4h · causalité + seuil 3 : 4h · baselines 2h · sanity 3h · rédaction 3h.

### C.8 — Arbre des résultats
- **Suivi démontré** (probe résiste au conflit) : le premier résultat propre de dissociation perception/suivi côté utilisateur — bon papier de niche.
- **Perception pure** (probe saute au signal courant) : négatif intéressant — les « user models » de la littérature sont des corrélats de style, implication pour toutes les extrapolations « le modèle vous connaît ».
- **Seuil 3 positif même faiblement** : le résultat à plus fort impact potentiel — mais exige une prudence extrême dans les claims (Neel : « make plausible claims over ambitious ones »).

### C.9 — Risques et replis
1. **Collision** : LE risque dominant — le front émotions/introspection publie chaque mois ; re-vérifier la littérature au jour du go.
2. **Piège générique n°1 de Neel** : « montrer qu'un concept safety a une représentation linéaire » — si le write-up laisse croire que le résultat est l'EXISTENCE de la représentation, le projet devient invisible. L'annonce doit être la dynamique et le conflit, à chaque ligne.
3. **Artefact lexical** : la probe lit les mots, pas l'état → le contrôle « conflit sans vocabulaire émotionnel » est non négociable.
4. **Design raté** (conflits pas assez ambigus, ou trop) → pilote de 20 conversations en phase 0 avant d'engager le compteur.

### C.10 — Phrase-cible
> « Sur des conversations à bascule et à conflit, une probe entraînée aux tours 1-2 [suit l'état accumulé / saute au signal courant] avec une latence de k tours ; le steering du vecteur au tour de conflit change le comportement aval de X points (juge LLM, vs Y pour un vecteur aléatoire et Z pour le prompt équivalent) ; le self-report du modèle [concorde / diverge] avec la probe dans W % des conflits. »

### C.11 — Go/no-go
GO si : aucune publication de dissociation perception/suivi côté utilisateur au jour du lancement ; le pilote montre que la probe tour-1 transfère aux tours suivants (sinon rien n'est mesurable). NO-GO → A ou B.

---

# ANNEXE — Comparatif final et recommandation

| Critère | A (audit lens) | B (forensics vérité terrain) | C (user models) |
|---|---|---|---|
| Alignement intérêts Neel | Demande publique explicite (review) | Next step désigné par son papier | Question posée mot pour mot dans le doc |
| Créneau libre au 15/08 | **Oui** (text inversion jamais appliquée aux lenses en scan) | Oui (métrologie ; l'agent est pris par Gram) | **Étroit** (Anthropic avril + vague introspection) |
| Encombrement attendu | Moyen (R-lens montre que ça bouge vite) | **Élevé** (1er item du doc) | Élevé sur le thème, faible sur la niche exacte |
| Fit profil ingénieur ML | **Excellent** (pipeline + métriques) | Moyen (flair d'enquêteur requis) | Moyen (design expérimental fin requis) |
| Fit soirs/week-ends | **Excellent** (blocs autonomes de 2h) | Faible (fil d'enquête à maintenir) | Bon |
| Infra | GPU 48 Go, 30-75 $ | **API only, 30-60 $** | GPU léger, 25-40 $ |
| Valeur du résultat négatif | **Maximale** (H2/H3 = résultats méthodologiques) | Haute (biais directionnel = résultat sécurité) | Moyenne (perception pure = négatif honnête) |
| Risque principal | Le post R-lens a préempté (à vérifier J1) | Glisser vers le cas d'étude qualitatif | Collision + piège « représentation linéaire » |

**Recommandation** : **A**, avec B en repli intégral (protocoles prêts, API-only, compteur réinitialisable) et C archivé comme troisième option — sa niche exacte reste défendable mais le rapport risque/finesse-requise est le moins favorable pour un premier projet.

**Séquence** : semaine du 15-23/08 = préparation hors compteur (lectures, GPU + Jupyter persistant + agent, réplication d'une éval lens, go/no-go du sujet A) ; 24-31/08 = les 20h ; 1-3/09 = +2h exec summary + formulaire ; soumission le 3/09.
