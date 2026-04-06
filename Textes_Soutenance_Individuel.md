# Textes de Soutenance Finale - Groupe 553
## Detection Automatisee d'Exoplanetes par Analyse de Courbe de Lumiere

Duree totale visee : 20 minutes (environ 3min20 par personne)

Conseil general : parlez lentement, articulez, regardez le jury. Ne lisez pas mot pour mot, appropriez-vous le texte.

---

## SIMON GALLAIS - Contexte, Enjeux & Problematique (Slides 1 a 4)
**Duree : ~3min20**

**[SLIDE 1 - Titre]**

Bonjour a tous. Nous sommes le groupe 553 et nous allons vous presenter notre projet de detection automatisee d'exoplanetes par analyse de courbes de lumiere. Notre equipe est composee de six membres que vous voyez ici. Je vais commencer par vous presenter le contexte de notre projet.

**[SLIDE 2 - Sommaire]**

Voici le deroulement de notre presentation. Nous allons d'abord poser le contexte et la problematique, puis Oscar vous presentera notre etat de l'art et nos choix techniques. Mathis detaillera l'architecture, Charles les resultats, Mederic fera une demonstration live, et Kamil terminera avec l'analyse critique et les perspectives.

**[SLIDE 3 - Contexte & Enjeux]**

Alors, pourquoi ce sujet ? Aujourd'hui, plus de 5 600 exoplanetes ont ete confirmees. C'est un chiffre impressionnant, mais il cache une realite : des millions de courbes de lumiere restent a analyser. Les missions spatiales Kepler et TESS observent des centaines de milliers d'etoiles et produisent des teraoctets de donnees photometriques. C'est un veritable probleme de Big Data astronomique.

L'analyse humaine a atteint ses limites. Un astronome peut prendre des mois pour analyser une seule etoile en detail. Avec l'intelligence artificielle, on peut traiter des milliers d'etoiles en quelques heures. C'est la que notre projet prend tout son sens.

**[SLIDE 4 - Problematique]**

Notre problematique est la suivante : comment automatiser la detection et la caracterisation d'exoplanetes a partir de courbes de lumiere bruitees, avec un modele qui soit explicable et accessible ?

Le verrou technique principal, c'est la distinction entre un vrai transit planetaire et un faux positif. Quand une planete passe devant son etoile, elle provoque une baisse de luminosite. Mais cette meme baisse peut etre causee par d'autres phenomenes : des etoiles binaires a eclipses, du bruit instrumental, ou des variations stellaires naturelles.

Notre opportunite, c'est de construire un pipeline completement automatise, du telechargement des donnees jusqu'a la visualisation des resultats, avec un modele explicable qui fonctionne sur une machine standard. Je passe maintenant la parole a Oscar.

---

## OSCAR SCHWARTZ - Etat de l'Art & Choix Technique (Slide 5)
**Duree : ~3min20**

**[SLIDE 5 - Etat de l'Art & Notre Pivot]**

Merci Simon. Avant de choisir notre approche, nous avons etudie ce qui se fait dans la litterature scientifique. Les travaux de reference, comme ceux de Shallue et Vanderburg a la NASA en 2018, utilisent du Deep Learning, principalement des reseaux de neurones convolutifs, les CNN. Ces modeles atteignent environ 96% d'AUC sur les donnees Kepler.

Le probleme, c'est que si on reproduit exactement la meme methode, on fait la meme chose en moins bien. On n'a pas les GPU de la NASA, ni leurs equipes de chercheurs. C'est pour ca que nous avons fait un pivot strategique.

Nous avons choisi le Machine Learning classique avec XGBoost, couple a du feature engineering avance grace a TSFRESH. Comme vous pouvez le voir dans ce tableau comparatif, notre approche presente plusieurs avantages concrets.

En termes d'explicabilite, le Deep Learning est une boite noire, alors que XGBoost nous donne la contribution exacte de chaque feature dans la decision. Pour les ressources de calcul, le Deep Learning necessite un GPU, nous on tourne sur un CPU standard. Et le resultat : notre modele atteint 98.1% d'AUC-ROC, ce qui est comparable voire superieur aux approches par Deep Learning.

Notre pivot n'est donc pas un choix par defaut, c'est un choix strategique : proposer une alternative explicable, legere, et performante. Un astronome peut comprendre pourquoi le modele a pris sa decision, ce qui est essentiel pour la validation scientifique. Je laisse la parole a Mathis pour l'architecture.

---

## MATHIS LEITAO - Architecture & Technologies (Slides 6 et 7)
**Duree : ~3min20**

**[SLIDE 6 - Justification des Technologies]**

Merci Oscar. Je vais maintenant vous presenter notre stack technique et justifier chacun de nos choix.

Pour le moteur d'IA, XGBoost est un algorithme de gradient boosting qui excelle sur les donnees tabulaires. Couple a TSFRESH, qui extrait automatiquement plus de 700 features statistiques sur des series temporelles, on elimine le biais humain dans la selection des features.

Cote backend, nous avons choisi Flask en Python. C'est un framework leger pour construire des API REST, et surtout il est parfaitement coherent avec l'ecosysteme scientifique Python : NumPy, Pandas, Lightkurve, Astropy. Lightkurve et Astropy sont les librairies de reference de la NASA pour acceder aux donnees Kepler et TESS via l'API MAST.

Pour le frontend, nous avons choisi React avec Vite. Pourquoi React plutot que Vue ou Angular ? Principalement pour son ecosysteme de visualisation : Three.js pour les orbites 3D, Recharts pour les graphiques interactifs. Vite nous apporte un hot reload instantane pendant le developpement.

Enfin, la securite : authentification JWT avec hachage bcrypt des mots de passe cote serveur.

**[SLIDE 7 - Architecture du Systeme]**

Voici l'architecture globale de notre systeme en trois blocs. A gauche, le frontend React avec ses 6 onglets : visualisation des courbes, orbites 3D, upload CSV, catalogue, metriques du modele, et un glossaire de 50 termes astronomiques.

Au centre, le backend Flask avec 21 endpoints API, l'authentification JWT, le pipeline ML, et un cache de 2 688 fichiers de courbes de lumiere pour optimiser les performances.

A droite, le moteur d'intelligence artificielle : XGBoost avec 32 features extraites, validation croisee 5-fold, et caracterisation physique des candidats.

Le tout s'appuie sur 9 961 echantillons reels provenant des catalogues Kepler KOI et TESS TOI de la NASA. Je passe la parole a Charles.

---

## CHARLES DE BLAUWE - Pipeline & Resultats (Slides 8, 9 et 10)
**Duree : ~3min20**

**[SLIDE 8 - Pipeline de Traitement]**

Merci Mathis. Je vais vous montrer comment fonctionne concretement notre pipeline, etape par etape.

Premiere etape : l'acquisition. On telecharge les courbes de lumiere brutes via l'API MAST de la NASA et la librairie Lightkurve. Ca prend entre 5 et 15 secondes par etoile, principalement du temps reseau.

Deuxieme etape : le nettoyage. On supprime les outliers, on applique un detrending pour retirer les tendances a long terme, et on fait du binning temporel pour reduire le bruit.

Troisieme etape, et c'est la plus importante : le folding. On replie la courbe de lumiere sur la periode orbitale connue. C'est cette etape qui a fait passer notre modele de 57% a 92% d'accuracy. Sans le folding, le signal de transit est dilue sur toute la serie temporelle et les features ne peuvent pas distinguer les classes.

Quatrieme etape : l'extraction de 32 features statistiques et physiques, incluant des parametres orbitaux, planetaires, stellaires et spatiaux.

Cinquieme etape : la prediction par XGBoost avec une caracterisation physique du candidat. Le tout prend entre 13 et 38 secondes par etoile.

**[SLIDE 9 - Resultats du Modele]**

Passons aux resultats. Notre modele atteint 92.6% d'accuracy sur l'ensemble de test de 1 993 echantillons. L'AUC-ROC est de 98.1%, ce qui signifie une excellente capacite de discrimination. La precision est de 90.5% et le recall de 91.3%.

La matrice de confusion montre 1 204 vrais negatifs et 654 vrais positifs, avec seulement 69 faux positifs et 66 faux negatifs. En validation croisee 5-fold, on obtient 92.6% plus ou moins 0.73%, ce qui confirme la stabilite du modele.

Nous avons aussi valide sur 18 exoplanetes reelles celebres : 15 sur 18 correctement identifiees, soit 83.3%. Les trois echecs concernent des systemes multi-planetes comme Kepler-90 et un cas de bruit stellaire atypique.

**[SLIDE 10 - Feature Importance]**

Ce graphique montre les features les plus importantes pour le modele. En tete, les incertitudes sur la temperature stellaire et la duree du transit. C'est interessant car cela signifie que le modele a appris que les faux positifs presentent des mesures plus incertaines. Le rayon planetaire est la troisieme feature la plus importante.

C'est la tout l'avantage par rapport au Deep Learning : chaque prediction est explicable et auditable par un astronome. Place a Mederic pour la demonstration.

---

## MEDERIC ROLLAND - Demonstration Live (Slide 11)
**Duree : ~3min20**

**[SLIDE 11 - Demo Live]**

Merci Charles. Plutot que de vous montrer des slides supplementaires, on va vous faire une demonstration en direct de notre application.

**[BASCULER SUR LA DEMO - Ouvrir le navigateur]**

Voici notre interface web. Je commence par me connecter avec un compte utilisateur.

Une fois connecte, vous voyez le dashboard principal. Je vais analyser une etoile du catalogue Kepler. Je selectionne par exemple Kepler-22, une etoile connue pour heberger une exoplanete dans la zone habitable.

L'application lance le pipeline en temps reel. Vous pouvez voir la barre de progression qui indique chaque etape : telechargement, nettoyage, folding, extraction de features, prediction.

Voici le resultat. On voit la courbe de lumiere avec le transit en forme de U characteristique. Le modele attribue un score de confiance eleve, confirmant la presence d'une exoplanete. On peut aussi voir les parametres physiques estimes : rayon planetaire, duree du transit, et la classification du type de planete.

Dans l'onglet Catalogue, on peut explorer toutes les etoiles analysees, filtrer par score de confiance, et comparer les resultats.

L'onglet Metriques montre les performances globales du modele, et l'onglet Documentation contient un glossaire interactif de plus de 50 termes astronomiques pour rendre l'outil accessible meme aux non-specialistes.

**[REVENIR SUR LES SLIDES]**

Je laisse la parole a Kamil.

---

## KAMIL BENJELLOUN - Analyse Critique, Gestion de Projet & Conclusion (Slides 12 a 15)
**Duree : ~3min20**

**[SLIDE 12 - Analyse Critique & Limites]**

Merci Mederic. Parlons maintenant de nos difficultes et de nos limites, parce qu'un bon projet c'est aussi savoir reconnaitre ce qui n'a pas marche.

Premiere difficulte majeure : notre pipeline initial etait defaillant. Nous faisions l'extraction de features sur des courbes non repliees, ce qui donnait seulement 57% d'accuracy. Le modele ne pouvait pas distinguer les classes. C'est le folding sur la periode orbitale qui a tout change.

Deuxieme difficulte : le calcul du periodogramme BLS sur les longues courbes Kepler de 60 000 points prenait des heures. On l'a finalement supprime car les periodes du catalogue NASA sont deja validees.

Troisieme difficulte : notre dataset initial ne contenait que 28 etoiles codees en dur. On l'a elargi a 9 961 echantillons reels.

Cote limites actuelles : le modele est moins performant sur les systemes multi-planetes comme Kepler-90, il ne produit pas d'intervalle de confiance autour de ses predictions, et il depend des periodes orbitales du catalogue NASA.

**[SLIDE 13 - Gestion de Projet]**

Notre projet s'est deroule en quatre phases de septembre a avril. La phase d'analyse et cadrage, la conception du pipeline, le developpement complet, et la validation actuelle.

En retrospective, notre erreur principale a ete de commencer le developpement du modele ML avant d'avoir stabilise le pipeline de donnees. Le detrending et le folding auraient du etre valides avant tout entrainement. Cette approche iterative, avec validation a chaque etape, nous a permis de passer de 57% a 92.6% d'accuracy.

**[SLIDE 14 - Perspectives]**

Pour la suite, trois axes d'amelioration. Ameliorer le modele en ajoutant des features TSFRESH sur les courbes repliees et en l'entrainant specifiquement sur les systemes multi-planetes. Etendre les donnees avec les observations du telescope JWST. Et enfin, publier l'outil en open-source pour democratiser la recherche exoplanetaire aupres des astronomes amateurs.

**[SLIDE 15 - Conclusion]**

Pour conclure, nous avons developpe un pipeline complet de detection d'exoplanetes, de l'acquisition des donnees jusqu'a la visualisation interactive. Notre modele atteint 92.6% d'accuracy et 98.1% d'AUC-ROC, avec 83.3% de reussite sur 18 exoplanetes reelles. Et surtout, c'est une alternative explicable et accessible au Deep Learning.

Merci de votre attention. Nous sommes prets pour vos questions.
