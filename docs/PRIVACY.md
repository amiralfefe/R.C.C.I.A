# Confidentialite du demonstrateur

Cette notice decrit le comportement applicatif inspecte le 20 septembre 2026.
Elle ne constitue pas une certification RGPD ni une autorisation d'utiliser des
donnees de sante.

## Donnees traitees

- L'image chargee, son nom de fichier, les probabilites et la visualisation
  Grad-CAM sont traites sur le serveur Streamlit, pas uniquement sur l'appareil.
- Le code du hub conserve temporairement l'image et le resultat dans la session.
  Il n'enregistre pas ces uploads dans les datasets ou les outputs et ne les
  reutilise pas pour entrainer les modeles.
- Changer ou retirer l'image efface le resultat associe dans le hub. La liberation
  physique de la memoire depend du runtime et du cycle de vie de la session :
  aucune suppression instantanee ou duree de conservation precise n'est garantie.
- Les scripts CLI locaux peuvent ecrire des rapports et des exemples dans
  outputs/. Ce comportement est distinct des uploads du hub public.

## Services tiers

Streamlit Community Cloud heberge l'application. Hugging Face distribue les
checkpoints au serveur ; le code ne lui envoie pas les images chargees.
L'hebergeur peut traiter des journaux techniques et des donnees de connexion.
Leurs durees de conservation, contrats, cookies et transferts n'ont pas ete
verifies dans cet audit.

La telemetrie optionnelle Streamlit est desactivee dans la configuration locale
du depot. Cela ne controle pas les journaux ou cookies de la plateforme Cloud,
et n'est effectif en ligne qu'apres publication de cette configuration.
Aucun analytics marketing personnalise n'a ete identifie dans le code.
Aucune banniere de consentement fictive n'a ete ajoutee.

## Usage autorise pour cette demonstration

Utiliser uniquement des images publiques de demonstration, sans identifiants
personnels ni donnees medicales privees. Ne pas transmettre de dossier patient.
Ce projet est educatif / portfolio uniquement, sans diagnostic ni validation clinique.
Avant tout usage avec de vraies donnees personnelles, l'editeur doit definir les
responsabilites, la base du traitement, les informations de contact et les
obligations applicables avec une revue qualifiee.
