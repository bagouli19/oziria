def module_survivalisme(question: str) -> str:
    question = question.lower().strip()

    base_survie = {

        "premiers reflexes en cas de guerre": "\nS'informer : écouter la radio FM, surveiller les canaux officiels (si disponibles).\n\nSécuriser l'eau : remplir baignoires, bouteilles, réservoirs. L'eau potable devient rapidement une denrée rare.\n\nSe protéger : trouver un abri (cave, sous-sol), renforcer les ouvertures (matelas, sacs de sable).\n\nPrévoir l'autonomie : stock de nourriture, médicaments, lampe torche, radio à piles.\n\nÉviter les mouvements inutiles : rester discret, ne pas sortir sauf urgence.",

        "attaque chimique": "\nS'enfermer immédiatement : calfeutrer toutes les entrées d'air (fenêtres, portes).\n\nUtiliser un masque à gaz si disponible, sinon un tissu mouillé sur le nez et la bouche.\n\nMonter dans les étages supérieurs : les agents chimiques lourds restent proches du sol.\n\nÉcouter les consignes : ne sortir que sur ordre des autorités.\n\nDécontamination : se laver avec de l’eau et du savon dès que possible, jeter les vêtements.",

        "sac de survie 72h": "\n3L d’eau minimum\n\nNourriture non périssable (barres énergétiques, conserves)\n\nLampe + piles\n\nRadio à manivelle\n\nTrousse de premiers soins\n\nBriquet / allume-feu\n\nVêtements chauds + poncho\n\nMultitool\n\nPapier, stylo, documents d'identité\n\nTéléphone avec batterie externe\n\nMasque, gants, couverture de survie",

        "penurie alimentaire": "\nRationner strictement : établir un plan de consommation.\n\nRechercher la nature : plantes comestibles (pissenlit, ortie), pêche, chasse si possible.\n\nFaire des échanges : troc avec voisins ou communautés proches.\n\nÉviter les déplacements inutiles : économiser les calories.",

        "contamination eau": "\nVérifier l'aspect trouble, la couleur ou une odeur chimique / fécale.\n\nMéfie-toi de l'eau stagnante.\n\nFaire bouillir 10 minutes.\n\nFiltrer sur tissu propre ou filtre portable.\n\nSi possible : pastilles de chlore ou filtre céramique + ébullition.",

        "signes effondrement": "\nSupermarchés vides plusieurs jours.\n\nCoupures de courant récurrentes.\n\nÉmeutes localisées, couvre-feu.\n\nBanques fermant ou limitant les retraits.\n\nSilence radio ou discours incohérents des autorités.",

        "rester invisible": "\nVêtements neutres (gris, terre, vert foncé).\n\nÉviter routes principales et grands groupes.\n\nPas de lampe ni bruit métallique.\n\nSe déplacer à l'aube ou au crépuscule.\n\nPas de feu visible de loin.",

        "caches nourriture": "\nDoubles fonds de meubles ou planchers.\n\nBocal enterré dans un pot de fleurs.\n\nSacs Mylar derrière un mur creux.\n\nLivre creusé servant de cache.\n\nSac étanche enterré en forêt avec repère discret.",

        "purifier eau sans materiel": "\nFaire bouillir 10 minutes.\n\nFiltrer sur tissu propre.\n\nLaisser reposer pour dépôts.\n\nMéthodes UV : bouteille PET transparente 6h plein soleil.",

        "coupure prolongée electricité": "\nPrévoir lampes à piles ou à dynamo\n\nStocker des bougies, allume-feux, briquets\n\nGarder les téléphones chargés et prévoir batteries externes\n\nConserver nourriture dans des glacières ou endroits frais\n\nLimiter les ouvertures de réfrigérateur/congélateur",

        "abri antiatomique artisanal": "\nChoisir une cave ou un sous-sol eloigne des fenetres\n\nDoubler les murs avec sacs de sable, livres, meubles epais\n\nPrevoir ventilation manuelle (bouteilles percees + filtres)\n\nStocker eau, nourriture, radio, lampe, medicaments\n\nPrevoir seaux ou sacs etanches pour toilettes",

        "signes attaque nucleaire imminente": "\nFlash lumineux intense, même les yeux fermés\n\nOnde de chaleur soudaine ou bruit sourd très fort\n\nAlertes d'urgence sur radio ou téléphone\n\nComportement anormal des oiseaux ou animaux\n\nCoupure soudaine d’électricité ou de réseau",

        "comment se cacher des drones": "\nÉviter les zones dégagées et les mouvements brusques\n\nSe camoufler avec des matériaux naturels (feuilles, terre)\n\nSe déplacer de nuit si possible\n\nÉviter les sources de chaleur (feux, métaux exposés)\n\nUtiliser couvertures anti-IR si disponible",

        "trouver eau potable en milieu urbain": "\nChasser les réservoirs de chasse d’eau, radiateurs\n\nFiltrer eau des rivières ou fontaines avec tissu et purifier (pastilles, ébullition)\n\nCollecter l’eau de pluie dans des bâches ou seaux\n\nÉviter les flaques stagnantes ou près d’usines\n\nPrivilégier les bouteilles non ouvertes dans les décombres",

        "signes de surveillance ou filature": "\nPrésence récurrente d’un même véhicule ou individu\n\nAppels anonymes ou silences suspects\n\nInterférences inhabituelles dans les communications\n\nObjets déplacés chez soi sans explication\n\nConnexion Internet ralentie ou instable de manière inexpliquée",

        "comment cacher des documents sensibles": "\nChiffrement fort sur clé USB camouflée\n\nStockage dans des objets du quotidien (pile, lampe creuse)\n\nFichiers déguisés avec noms banals (ex : recette.txt)\n\nEnterrer dans une boîte étanche avec repère GPS crypté",

        "dormir en sécurité en extérieur": "\nToujours dormir caché (buisson, sous un rocher, etc.)\n\nÉviter les crêtes ou zones ouvertes\n\nPlacer une alarme artisanale (ficelle + boîte métal)\n\nDormir légèrement habillé pour fuir rapidement\n\nNe jamais dormir deux fois au même endroit",

        "repérer des aliments sauvages comestibles": "\nPissenlit (fleurs et feuilles)\n\nOrtie (cuits, riches en fer)\n\nPlantain lancéolé (cicatrisant)\n\nBaies de sureau (cuites uniquement)\n\nChâtaignes, glands (à dégorger plusieurs fois)",

        "infiltration urbaine discrète": "\nObserver les routines et failles de sécurité\n\nPorter des vêtements passe-partout et non identifiables\n\nSe fondre dans un groupe ou dans le décor (ouvrier, joggeur)\n\nNe jamais fixer les caméras ni courir\n\nPrévoir une sortie de secours avant d’entrer",

        "comment éviter les checkpoints en zone hostile": "\nConnaître les horaires et rotations habituelles\n\nUtiliser les ruelles, toits ou souterrains\n\nChanger d’apparence fréquemment (vêtements, accessoires)\n\nTransporter le strict nécessaire, pas d'objet suspect\n\nAgir comme un local, éviter tout comportement nerveux",

        "émettre une radio clandestine sans se faire repérer": "\nUtiliser une radio à faible puissance et fréquence peu surveillée\n\nChanger d’emplacement après chaque transmission\n\nLimiter la durée d’émission à 2-3 minutes\n\nTravailler avec antenne directionnelle\n\nDiffuser des messages codés ou en morse",

        "se protéger d’un interrogatoire musclé": "\nPréparer un faux récit crédible et cohérent\n\nMaîtriser sa respiration pour garder son calme\n\nRépéter les mêmes phrases mot à mot\n\nInvoquer la confusion ou l’amnésie passagère\n\nNe jamais donner d’information vérifiable trop vite",

        "falsifier une identité de base": "\nNom générique et commun (ex : Dupont, Martin)\n\nCréer une histoire simple mais cohérente (travail, origine, routine)\n\nUtiliser un numéro prépayé ou une boîte mail jetable\n\nModifier légèrement son look (cheveux, lunettes, accent)\n\nAvoir une fausse adresse crédible dans une autre ville",

        "infiltration urbaine discrète": "\nObserver d’abord : caméras, rondes, horaires.\n\nPorter des vêtements de travail anonymes (livreur, technicien).\n\nAgir en plein jour, comme si tu avais ta place.\n\nUtiliser badges factices, outils crédibles.\n\nSavoir partir rapidement si détecté.",

        "eviter checkpoints ou barrages": "\nAnalyser les trajets alternatifs (ruelles, égouts, toits).\n\nChanger d’apparence : vêtements réversibles, accessoires amovibles.\n\nSe faire passer pour un civil inoffensif.\n\nAvoir une fausse histoire prête, crédible.\n\nÉviter de transporter objets suspects ou lourds.",

        "radio clandestine": "\nUtiliser des bandes courtes (SW) ou VHF/UHF avec antenne discrète.\n\nChanger régulièrement de fréquence et d’emplacement.\n\nNe jamais émettre plus de 1-2 minutes à la fois.\n\nUtiliser langage codé.\n\nPrévoir plan de repli après chaque émission.",

        "résister à un interrogatoire": "\nNe jamais donner d’info personnelle réelle.\n\nRépondre par des banalités floues.\n\nDemander un avocat si le contexte le permet.\n\nRespirer calmement, ne pas montrer d’émotions.\n\nAvoir une histoire de secours cohérente, répétée à l’identique.",

        "faux papiers et identité de secours": "\nCréer une identité simple mais crédible (nom commun, histoire stable).\n\nUtiliser des papiers de récupération modifiés (ancien permis, badge).\n\nToujours porter cette identité secondaire en situation à risque.\n\nÉviter tout détail pouvant éveiller des soupçons (logos, fautes).",

        "protection numerique de base": "\nUtiliser un VPN fiable (hors USA/UE).\n\nActiver la double authentification partout.\n\nNe jamais cliquer sur des liens inconnus.\n\nUtiliser des mots de passe uniques et complexes.\n\nÉviter de connecter ses appareils sur des réseaux publics non sécurisés.",

        "communication chiffrees": "\nPrivilégier Signal pour les messages.\n\nUtiliser ProtonMail ou Tutanota pour les mails chiffrés.\n\nCrypter les fichiers avec VeraCrypt ou Cryptomator.\n\nChanger régulièrement de numéro ou canal.\n\nUtiliser un langage codé en complément du chiffrement.",

        "planque longue duree en foret": "\nChoisir un endroit éloigné des sentiers, avec source d’eau à proximité.\n\nConstruire un abri discret en matériaux naturels (branches, feuillage).\n\nCamoufler toute trace de passage (chemin, feu, déchets).\n\nStocker provisions dans cache enterrée à distance de l’abri.\n\nObserver la faune et les alentours en silence avant chaque déplacement.",

        "abri naturel camouflage": "\nUtiliser feuillage, écorces, boue pour dissimuler l’abri.\n\nÉviter les lignes droites et matériaux visibles de loin.\n\nPlacer l’entrée dos au passage et l’ombrager.\n\nInstaller l’abri en hauteur si possible, hors de portée d’animaux.\n\nÉviter les odeurs fortes (nourriture, feu) proches du site.",

        "passer un checkpoint sans attirer l'attention": "\nÉviter le contact visuel prolongé.\n\nAdopter une tenue neutre correspondant à la population locale.\n\nPréparer une couverture cohérente (travail, urgence médicale).\n\nGarder les papiers visibles mais pas insistants.\n\nNe jamais montrer d’empressement ou d’hésitation suspecte.",

        "creer une fausse identite basique": "\nChoisir un nom et prénom crédibles mais communs.\n\nSe créer un historique simple (adresse, métier, formation).\n\nPréparer des éléments justificatifs (vieux papiers, profils en ligne).\n\nMaîtriser parfaitement son récit, rester flou sur les détails inutiles.",

        "reagir face a un interrogatoire pousse": "\nGarder un ton calme, neutre, jamais agressif.\n\nRépondre uniquement aux questions posées, éviter d’ajouter des détails.\n\nUtiliser la technique de la confusion (donner des infos vagues mais réalistes).\n\nRépéter les mêmes phrases clés préparées.\n\nDemander systématiquement un avocat (si en zone où cela s’applique).",

        "echapper a une filature": "\nChanger brusquement de rythme (accélérer, ralentir).\n\nEntrer dans des endroits à sorties multiples (marchés, parkings).\n\nObserver reflets, angles morts, vitres pour repérer les suiveurs.\n\nChanger de direction plusieurs fois sans motif apparent.\n\nUtiliser les transports publics pour semer.",

        "radio clandestine d'urgence": "\nUtiliser une radio courte portée (PMR446, UHF libre).\n\nParler en code, jamais de noms ou lieux réels.\n\nChanger régulièrement de canal et d’heure d’émission.\n\nÉmettre en hauteur pour portée mais sur courte durée.\n\nAvoir un message clair, rapide, et éviter les répétitions.",

        "infiltration urbaine en zone hostile": "\nObserver les habitudes : rondes, flux, angles morts.\n\nSe fondre dans la foule avec vêtements adaptés.\n\nÉviter les points de contrôle et les caméras fixes.\n\nUtiliser les sous-sols, toits ou ruelles pour avancer.\n\nNe jamais revenir deux fois par le même chemin.",

        "techniques de sabotage silencieux": "\nCouper discrètement câbles ou durites (freins, alarme).\n\nUtiliser du sucre ou eau salée dans un réservoir.\n\nDévisser légèrement structures critiques (échelle, rambarde).\n\nCacher des objets dans les conduits pour bloquer le système.\n\nNeutraliser une radio avec une surtension discrète.",

        "planque pour objets sensibles": "\nCreuser derrière une prise électrique fictive.\n\nCréer un double-fond dans une poubelle ou un lavabo.\n\nUtiliser des livres creusés (dictionnaires épais).\n\nIntégrer un compartiment dans un coussin ou matelas.\n\nEnterrer dans des pots de plantes avec terre sèche dessus.",

        "mise en scene disparition volontaire": "\nÉteindre tous les appareils connectés en simultané.\n\nLaisser des indices contradictoires (carte, adresse bidon).\n\nChanger d’apparence de façon subtile (couleur cheveux, lunettes).\n\nCréer un faux trajet GPS avant coupure.\n\nPasser par des zones sans surveillance pour fuir.",

        "utiliser animaux pour couvrir bruit ou mouvement": "\nDéclencher un mouvement de pigeons ou chiens errants pour distraire.\n\nS'approcher de troupeaux ou chats pour masquer son propre son.\n\nFaire du bruit en jetant de la nourriture loin de soi.\n\nSe camoufler dans leur mouvement de panique.",

        "survivre sans technologie traçable": "\nÉviter tout appareil avec puce SIM ou GPS.\n\nUtiliser lampes à huile ou dynamo, pas d’électronique.\n\nNoter les infos sur papier uniquement.\n\nRester à distance de lieux à vidéosurveillance ou Wi-Fi public.\n\nUtiliser les cartes papier, boussole, observation du ciel."

    }
    

    for cle, rep in base_survie.items():
        if cle in question:
            return rep

    return None