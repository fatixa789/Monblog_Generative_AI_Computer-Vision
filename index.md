 # L'IA qui imagine — Generative AI en Computer Vision

> *Par [Fatima-zahra akeznanay] — Avril 2026*  
> *Cours : Computer Vision , S8*

---

Ok donc ce semestre on a pas mal parlé de CNN, de transformers, de tout ce qui permet à une machine de **voir** et de **comprendre** une image. Mais là on va parler de quelque chose encore plus fou : des machines qui **créent** des images.

Des images qui n'existent pas. Des visages de personnes qui ne sont jamais nées. Des paysages que personne n'a jamais photographiés. Et pourtant... ça a l'air 100% réel.

C'est ça le **Generative AI en Computer Vision**. Et dans ce post, je vais tout t'expliquer simplement — comme si t'avais jamais entendu parler de tout ça.

---

## Table des matières

1. [C'est quoi exactement ?](#cest-quoi)
2. [Pourquoi c'est utile ?](#pourquoi)
3. [Les 3 grandes familles](#architectures)
   - [GAN — le combat de boxe](#gan)
   - [VAE — le carnet de notes](#vae)
   - [Diffusion Models — la photo sous la pluie](#diffusion)
4. [Qui gagne ? Comparaison](#comparaison)
5. [Ce que j'en retiens](#conclusion)

---

## C'est quoi exactement ? <a name="cest-quoi"></a>

Laisse moi te poser une question simple.

T'as déjà regardé tellement de photos de chats que tu pourrais en dessiner un de mémoire, sans modèle devant toi ? Même un chat imaginaire, avec des couleurs bizarres et une tête un peu différente ?

Bah c'est exactement ce qu'on demande à une machine de faire.

On lui montre des milliers (parfois des millions) d'images. Elle apprend les patterns, les textures, les formes, les couleurs. Et ensuite... elle peut en **inventer de nouvelles**. Pas copier-coller. Vraiment inventer.

Quelques exemples qui montrent jusqu'où ça va :

- Tu tapes *"un astronaute qui joue de la guitare sur la lune au coucher du soleil"* → **DALL-E** te génère l'image en 5 secondes
- Tu vas sur **ThisPersonDoesNotExist.com** → tu vois un visage humain ultra-réaliste d'une personne qui n'a jamais existé
- **Stable Diffusion** peut transformer ta photo en peinture impressionniste
- **Sora d'OpenAI** génère des vidéos entières à partir d'une phrase

On est loin des filtres Instagram, hein.

![Exemples de génération d'images par IA](genai_examples.png)

---

## Pourquoi c'est utile ? <a name="pourquoi"></a>

La première fois que j'ai entendu parler de tout ça, ma première réaction c'était : *"ok c'est cool pour faire des belles images, mais c'est vraiment utile ?"*

Spoiler : ouais, vraiment.

**En médecine**, par exemple, les maladies rares ont très peu de cas documentés. Donc peu de photos pour entraîner un modèle de détection. La solution ? Générer artificiellement des milliers d'images de la maladie pour compléter le dataset. Ça sauve des vies, littéralement.

**Pour les voitures autonomes**, on ne peut pas tester tous les scénarios dangereux en vrai (pluie intense, brouillard, route inondée...). On génère ces situations virtuellement pour entraîner les voitures sans risquer des accidents réels.

**Pour la vie privée**, si une caméra de surveillance doit être partagée avec des chercheurs, on peut remplacer les vrais visages par des visages synthétiques générés par IA. Personne n'est identifiable, mais les données restent utilisables.

Et bien sûr, **pour la créativité** : des artistes, des designers, des cinéastes utilisent ces outils pour explorer des idées qu'ils n'auraient jamais pu concrétiser autrement.

![Applications du Generative AI](genai_applications.png)

---

## Les 3 grandes familles <a name="architectures"></a>

Il existe plusieurs façons de faire du Generative AI. Je vais te présenter les 3 principales, chacune avec une analogie du quotidien pour que ça colle vraiment.

---

### 🥊 GAN — le combat de boxe <a name="gan"></a>

**GAN = Generative Adversarial Network**

#### L'idée en une phrase

Deux réseaux de neurones se battent l'un contre l'autre jusqu'à ce que l'un d'eux devienne tellement bon que personne ne peut plus détecter ses faux.

#### L'analogie

Imagine un étudiant en art qui essaie de reproduire des tableaux de Picasso pour les vendre comme des vrais (oui, c'est pas légal, mais reste avec moi 😅).

En face de lui, un expert en art qui a passé 30 ans à étudier Picasso et qui est capable de repérer le moindre faux.

L'étudiant montre son tableau. L'expert dit *"faux"*. L'étudiant comprend où il a merdé, il améliore sa technique. Il revient. L'expert dit encore *"faux"*. Et ainsi de suite...

Jusqu'au jour où l'expert n'est plus capable de faire la différence. À ce moment-là, l'étudiant est devenu aussi fort que Picasso lui-même.

C'est exactement ça un GAN.

#### Techniquement

```
Bruit aléatoire ──► [ Générateur G ] ──► Image Fake
                                               │
Image Réelle ────► [ Discriminateur D ] ◄──────┘
                           │
                    "Vrai" ou "Faux" ?
                           │
              Feedback pour améliorer G
```

![Architecture GAN — Générateur vs Discriminateur](gan.png)

- Le **Générateur** part de bruit pur (des pixels aléatoires) et essaie de créer quelque chose de convaincant
- Le **Discriminateur** reçoit soit une vraie image, soit une image générée, et doit deviner laquelle c'est
- Les deux s'améliorent ensemble, en opposition

#### Ce que j'aime bien avec les GAN ✅

Les résultats peuvent être vraiment impressionnants. StyleGAN2 génère des visages humains tellement réalistes que même des humains s'y trompent. CycleGAN peut transformer une photo de cheval en zèbre, ou une scène d'été en hiver, avec une précision hallucinante.

Et une fois entraîné, générer une image est très rapide.

#### Ce qui est galère avec les GAN ❌

L'entraînement est une vraie souffrance. Le Générateur et le Discriminateur doivent s'améliorer au même rythme — si l'un prend trop d'avance, tout s'effondre.

Il y a aussi le fameux **mode collapse** : le Générateur découvre qu'un certain type d'image trompe toujours le Discriminateur... et il génère toujours la même. Un seul visage parfait, en boucle. Inutile.

![Exemple de résultats GAN — StyleGAN visages générés](gan_results.png)

---

### 📓 VAE — le carnet de notes <a name="vae"></a>

**VAE = Variational Autoencoder**

#### L'idée en une phrase

On compresse une image en quelques chiffres clés, puis on apprend à reconstruire une image à partir de ces chiffres.

#### L'analogie

T'as déjà essayé de résumer un film de 2h en 3 phrases pour un ami qui n'a pas le temps de le regarder ?

Tu gardes l'essentiel : le personnage principal, le conflit, la fin. Ton ami reconstitue le film dans sa tête à partir de ton résumé.

Ce que tu as fait, c'est **encoder** le film en information compressée. Ce que ton ami a fait, c'est **décoder** cette information pour reconstruire quelque chose de cohérent.

Maintenant imagine que tu changes légèrement ton résumé — tu modifies la fin, par exemple. Ton ami va reconstruire une version différente du film. Un film qui n'existait pas avant.

C'est ça un VAE.

#### Techniquement

```
Image Originale ──► [ Encodeur ] ──► Vecteur z (espace latent)
                                              │
                                     [ Décodeur ]
                                              │
                                     Image Reconstruite
```

![Architecture VAE — Encodeur et Décodeur](vae.png)

L'espace latent, c'est comme un "ADN numérique" de l'image. Modifier quelques chiffres dans ce vecteur = modifier l'image de façon contrôlée.

Tu peux, par exemple, avoir un vecteur qui code un visage. Tu augmentes la valeur du paramètre "sourire" → le visage sourit. Tu changes "âge" → le visage vieillit. Tout ça sans jamais voir une vraie photo.

#### Ce que j'aime bien avec les VAE ✅

C'est beaucoup plus stable à entraîner que les GAN. Et surtout, l'espace latent est **continu et structuré** — ça veut dire qu'on peut naviguer dedans de façon cohérente. Deux images proches dans l'espace latent se ressemblent visuellement.

C'est très utile pour comprendre comment un modèle "voit" les données.

#### Ce qui est galère avec les VAE ❌

Les images générées ont tendance à être un peu floues. C'est inhérent à la compression — quand tu résumes un film en 3 phrases, tu perds des détails. Pareil ici.

Donc si tu cherches un rendu ultra-réaliste, le VAE n'est pas forcément le meilleur choix.

![Espace latent VAE — interpolation entre images](vae_latent.png)

---

### 🌧️ Diffusion Models — la photo sous la pluie <a name="diffusion"></a>

#### L'idée en une phrase

On apprend à "débruiter" une image — et une fois qu'on sait faire ça, on peut générer une image à partir de rien.

#### L'analogie

Imagine que tu as une très belle photo de famille.

Maintenant, imagine que tu la laisses sous une pluie de sable fin, de plus en plus dense. Au bout d'un moment, on ne voit plus rien — juste du sable, du bruit, du chaos.

Maintenant, imagine que tu apprennes à **inverser** ce processus. Étape par étape, tu enlèves le sable. Un peu à la fois. Et la photo réapparaît.

Une fois que tu maîtrises ça, tu peux partir d'un tas de sable pur (du bruit aléatoire) et construire une image belle et cohérente, étape par étape. En guidant le processus avec des instructions (*"crée un chat roux qui dort sur un canapé"*), tu contrôles ce qui émerge.

C'est le principe des Diffusion Models.

#### Techniquement

```
PHASE 1 — Forward (on ajoute du bruit progressivement) :
🖼️ Image nette → 🌫️ → 🌫️🌫️ → 🌫️🌫️🌫️ → ❄️ Bruit pur

PHASE 2 — Reverse (le modèle apprend à débruiter) :
❄️ Bruit pur → 🌫️🌫️🌫️ → 🌫️🌫️ → 🌫️ → 🖼️ Image générée
```

![Processus de diffusion — forward et reverse](diffusion.png)

Le modèle n'apprend pas à créer une image de zéro. Il apprend juste une chose : *"étant donné cette image bruitée, qu'est-ce qui ressemblerait à l'original ?"* Répété des centaines de fois, ça donne quelque chose de magnifique.

#### Ce que j'aime bien avec les Diffusion Models ✅

La qualité est imbattable. Stable Diffusion, DALL-E 3, Midjourney — tous basés sur ce principe. Les images générées sont souvent indiscernables du réel.

L'entraînement est stable, le contrôle est précis (tu peux guider avec du texte, une esquisse, une autre image...).

#### Ce qui est galère avec les Diffusion Models ❌

C'est **lent**. Parce qu'il faut faire des dizaines (parfois des centaines) d'étapes de débruitage pour générer une seule image. Comparé à un GAN qui génère en une passe, c'est beaucoup plus lourd.

Et les ressources nécessaires sont importantes — un GPU costaud, beaucoup de mémoire.

![Résultats Stable Diffusion — exemples de génération](diffusion_results.png)

---

## Qui gagne ? Comparaison <a name="comparaison"></a>

Honnêtement, il n'y a pas un grand gagnant. Tout dépend de ce que tu veux faire.

| | GAN | VAE | Diffusion |
|---|---|---|---|
| **Idée** | Deux réseaux en duel | Compresser puis reconstruire | Bruiter puis débruiter |
| **Qualité** | ⭐⭐⭐⭐ | ⭐⭐ (flou) | ⭐⭐⭐⭐⭐ |
| **Entraînement** | 😤 Instable | 😌 Facile | 😌 Stable |
| **Vitesse** | ⚡ Rapide | ⚡ Rapide | 🐢 Lent |
| **Contrôle** | Moyen | Bon | Excellent |
| **Exemples** | StyleGAN, CycleGAN | Beta-VAE | Stable Diffusion, DALL-E 3 |
| **Utilisation idéale** | Visages réalistes, transfert de style | Explorer l'espace latent | Texte → image, édition fine |

En gros :
- Si tu veux du **rapide et réaliste** → GAN
- Si tu veux **comprendre et contrôler** les données → VAE
- Si tu veux la **meilleure qualité possible** → Diffusion Models

---

## Ce que j'en retiens <a name="conclusion"></a>

Avant ce cours, pour moi l'IA c'était surtout "reconnaître des trucs" — classifier des images, détecter des objets. Ce qu'on a vu avec le GenAI m'a un peu changé la façon de voir les choses.

On ne parle plus juste d'une machine qui analyse. On parle d'une machine qui **imagine**.

Et ce qui est vraiment intéressant, c'est que GAN, VAE et Diffusion Models ne font pas la même chose de la même façon. Chacun a sa propre "philosophie" pour créer.

- Le GAN apprend **par compétition**
- Le VAE apprend **par compression**
- Le Diffusion Model apprend **par reconstruction**

Trois visions différentes de ce que "créer" veut dire pour une machine.

Franchement, j'aurais jamais pensé qu'un jour j'écrirais un blog pour expliquer comment un algorithme peut inventer un visage humain. Et pourtant, me voilà.

---

*Rédigé dans le cadre du cours Computer Vision & Generative AI — S8, Avril 2026.*  
*Sources : cours de M. SAAD, documentation Stable Diffusion, paper GAN original (Goodfellow et al., 2014)*
