# 💬 NLP Labs

Travaux pratiques du module de **traitement automatique du langage** (S6) : tokenisation, détection de sarcasme (des sacs de mots aux embeddings) et analyse de sentiment en français, avec un dashboard Flask.

Les notebooks sont **déjà exécutés** : on peut lire tous les résultats directement sur GitHub.

## Notebooks

### [01 · Tokenisation avec Keras](01_tokenization_keras.ipynb)
Les bases de la préparation de texte : `Tokenizer` (effet de `num_words`), conversion en séquences et retour au texte, token hors vocabulaire `<unk>`, puis `pad_sequences`.

### [02 · Détection de sarcasme](02_detection_sarcasme.ipynb)
Dataset *News Headlines for Sarcasm Detection* (26 709 titres de The Onion et du HuffPost). Plusieurs représentations y sont comparées :

| Approche | Accuracy |
|---|---|
| Tokenizer Keras + padding + régression logistique | 0,64 (test) |
| Bag-of-Words + régression logistique | 0,97 (test)* |
| TF-IDF + SGDClassifier | 0,96 (test)* |
| Embedding appris + Global Average Pooling | 0,98 (validation)* |
| Embedding appris + CNN 1D + Global Max Pooling | 0,97 (validation)* |
| GloVe 100d figé + Global Average Pooling | 0,86 (validation)* |

> ⚠️ **\* Limite connue : fuite de données.** Au prétraitement, les mots vides (stopwords) ne sont retirés que des titres **sarcastiques**. Les modèles peuvent donc reconnaître un titre sarcastique à l'absence de mots vides, sans comprendre le sarcasme. Cela explique le rappel de 1,00 sur cette classe. Les scores marqués d'un \* sont surestimés : il faudrait appliquer le même prétraitement à tous les titres pour mesurer les vraies performances.

Le notebook présente aussi le fonctionnement d'une couche d'`Embedding` et la construction d'une matrice d'embedding à partir de GloVe.

### [03 · Analyse de sentiment en français, avec dashboard Flask](03_sentiment_francais_flask.ipynb)
Un pipeline complet sur le dataset **Allociné** (critiques de films, positif ou négatif) :
- nettoyage du texte (Unicode, URLs, mentions, chiffres) et lemmatisation avec **spaCy** (`fr_core_news_sm`) ;
- Bag-of-Words et TF-IDF (1-2 grammes) ;
- comparaison de 5 modèles sur 12 000 avis :

| Modèle | Accuracy | F1 pondéré |
|---|---|---|
| **TF-IDF + LinearSVC** | **0,905** | **0,905** |
| BoW + MultinomialNB | 0,902 | 0,902 |
| BoW + ComplementNB | 0,902 | 0,902 |
| TF-IDF + régression logistique | 0,896 | 0,896 |
| TF-IDF + GradientBoosting | 0,821 | 0,821 |

- interprétabilité : les mots les plus discriminants selon la régression logistique ;
- prédiction avec un **score de confiance** et une classe « neutre » en option (abstention sous un seuil) ;
- application : classement de 3 projets étudiants selon le sentiment de leurs commentaires ;
- **dashboard Flask** (ajout et import CSV de commentaires, graphiques), exposé depuis Colab grâce à ngrok.

## Lancer les notebooks

Les notebooks ont été conçus pour **Google Colab**. En local :

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
jupyter notebook
```

Pour le dashboard du notebook 03, ajoutez `NGROK_AUTHTOKEN` aux secrets Colab. Le notebook 02 télécharge GloVe (822 Mo).

## Technologies

Python · TensorFlow / Keras · scikit-learn · spaCy · NLTK · Hugging Face Datasets · Flask · ngrok
