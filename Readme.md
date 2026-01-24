# 🤖 Système de Classification de Documents

[![Technical Report](https://img.shields.io/badge/Documentation-Technical%20Report-blue?style=for-the-badge&logo=read-the-dots)](./Technical_Report.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview
This project involves the ground-up construction of a Large Language Model (LLM), focusing on the core architectural principles of modern generative AI. Beyond simple implementation, this project serves as a research framework to explore **neural architecture design**, **tokenization strategies**, and **model interpretability**.

## 📄 Academic Documentation
For a deep dive into the mathematical foundations and performance analysis of this implementation, please refer to the full technical report:
> **[Download Technical Report (PDF)](./Technical_Report_LLM.pdf)**

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture du système](#architecture-du-système)
3. [Technologies utilisées](#technologies-utilisées)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Structure du projet](#structure-du-projet)
7. [Utilisation](#utilisation)
8. [Classes de documents supportées](#classes-de-documents-supportées)
9. [Algorithme de fusion](#algorithme-de-fusion)
10. [API et Fonctions principales](#api-et-fonctions-principales)
11. [Dépannage](#dépannage)
12. [Performance et optimisation](#performance-et-optimisation)
13. [Roadmap](#roadmap)

---

## 🎯 Vue d'ensemble

Ce projet implémente un **système de classification hybride de documents** combinant :
- **NLP (Natural Language Processing)** via Gemini AI (90% du poids)
- **Computer Vision** via ResNet50 fine-tuné (10% du poids)

### Objectif
Classifier automatiquement des documents administratifs marocains/francophones (cartes d'identité, factures, relevés bancaires, etc.) avec une haute précision en combinant l'analyse textuelle et visuelle.

### Cas d'usage
- ✅ Automatisation du traitement de documents administratifs
- ✅ Numérisation et indexation de dossiers clients
- ✅ Système de gestion documentaire intelligent
- ✅ Vérification et validation de documents
- ✅ Recherche sémantique dans une base documentaire

---

## 🏗️ Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT D'ENTRÉE                         │
│                  (PDF ou Image JPG/PNG)                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────┬──────────────────────────┐
                     │              │                          │
                     ▼              ▼                          ▼
           ┌─────────────┐  ┌─────────────┐         ┌─────────────┐
           │  EXTRACTION │  │  EXTRACTION │         │  CONVERSION │
           │    TEXTE    │  │    IMAGE    │         │  PDF → IMG  │
           │   (PyMuPDF) │  │    (PIL)    │         │  (PyMuPDF)  │
           └──────┬──────┘  └──────┬──────┘         └──────┬──────┘
                  │                │                        │
                  │  ┌─────────────┘                        │
                  │  │                                      │
                  ▼  ▼                                      ▼
         ┌─────────────────┐                      ┌─────────────────┐
         │   OCR (optionnel)│                      │                 │
         │   Pytesseract    │                      │                 │
         │  fra+ara+eng     │                      │                 │
         └────────┬─────────┘                      │                 │
                  │                                │                 │
                  ▼                                ▼                 │
         ┌─────────────────┐              ┌─────────────────┐       │
         │  CLASSIFICATION  │              │  CLASSIFICATION │       │
         │       NLP        │              │       CV        │       │
         │   (Gemini AI)    │              │   (ResNet50)    │       │
         │                  │              │                 │       │
         │  7 classes       │              │  5 classes      │       │
         │  détaillées      │              │  générales      │       │
         └────────┬─────────┘              └────────┬────────┘       │
                  │                                 │                │
                  │   Classe + Confiance            │  Classe + Conf │
                  │                                 │                │
                  └──────────────┬──────────────────┘                │
                                 │                                   │
                                 ▼                                   │
                        ┌─────────────────┐                          │
                        │  FUSION SCORES  │                          │
                        │                 │                          │
                        │  90% × NLP +    │                          │
                        │  10% × CV       │                          │
                        │                 │                          │
                        │  + Bonus accord │                          │
                        └────────┬────────┘                          │
                                 │                                   │
                                 ▼                                   │
                        ┌─────────────────┐                          │
                        │ RÉSULTAT FINAL  │                          │
                        │                 │                          │
                        │ • Classe finale │                          │
                        │ • Score final   │                          │
                        │ • Accord NLP/CV │                          │
                        └────────┬────────┘                          │
                                 │                                   │
                                 ▼                                   │
                        ┌─────────────────┐                          │
                        │   INDEXATION    │◄─────────────────────────┘
                        │                 │
                        │ • ChromaDB      │
                        │ • Embeddings    │
                        │ • Métadonnées   │
                        └─────────────────┘
```

---

## 🛠️ Technologies utilisées

### Frameworks & Libraries

| Technologie | Version | Usage |
|------------|---------|-------|
| **Python** | 3.8+ | Langage principal |
| **Streamlit** | 1.28+ | Interface web |
| **PyTorch** | 2.0+ | Deep Learning (ResNet) |
| **Google Generative AI** | Latest | NLP (Gemini) |
| **ChromaDB** | 0.4+ | Base de données vectorielle |
| **PyMuPDF (fitz)** | 1.23+ | Extraction PDF |
| **Pytesseract** | 0.3+ | OCR |
| **Pillow (PIL)** | 10.0+ | Traitement d'images |
| **TorchVision** | 0.15+ | Modèles CV pré-entraînés |

### Modèles IA

1. **Gemini 2.0 Flash Exp** (NLP)
   - Classification textuelle
   - Génération d'embeddings
   - Réponses RAG

2. **ResNet50 Fine-tuné** (CV)
   - Classification d'images
   - 5 classes de documents
   - Poids personnalisés (`resnet_finetuned.pth`)

---

## 📦 Installation

### Prérequis

- **Python 3.8+**
- **Tesseract OCR** (pour l'extraction de texte des images scannées)
- **Clé API Google Gemini**

### Étape 1 : Cloner le repository

```bash
git clone https://github.com/votre-repo/document-classifier.git
cd document-classifier
```

### Étape 2 : Créer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Étape 3 : Installer les dépendances Python

```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**
```txt
streamlit>=1.28.0
google-generativeai>=0.3.0
chromadb>=0.4.0
PyPDF2>=3.0.0
PyMuPDF>=1.23.0
pytesseract>=0.3.10
pdf2image>=1.16.3
Pillow>=10.0.0
torch>=2.0.0
torchvision>=0.15.0
python-dotenv>=1.0.0
numpy>=1.24.0
```

### Étape 4 : Installer Tesseract OCR

#### Windows
1. Télécharger l'installeur : https://github.com/UB-Mannheim/tesseract/wiki
2. Installer dans `C:\Program Files\Tesseract-OCR\`
3. Ajouter au PATH ou configurer dans `.env`

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-fra tesseract-ocr-ara tesseract-ocr-eng
```

#### macOS
```bash
brew install tesseract
brew install tesseract-lang
```

### Étape 5 : Vérifier l'installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
tesseract --version
```

---

## ⚙️ Configuration

### Fichier `.env`

Créez un fichier `.env` à la racine du projet :

```env
# API Key Google Gemini (OBLIGATOIRE)
GEMINI_API_KEY=votre_cle_api_ici

# Chemin Tesseract (Windows uniquement)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Nom de la collection ChromaDB (optionnel)
COLLECTION_NAME=documents_hybrid

# Dossier de documents local (optionnel)
DOCUMENTS_FOLDER=./documents
```

### Obtenir une clé API Gemini

1. Allez sur https://makersuite.google.com/app/apikey
2. Connectez-vous avec votre compte Google
3. Cliquez sur "Create API Key"
4. Copiez la clé dans votre fichier `.env`

### Fichier `gabarits.json`

Ce fichier définit les classes pour le modèle ResNet et les seuils de confiance :

```json
{
  "classes": ["id_card", "bank_statement", "elec_and_water_bill", "employer_doc", "other"],
  "thresholds": {
    "cv_confidence": 0.8,
    "nlp_confidence": 0.7,
    "fusion_rejection": 0.5
  },
  "geometry": {
    "id_card": {
      "min_ratio": 1.3,
      "max_ratio": 1.8,
      "requires_face": true
    },
    "others": {
      "min_ratio": 0.5,
      "max_ratio": 1.0,
      "requires_face": false
    }
  },
  "keywords": {
    "id_card": ["CNIE", "Nationale", "Royaume", "Maroc", "Né le", "Nom", "Prénom"],
    "bank_statement": ["Solde", "Banque", "IBAN", "Virement", "Retrait"],
    "elec_and_water_bill": ["kWh", "Compteur", "Lydec", "ONE", "Facture"],
    "employer_doc": ["Salaire", "Paie", "Attestation", "CNSS"],
    "other": ["Autre", "Document"]
  }
}
```

---

## 📁 Structure du projet

```
document-classifier/
│
├── app.py                      # Application Streamlit principale
├── gabarits.json              # Configuration des classes ResNet
├── resnet_finetuned.pth       # Poids du modèle ResNet50
├── .env                       # Variables d'environnement
├── requirements.txt           # Dépendances Python
├── README.md                  # Cette documentation
│
├── chroma_db/                 # Base de données ChromaDB (généré)
│   └── ...
│
├── documents/                 # Dossier de documents à indexer (optionnel)
│   ├── cin_001.pdf
│   ├── facture_eau.jpg
│   └── ...
│
└── venv/                      # Environnement virtuel (ignoré par git)
    └── ...
```

---

## 🚀 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à `http://localhost:8501`

### Interface utilisateur

L'application comporte **3 onglets principaux** :

#### 📤 Onglet 1 : Classification

**Upload de documents :**
1. Cliquez sur "Browse files"
2. Sélectionnez vos documents :
   - **PDFs** : Multi-pages supportées
   - **Images** : JPG, PNG, JPEG, BMP, TIFF
3. La classification démarre **automatiquement**

**Résultats affichés :**
- **Score NLP** : Classification Gemini + confiance
- **Score CV** : Classification ResNet + confiance
- **Score Final** : Fusion pondérée (90% NLP + 10% CV)
- **Indicateur d'accord** : ✅ accord ou ⚠️ désaccord
- **Texte extrait** (pour les images via OCR)

#### 🔍 Onglet 2 : Recherche

**Recherche sémantique :**
1. Tapez une question en langage naturel
   - Exemple : *"Quelles sont mes factures d'eau de janvier ?"*
2. Filtrez par type de document (optionnel)
3. Ajustez le nombre de résultats (1-10)
4. Cliquez sur "Rechercher"

**Résultats :**
- Réponse générée par Gemini basée sur le contexte
- Sources avec métadonnées complètes
- Scores de classification pour chaque source

#### 📊 Onglet 3 : Statistiques

**Visualisation de la collection :**
- Nombre total de documents indexés
- Nombre de types différents
- Taux d'accord NLP/CV (en %)
- Scores moyens (NLP, CV, Final)
- Répartition par type de document
- Option de suppression de la collection

---

## 📋 Classes de documents supportées

### Classes NLP (Gemini) - 7 classes détaillées

| Classe | Description | Mots-clés |
|--------|-------------|-----------|
| **CIN_RECTO** | Carte d'identité (recto) | Photo, nom, prénom, date de naissance, lieu de naissance, nationalité |
| **CIN_VERSO** | Carte d'identité (verso) | Adresse, profession, taille, signature, autorité |
| **FACTURE_EAU** | Facture d'eau | LYDEC, RADEEMA, ONEP, m³, index, consommation |
| **FACTURE_ELECTRICITE** | Facture d'électricité | ONE, LYDEC, kWh, puissance, tension |
| **RELEVE_BANCAIRE** | Relevé bancaire | Attijariwafa, BMCE, IBAN, virement, solde |
| **DOCUMENT_EMPLOYEUR** | Attestation de travail | Bulletin de paie, CNSS, salaire, employeur |
| **AUTRE_DOCUMENT** | Autre type | Documents non classifiés |

### Classes CV (ResNet50) - 5 classes générales

| Classe | Description | Mapping NLP |
|--------|-------------|-------------|
| **id_card** | Carte d'identité | CIN_RECTO, CIN_VERSO |
| **bank_statement** | Relevé bancaire | RELEVE_BANCAIRE |
| **elec_and_water_bill** | Facture eau/électricité | FACTURE_EAU, FACTURE_ELECTRICITE |
| **employer_doc** | Document employeur | DOCUMENT_EMPLOYEUR |
| **other** | Autre document | AUTRE_DOCUMENT |

---

## 🧮 Algorithme de fusion

### Formule de base

```python
Score_Final = (Score_NLP × 0.90) + (Score_CV × 0.10)
```

### Bonus d'accord

Si les deux modèles sont d'accord (même famille de classes) :

```python
Score_Final = (Score_NLP × 0.90) + (Score_CV × 0.10) + 0.05
```

### Exemple de calcul

**Cas 1 : Accord entre les modèles**
```
NLP : CIN_RECTO (confiance 0.95)
CV  : id_card (confiance 0.88)
→ Accord ✅ (CIN_RECTO → id_card)

Score_Final = (0.95 × 0.90) + (0.88 × 0.10) + 0.05
            = 0.855 + 0.088 + 0.05
            = 0.993 (99.3%)

Classe finale : CIN_RECTO
```

**Cas 2 : Désaccord entre les modèles**
```
NLP : FACTURE_EAU (confiance 0.82)
CV  : id_card (confiance 0.65)
→ Désaccord ⚠️

Score_Final = (0.82 × 0.90) + (0.65 × 0.10)
            = 0.738 + 0.065
            = 0.803 (80.3%)

Classe finale : FACTURE_EAU (priorité au NLP)
```

### Mapping des classes

```python
CLASS_MAPPING = {
    "id_card": ["CIN_RECTO", "CIN_VERSO"],
    "elec_and_water_bill": ["FACTURE_EAU", "FACTURE_ELECTRICITE"],
    "bank_statement": ["RELEVE_BANCAIRE"],
    "employer_doc": ["DOCUMENT_EMPLOYEUR"],
    "other": ["AUTRE_DOCUMENT"]
}
```

---

## 🔧 API et Fonctions principales

### 1. Classification NLP

```python
def classify_document_page_nlp(model, page_content: str) -> Tuple[str, float]:
    """
    Classifie une page avec le LLM Gemini
    
    Args:
        model: Instance du modèle Gemini
        page_content: Texte à classifier
    
    Returns:
        (classe_nlp, score_confiance)
        - classe_nlp: Une des 7 classes NLP
        - score_confiance: Float entre 0.0 et 1.0
    """
```

### 2. Classification CV

```python
class ResNetClassifier:
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict]:
        """
        Prédit la classe d'une image avec ResNet50
        
        Args:
            image: Image PIL
        
        Returns:
            (classe_cv, score_confiance, probabilites_toutes_classes)
            - classe_cv: Une des 5 classes CV
            - score_confiance: Float entre 0.0 et 1.0
            - probabilites_toutes_classes: Dict {classe: proba}
        """
```

### 3. Fusion des prédictions

```python
def fuse_predictions(nlp_class: str, nlp_conf: float, 
                     cv_class: str, cv_conf: float,
                     weights: Dict = {"nlp": 0.90, "cv": 0.10}) -> Tuple[str, float, Dict]:
    """
    Fusionne les prédictions NLP et CV avec pondération
    
    Args:
        nlp_class: Classe prédite par NLP
        nlp_conf: Confiance NLP
        cv_class: Classe prédite par CV
        cv_conf: Confiance CV
        weights: Poids de fusion (défaut: 90% NLP, 10% CV)
    
    Returns:
        (classe_finale, score_final, details)
    """
```

### 4. Extraction de texte

```python
def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> List[Dict]:
    """
    Extrait le texte page par page d'un PDF
    
    Args:
        pdf_path: Chemin du fichier PDF
        use_ocr: Activer l'OCR pour les pages scannées
    
    Returns:
        Liste de dictionnaires avec :
        - page_num: Numéro de page
        - content: Texte extrait
        - file_name: Nom du fichier
    """
```

### 5. Génération d'embeddings

```python
def generate_embedding(text: str) -> List[float]:
    """
    Génère un embedding vectoriel avec Gemini
    
    Args:
        text: Texte à vectoriser
    
    Returns:
        Vecteur d'embedding (768 dimensions)
    """
```

### 6. Recherche sémantique

```python
def search_documents(collection, query: str, filters: Dict = None, n_results: int = 5):
    """
    Recherche sémantique dans la base ChromaDB
    
    Args:
        collection: Collection ChromaDB
        query: Question en langage naturel
        filters: Filtres optionnels (par type de document)
        n_results: Nombre de résultats à retourner
    
    Returns:
        Résultats de recherche avec documents et métadonnées
    """
```

---

## 🐛 Dépannage

### Problème 1 : Erreur de chargement ResNet

**Erreur :**
```
size mismatch for fc.weight: copying a param with shape torch.Size([5, 2048]) 
from checkpoint, the shape in current model is torch.Size([4, 2048])
```

**Solution :**
Le modèle a été entraîné sur 5 classes mais `gabarits.json` n'en contient que 4.

Ajoutez la classe manquante dans `gabarits.json` :
```json
{
  "classes": ["id_card", "bank_statement", "elec_and_water_bill", "employer_doc", "other"]
}
```

### Problème 2 : OCR ne fonctionne pas

**Erreur :**
```
TesseractNotFoundError: tesseract is not installed or it's not in your PATH
```

**Solution Windows :**
1. Téléchargez Tesseract : https://github.com/UB-Mannheim/tesseract/wiki
2. Installez dans `C:\Program Files\Tesseract-OCR\`
3. Ajoutez dans `.env` :
```env
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Solution Linux :**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

### Problème 3 : Clé API Gemini invalide

**Erreur :**
```
google.api_core.exceptions.PermissionDenied: 403 API key not valid
```

**Solution :**
1. Vérifiez que la clé API est correcte dans `.env`
2. Assurez-vous que l'API Gemini est activée : https://makersuite.google.com/
3. Vérifiez les quotas de votre API

### Problème 4 : ChromaDB ne persiste pas

**Symptôme :** Les documents disparaissent après redémarrage

**Solution :**
Vérifiez que ChromaDB utilise bien `PersistentClient` :
```python
client = chromadb.PersistentClient(path="./chroma_db")
```

Le dossier `chroma_db/` doit être créé et contenir des fichiers.

### Problème 5 : Mémoire insuffisante

**Erreur :**
```
RuntimeError: CUDA out of memory
```

**Solution :**
1. Réduire la taille du batch (traiter les documents un par un)
2. Utiliser CPU au lieu de GPU :
```python
self.device = torch.device("cpu")
```
3. Fermer les autres applications

---

## ⚡ Performance et optimisation

### Temps de traitement moyen

| Opération | Temps (1 page) |
|-----------|----------------|
| Extraction texte PDF | ~0.5s |
| OCR image (si nécessaire) | ~2-3s |
| Classification NLP | ~1-2s |
| Classification CV | ~0.3s |
| Génération embedding | ~0.5s |
| Indexation ChromaDB | ~0.2s |
| **Total par page** | **~3-7s** |

### Optimisations possibles

1. **Batch processing** : Traiter plusieurs pages en parallèle
```python
# À implémenter
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_page, pages)
```

2. **Cache des embeddings** : Éviter de recalculer les mêmes textes
```python
@st.cache_data
def generate_embedding(text: str):
    # ...
```

3. **GPU pour ResNet** : Accélération avec CUDA
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

4. **Compression des images** : Réduire la taille avant classification
```python
image.thumbnail((800, 800))
```

### Limites actuelles

- **Taille max PDF** : ~50 pages (au-delà, traitement lent)
- **Taille max image** : 10 MB
- **Quota API Gemini** : 60 requêtes/minute (gratuit)
- **ChromaDB** : ~100K documents recommandé

---

## 🗺️ Roadmap

### Version 1.1 (À venir)
- [ ] Support de formats Word (.docx)
- [ ] Export des résultats en CSV/JSON
- [ ] Traitement batch de dossiers entiers
- [ ] Interface de correction manuelle

### Version 1.2
- [ ] API REST pour intégration externe
- [ ] Authentification utilisateurs
- [ ] Multi-langue (anglais, arabe complet)
- [ ] Dashboard analytics avancé

### Version 2.0
- [ ] Fine-tuning du modèle ResNet sur vos données
- [ ] Extraction automatique de champs (nom, montant, date)
- [ ] Workflow de validation documentaire
- [ ] Intégration OCR cloud (Google Vision, AWS Textract)

---

## 📧 Support et Contribution

### Rapporter un bug

Créez une issue sur GitHub avec :
1. Description du problème
2. Étapes de reproduction
3. Logs d'erreur
4. Version de Python et dépendances

### Contribuer

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 👥 Auteurs

- **Votre Nom** - *Développement initial* - [VotreGitHub](https://github.com/votre-username)

---

## 🙏 Remerciements

- Google Gemini pour l'API NLP
- PyTorch pour le framework Deep Learning
- Streamlit pour l'interface utilisateur
- Anthropic Claude pour l'assistance au développement

---

## 📊 Statistiques du projet

- **Lignes de code** : ~1200
- **Fonctions** : 25+
- **Classes** : 1 (ResNetClassifier)
- **Formats supportés** : PDF, JPG, PNG, JPEG, BMP, TIFF
- **Langues OCR** : Français, Arabe, Anglais

---

**Dernière mise à jour** : Janvier 2025  
**Version** : 1.0.0
