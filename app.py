import streamlit as st
import google.generativeai as genai
from pathlib import Path
import chromadb
from chromadb.config import Settings
import PyPDF2
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
import tempfile
from dotenv import load_dotenv
import fitz  # PyMuPDF
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import pdf2image
import pytesseract
import numpy as np

# Configuration de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chargement des variables d'environnement
load_dotenv()

if os.getenv('TESSERACT_CMD'):
    try:
        pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
    except ImportError:
        pass

# Vérification de pytesseract (OCR optionnel)
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="RAG + ResNet - Classification Hybride",
    page_icon="🤖",
    layout="wide"
)

# ==================== MAPPING DES CLASSES ====================
# Mapping entre les classes ResNet (gabarits.json) et NLP (détaillées)
CLASS_MAPPING = {
    "id_card": ["CIN_RECTO", "CIN_VERSO"],
    "elec_and_water_bill": ["FACTURE_EAU", "FACTURE_ELECTRICITE"],
    "bank_statement": ["RELEVE_BANCAIRE"],
    "employer_doc": ["DOCUMENT_EMPLOYEUR"],
    "other": ["AUTRE_DOCUMENT"],
    "class_4": ["AUTRE_DOCUMENT"]  # Fallback pour la 5ème classe si détectée automatiquement
}

# Classes NLP détaillées (ordre important pour l'affichage)
NLP_CLASSES = [
    "CIN_RECTO", "CIN_VERSO", 
    "FACTURE_EAU", "FACTURE_ELECTRICITE",
    "RELEVE_BANCAIRE", "DOCUMENT_EMPLOYEUR",
    "AUTRE_DOCUMENT"
]

# Mapping inverse : NLP -> ResNet
NLP_TO_RESNET = {}
for resnet_class, nlp_list in CLASS_MAPPING.items():
    for nlp_class in nlp_list:
        NLP_TO_RESNET[nlp_class] = resnet_class

# ==================== CHARGEMENT DES GABARITS ====================
@st.cache_data
def load_gabarits():
    """Charge les gabarits depuis le fichier JSON"""
    try:
        with open("gabarits.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Erreur chargement gabarits.json : {e}")
        return None

# ==================== MODÈLE RESNET ====================
class ResNetClassifier:
    def __init__(self, model_path: str, gabarits: dict):
        self.gabarits = gabarits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Chargement du modèle ResNet50
        self.model = models.resnet50(pretrained=False)
        
        # Chargement des poids pour détecter le nombre de classes
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Détection automatique du nombre de classes depuis les poids
            fc_weight_shape = state_dict['fc.weight'].shape
            num_classes_in_model = fc_weight_shape[0]
            
            st.info(f"🔍 Modèle ResNet détecté avec {num_classes_in_model} classes")
            
            # Vérification avec gabarits.json
            num_classes_in_json = len(gabarits["classes"])
            
            if num_classes_in_model != num_classes_in_json:
                st.warning(f"⚠️ Désaccord: gabarits.json a {num_classes_in_json} classes, mais le modèle a {num_classes_in_model} classes")
                st.info("🔧 Ajustement automatique en cours...")
                
                # Ajout de classes manquantes
                if num_classes_in_model > num_classes_in_json:
                    missing_count = num_classes_in_model - num_classes_in_json
                    for i in range(missing_count):
                        gabarits["classes"].append(f"class_{num_classes_in_json + i}")
                    st.success(f"✅ {missing_count} classe(s) ajoutée(s): {gabarits['classes'][num_classes_in_json:]}")
            
            self.classes = gabarits["classes"]
            
            # Reconstruction du modèle avec le bon nombre de classes
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes_in_model)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            st.success(f"✅ ResNet chargé avec {num_classes_in_model} classes sur {self.device}")
            st.write(f"📋 Classes: {', '.join(self.classes)}")
            
        except Exception as e:
            st.error(f"❌ Erreur chargement ResNet : {e}")
            self.model = None
            self.classes = gabarits["classes"]
        
        # Transformations pour les images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict]:
        """
        Prédit la classe d'une image avec score de confiance
        Returns: (classe_resnet, score_confiance, probabilités_toutes_classes)
        """
        if self.model is None:
            return "UNKNOWN", 0.0, {}
        
        try:
            # Conversion en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Transformation et prédiction
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.classes[predicted.item()]
            confidence_score = confidence.item()
            
            # Toutes les probabilités
            all_probs = {
                self.classes[i]: probabilities[0][i].item() 
                for i in range(len(self.classes))
            }
            
            return predicted_class, confidence_score, all_probs
            
        except Exception as e:
            st.error(f"❌ Erreur prédiction ResNet : {e}")
            return "UNKNOWN", 0.0, {}

@st.cache_resource
def load_resnet_model():
    """Charge le modèle ResNet avec cache"""
    gabarits = load_gabarits()
    if gabarits is None:
        return None
    return ResNetClassifier("resnet_finetuned.pth", gabarits)

# ==================== GEMINI / NLP ====================
def init_gemini(api_key: str):
    """Initialise l'API Gemini"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-3-flash-preview')

@st.cache_resource
def init_chromadb():
    """Initialise ChromaDB avec persistance"""
    client = chromadb.PersistentClient(path="./chroma_db")
    return client

# ==================== EXTRACTION PDF ====================
def extract_text_from_image(image_path: str, use_ocr: bool = True) -> str:
    """Extrait le texte d'une image avec OCR"""
    try:
        img = Image.open(image_path)
        
        if use_ocr and OCR_AVAILABLE:
            text = pytesseract.image_to_string(img, lang='fra+ara+eng')
            return text.strip()
        else:
            st.warning("⚠️ OCR non disponible pour extraire le texte de l'image")
            return ""
    except Exception as e:
        st.error(f"❌ Erreur extraction texte image : {e}")
        return ""

def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> List[Dict]:
    """Extrait le texte page par page d'un PDF avec support OCR"""
    pages_content = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            
            # Si le texte est vide ou trop court, essayer l'OCR
            if (not text or len(text) < 50) and use_ocr and OCR_AVAILABLE:
                st.info(f"📷 Page {page_num + 1} : Utilisation de l'OCR...")
                
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                try:
                    text = pytesseract.image_to_string(img, lang='fra+ara+eng')
                    text = text.strip()
                except Exception as e:
                    st.warning(f"⚠️ Erreur OCR page {page_num + 1}: {e}")
            
            if not text or len(text) < 20:
                st.warning(f"⚠️ Page {page_num + 1} : Contenu insuffisant ({len(text)} caractères)")
                continue
            
            pages_content.append({
                'page_num': page_num + 1,
                'content': text,
                'file_name': Path(pdf_path).name
            })
        
        doc.close()
        
    except Exception as e:
        st.error(f"❌ Erreur extraction PDF : {e}")
    
    if not pages_content:
        st.error(f"❌ Aucun texte extrait de {Path(pdf_path).name}")
    else:
        st.success(f"✅ {len(pages_content)} page(s) extraite(s)")
    
    return pages_content

def pdf_page_to_image(pdf_path: str, page_num: int) -> Image.Image:
    """Convertit une page PDF en image pour ResNet"""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        doc.close()
        return img
    except Exception as e:
        st.error(f"❌ Erreur conversion page {page_num + 1} en image : {e}")
        return None

# ==================== CLASSIFICATION NLP ====================
def classify_document_page_nlp(model, page_content: str) -> Tuple[str, float]:
    """
    Classifie une page avec le LLM (NLP)
    Returns: (classe_nlp, score_confiance)
    """
    if not page_content or len(page_content.strip()) < 20:
        return "AUTRE_DOCUMENT", 0.0
    
    prompt = f"""Tu es un expert en classification de documents administratifs marocains/francophones.

Analyse ce contenu et retourne UNIQUEMENT au format JSON suivant :
{{
  "classe": "...",
  "confiance": 0.XX
}}

Classes possibles (choisis LA PLUS PRÉCISE) :
- CIN_RECTO : Carte Nationale d'Identité (recto) - contient photo, nom, prénom, date/lieu naissance, nationalité
- CIN_VERSO : Carte Nationale d'Identité (verso) - contient adresse, profession, taille, signature, autorité
- FACTURE_EAU : Facture d'eau (LYDEC, RADEEMA, ONEP, consommation m³, index)
- FACTURE_ELECTRICITE : Facture d'électricité (ONE, LYDEC, kWh, puissance)
- RELEVE_BANCAIRE : Relevé bancaire (Attijariwafa, BMCE, Banque Populaire, IBAN, virements, solde)
- DOCUMENT_EMPLOYEUR : Attestation de travail, bulletin de paie, certificat CNSS
- AUTRE_DOCUMENT : Tout autre document

Le score de confiance doit être entre 0.0 et 1.0 (ex: 0.95 pour très sûr, 0.60 pour incertain).

Contenu à analyser :
{page_content[:3000]}

Réponds UNIQUEMENT avec le JSON (pas d'explication) :"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Nettoyage du JSON
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(result_text)
        classe = result.get("classe", "AUTRE_DOCUMENT").upper().replace(" ", "_")
        confiance = float(result.get("confiance", 0.5))
        
        # Validation de la classe
        if classe not in NLP_CLASSES:
            classe = "AUTRE_DOCUMENT"
            confiance = 0.3
        
        return classe, confiance
        
    except Exception as e:
        st.warning(f"⚠️ Erreur classification NLP : {e}")
        return "AUTRE_DOCUMENT", 0.0

# ==================== FUSION DES SCORES ====================
def fuse_predictions(nlp_class: str, nlp_conf: float, 
                     cv_class: str, cv_conf: float,
                     weights: Dict = {"nlp": 0.90, "cv": 0.10}) -> Tuple[str, float, Dict]:
    """
    Fusionne les prédictions NLP et CV avec pondération
    Returns: (classe_finale, score_final, details)
    """
    
    # Si les classes sont compatibles (même famille), on renforce la confiance
    nlp_resnet_class = NLP_TO_RESNET.get(nlp_class, "unknown")
    
    if nlp_resnet_class == cv_class:
        # Accord entre NLP et CV : on booste le score
        bonus = 0.05
        final_score = (nlp_conf * weights["nlp"]) + (cv_conf * weights["cv"]) + bonus
        final_class = nlp_class  # On garde la classe détaillée du NLP
        agreement = True
    else:
        # Désaccord : on fait la moyenne pondérée normale
        final_score = (nlp_conf * weights["nlp"]) + (cv_conf * weights["cv"])
        final_class = nlp_class  # NLP a plus de poids, donc on garde sa classe
        agreement = False
    
    # Clip le score entre 0 et 1
    final_score = min(1.0, max(0.0, final_score))
    
    details = {
        "nlp_class": nlp_class,
        "nlp_confidence": nlp_conf,
        "nlp_weight": weights["nlp"],
        "cv_class": cv_class,
        "cv_confidence": cv_conf,
        "cv_weight": weights["cv"],
        "agreement": agreement,
        "final_class": final_class,
        "final_score": final_score
    }
    
    return final_class, final_score, details

# ==================== CLASSIFICATION HYBRIDE POUR IMAGES ====================
def classify_image_hybrid(image_path: str, nlp_model, cv_model, file_name: str) -> Dict:
    """
    Classifie une image avec NLP + CV
    Returns: résultat complet de la classification
    """
    # 1. Extraction du texte avec OCR
    text_content = extract_text_from_image(image_path)
    
    if not text_content or len(text_content.strip()) < 20:
        st.warning(f"⚠️ Peu de texte extrait de {file_name}, classification CV uniquement")
        text_content = "Document image sans texte détectable"
    
    # 2. Classification NLP
    nlp_class, nlp_conf = classify_document_page_nlp(nlp_model, text_content)
    
    # 3. Classification CV
    img = Image.open(image_path)
    if cv_model:
        cv_class, cv_conf, cv_probs = cv_model.predict(img)
    else:
        cv_class, cv_conf, cv_probs = "UNKNOWN", 0.0, {}
    
    # 4. Fusion des prédictions
    final_class, final_score, fusion_details = fuse_predictions(
        nlp_class, nlp_conf, cv_class, cv_conf
    )
    
    result = {
        'file': file_name,
        'page': 1,  # Les images sont considérées comme une seule "page"
        'nlp_class': nlp_class,
        'nlp_conf': nlp_conf,
        'cv_class': cv_class,
        'cv_conf': cv_conf,
        'final_class': final_class,
        'final_score': final_score,
        'agreement': fusion_details['agreement'],
        'text_content': text_content
    }
    
    return result

# ==================== INDEXATION AVEC FUSION ====================
def index_documents_hybrid(collection, pages_data: List[Dict], nlp_model, cv_model, 
                          pdf_path: str, progress_callback=None):
    """Indexe les documents avec classification hybride NLP + CV"""
    indexed_count = 0
    skipped_count = 0
    all_results = []
    
    for idx, page_data in enumerate(pages_data):
        if progress_callback:
            progress_callback(idx, len(pages_data), page_data)
        
        page_num = page_data['page_num'] - 1  # 0-indexed pour PyMuPDF
        
        # 1. Classification NLP (texte)
        nlp_class, nlp_conf = classify_document_page_nlp(nlp_model, page_data['content'])
        
        # 2. Classification CV (image)
        img = pdf_page_to_image(pdf_path, page_num)
        if img and cv_model:
            cv_class, cv_conf, cv_probs = cv_model.predict(img)
        else:
            cv_class, cv_conf, cv_probs = "UNKNOWN", 0.0, {}
        
        # 3. Fusion des prédictions
        final_class, final_score, fusion_details = fuse_predictions(
            nlp_class, nlp_conf, cv_class, cv_conf
        )
        
        # Stockage des résultats
        result = {
            'file': page_data['file_name'],
            'page': page_data['page_num'],
            'nlp_class': nlp_class,
            'nlp_conf': nlp_conf,
            'cv_class': cv_class,
            'cv_conf': cv_conf,
            'final_class': final_class,
            'final_score': final_score,
            'agreement': fusion_details['agreement']
        }
        all_results.append(result)
        
        # 4. Génération embedding et indexation
        embedding = generate_embedding(page_data['content'])
        
        if embedding:
            try:
                collection.add(
                    embeddings=[embedding],
                    documents=[page_data['content']],
                    metadatas=[{
                        'file_name': page_data['file_name'],
                        'page_num': page_data['page_num'],
                        'doc_type': final_class,
                        'nlp_class': nlp_class,
                        'nlp_confidence': nlp_conf,
                        'cv_class': cv_class,
                        'cv_confidence': cv_conf,
                        'final_score': final_score,
                        'agreement': fusion_details['agreement'],
                        'indexed_at': datetime.now().isoformat()
                    }],
                    ids=[f"{page_data['file_name']}_page_{page_data['page_num']}_{datetime.now().timestamp()}"]
                )
                indexed_count += 1
            except Exception as e:
                st.error(f"❌ Erreur indexation page {page_data['page_num']}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1
    
    return indexed_count, skipped_count, all_results

# ==================== EMBEDDINGS ====================
def generate_embedding(text: str) -> List[float]:
    """Génère un embedding avec Gemini"""
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        text_truncated = text[:10000] if len(text) > 10000 else text
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text_truncated,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Erreur génération embedding : {e}")
        return None

# ==================== RECHERCHE ====================
def search_documents(collection, query: str, filters: Dict = None, n_results: int = 5):
    """Recherche dans la base vectorielle avec filtres"""
    query_embedding = generate_embedding(query)
    
    if not query_embedding:
        return []
    
    where_filter = None
    if filters and filters.get('doc_type') and filters['doc_type'] != "TOUS":
        where_filter = {"doc_type": {"$eq": filters['doc_type']}}
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        return results
    except Exception as e:
        st.error(f"Erreur recherche : {e}")
        return None

def generate_answer(model, query: str, context: List[str]) -> str:
    """Génère une réponse basée sur le contexte"""
    context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])
    
    prompt = f"""Tu es un assistant spécialisé dans les documents administratifs marocains/francophones.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.

Contexte :
{context_text}

Question : {query}

Réponds de manière précise et structurée en français."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur lors de la génération : {e}"

# ==================== INTERFACE STREAMLIT ====================
def main():
    st.title("🤖 Classification Hybride : NLP (90%) + Computer Vision (10%)")
    st.markdown("**Fusion intelligente de Gemini RAG et ResNet50 pour la classification de documents**")
    st.markdown("---")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        default_api_key = os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Clé API Gemini", 
            value=default_api_key,
            type="password",
            help="Obtenez votre clé sur https://makersuite.google.com/app/apikey"
        )
        
        if not api_key:
            st.warning("⚠️ Veuillez entrer votre clé API Gemini")
            st.stop()
        
        # Initialisation des modèles
        nlp_model = init_gemini(api_key)
        cv_model = load_resnet_model()
        chroma_client = init_chromadb()
        
        st.success("✅ NLP Model (Gemini) chargé")
        
        if cv_model:
            st.success("✅ CV Model (ResNet50) chargé")
        else:
            st.error("❌ ResNet50 non disponible")
        
        # Affichage des poids de fusion
        st.markdown("---")
        st.subheader("⚖️ Poids de Fusion")
        st.metric("NLP (Gemini RAG)", "90%", help="Confiance dans le modèle de langage")
        st.metric("CV (ResNet50)", "10%", help="Confiance dans le modèle visuel")
        
        # Gestion des collections
        st.markdown("---")
        st.header("📚 Collections")
        default_collection = os.getenv("COLLECTION_NAME", "documents_hybrid")
        collection_name = st.text_input("Nom de la collection", value=default_collection)
        
        try:
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Documents avec classification hybride"}
            )
            st.info(f"📊 {collection.count()} documents indexés")
        except Exception as e:
            st.error(f"Erreur collection : {e}")
            st.stop()
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📤 Classification", "🔍 Recherche", "📊 Statistiques"])
    
    # TAB 1 - Classification
    with tab1:
        st.header("Classification Hybride de Documents")
        
        st.info("📋 **Formats supportés** : PDF (multipage) et Images (JPG, PNG, JPEG, BMP, TIFF)")
        
        uploaded_files = st.file_uploader(
            "📁 Chargez vos documents (PDF ou Images)",
            type=['pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Classification automatique avec NLP + CV"
        )
        
        if uploaded_files:
            if 'last_uploaded_files' not in st.session_state:
                st.session_state.last_uploaded_files = []
            
            current_file_names = [f.name for f in uploaded_files]
            if current_file_names != st.session_state.last_uploaded_files:
                st.session_state.last_uploaded_files = current_file_names
                
                st.info("🔄 Classification hybride en cours...")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_results = []
                total_indexed = 0
                total_skipped = 0
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"📄 Traitement de {uploaded_file.name}...")
                    
                    # Détection du type de fichier
                    file_extension = uploaded_file.name.lower().split('.')[-1]
                    is_pdf = file_extension == 'pdf'
                    is_image = file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
                    
                    # Sauvegarde temporaire
                    suffix = '.pdf' if is_pdf else f'.{file_extension}'
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    if is_pdf:
                        # Traitement PDF (existant)
                        pages = extract_text_from_pdf(tmp_path)
                        
                        if pages:
                            def page_progress(current, total, page_data):
                                status_text.text(f"📄 {uploaded_file.name} - Page {page_data['page_num']}/{total}")
                            
                            indexed, skipped, results = index_documents_hybrid(
                                collection, pages, nlp_model, cv_model, tmp_path, page_progress
                            )
                            
                            total_indexed += indexed
                            total_skipped += skipped
                            all_results.extend(results)
                    
                    elif is_image:
                        # Traitement IMAGE (nouveau)
                        st.info(f"🖼️ Classification de l'image {uploaded_file.name}...")
                        
                        # Classification hybride
                        result = classify_image_hybrid(tmp_path, nlp_model, cv_model, uploaded_file.name)
                        all_results.append(result)
                        
                        # Indexation dans ChromaDB
                        if result['text_content'] and len(result['text_content']) > 10:
                            embedding = generate_embedding(result['text_content'])
                            
                            if embedding:
                                try:
                                    collection.add(
                                        embeddings=[embedding],
                                        documents=[result['text_content']],
                                        metadatas=[{
                                            'file_name': uploaded_file.name,
                                            'page_num': 1,
                                            'doc_type': result['final_class'],
                                            'nlp_class': result['nlp_class'],
                                            'nlp_confidence': result['nlp_conf'],
                                            'cv_class': result['cv_class'],
                                            'cv_confidence': result['cv_conf'],
                                            'final_score': result['final_score'],
                                            'agreement': result['agreement'],
                                            'indexed_at': datetime.now().isoformat()
                                        }],
                                        ids=[f"{uploaded_file.name}_{datetime.now().timestamp()}"]
                                    )
                                    total_indexed += 1
                                except Exception as e:
                                    st.error(f"❌ Erreur indexation image : {e}")
                                    total_skipped += 1
                            else:
                                total_skipped += 1
                        else:
                            st.warning(f"⚠️ {uploaded_file.name} : Pas assez de texte pour indexation")
                            total_skipped += 1
                    
                    os.unlink(tmp_path)
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Classification terminée : {total_indexed} pages indexées")
                
                # Affichage des résultats détaillés
                if all_results:
                    st.subheader("📋 Résultats de Classification")
                    
                    for result in all_results:
                        # Détection si c'est une image
                        is_single_image = result['file'].lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                        icon = "🖼️" if is_single_image else "📄"
                        
                        with st.expander(f"{icon} {result['file']} - Page {result['page']}", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**🧠 NLP (Gemini)**")
                                st.write(f"Classe: `{result['nlp_class']}`")
                                st.progress(result['nlp_conf'])
                                st.caption(f"Confiance: {result['nlp_conf']:.2%}")
                            
                            with col2:
                                st.markdown("**👁️ CV (ResNet)**")
                                st.write(f"Classe: `{result['cv_class']}`")
                                st.progress(result['cv_conf'])
                                st.caption(f"Confiance: {result['cv_conf']:.2%}")
                            
                            with col3:
                                st.markdown("**🎯 Résultat Final**")
                                st.write(f"Classe: `{result['final_class']}`")
                                st.progress(result['final_score'])
                                st.caption(f"Score: {result['final_score']:.2%}")
                            
                            # Indicateur d'accord
                            if result['agreement']:
                                st.success("✅ Les deux modèles sont d'accord!")
                            else:
                                st.warning("⚠️ Désaccord entre les modèles (priorité au NLP)")
                            
                            # Si c'est une image, afficher le texte extrait dans le même expander
                            if is_single_image and 'text_content' in result and result['text_content']:
                                st.markdown("---")
                                st.markdown("**📝 Texte extrait (OCR)**")
                                text_preview = result['text_content'][:500] + "..." if len(result['text_content']) > 500 else result['text_content']
                                st.text_area("Contenu", text_preview, height=150, disabled=True, key=f"text_{result['file']}_{result['page']}")
    
    # TAB 2 - Recherche
    with tab2:
        st.header("🔍 Recherche dans les documents")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("🔎 Votre question", placeholder="Ex: Quelles sont mes factures d'eau ?")
        
        with col2:
            doc_filter = st.selectbox(
                "Type de document",
                ["TOUS"] + NLP_CLASSES
            )
        
        n_results = st.slider("Nombre de résultats", 1, 10, 5)
        
        if query and st.button("🔍 Rechercher"):
            with st.spinner("Recherche en cours..."):
                filters = {'doc_type': doc_filter} if doc_filter != "TOUS" else None
                results = search_documents(collection, query, filters, n_results)
                
                if results and results['documents'][0]:
                    st.success(f"✅ {len(results['documents'][0])} résultat(s)")
                    
                    st.subheader("💬 Réponse générée")
                    answer = generate_answer(nlp_model, query, results['documents'][0])
                    st.markdown(answer)
                    
                    st.subheader("📚 Sources")
                    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                        with st.expander(f"Source {i+1} - {metadata['file_name']} (Page {metadata['page_num']})"):
                            st.write(f"**Type:** `{metadata['doc_type']}`")
                            st.write(f"**Score final:** {metadata.get('final_score', 0):.2%}")
                            st.write(f"**NLP:** {metadata.get('nlp_class', 'N/A')} ({metadata.get('nlp_confidence', 0):.2%})")
                            st.write(f"**CV:** {metadata.get('cv_class', 'N/A')} ({metadata.get('cv_confidence', 0):.2%})")
                            st.markdown("---")
                            st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                else:
                    st.warning("Aucun résultat trouvé")
    
    # TAB 3 - Statistiques
    with tab3:
        st.header("📊 Statistiques de la Collection")
        
        try:
            total_docs = collection.count()
            
            if total_docs > 0:
                all_data = collection.get()
                metadatas = all_data['metadatas']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📄 Total documents", total_docs)
                
                # Comptage par type
                type_counts = {}
                agreements = 0
                total_nlp_conf = 0
                total_cv_conf = 0
                total_final_score = 0
                
                for meta in metadatas:
                    doc_type = meta.get('doc_type', 'INCONNU')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    
                    if meta.get('agreement', False):
                        agreements += 1
                    
                    total_nlp_conf += meta.get('nlp_confidence', 0)
                    total_cv_conf += meta.get('cv_confidence', 0)
                    total_final_score += meta.get('final_score', 0)
                
                col2.metric("📊 Types différents", len(type_counts))
                col3.metric("✅ Accord NLP/CV", f"{(agreements/total_docs)*100:.1f}%")
                
                # Scores moyens
                st.subheader("📈 Scores Moyens")
                col1, col2, col3 = st.columns(3)
                col1.metric("🧠 NLP Moyen", f"{(total_nlp_conf/total_docs)*100:.1f}%")
                col2.metric("👁️ CV Moyen", f"{(total_cv_conf/total_docs)*100:.1f}%")
                col3.metric("🎯 Score Final Moyen", f"{(total_final_score/total_docs)*100:.1f}%")
                
                # Répartition par type
                st.subheader("📊 Répartition par Type")
                for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_docs) * 100
                    st.write(f"**{doc_type}** : {count} document(s) ({percentage:.1f}%)")
                
                # Option de suppression
                st.markdown("---")
                if st.button("🗑️ Vider la collection", type="secondary"):
                    if st.checkbox("Confirmer la suppression"):
                        chroma_client.delete_collection(collection_name)
                        st.success("Collection supprimée. Rechargez la page.")
            else:
                st.info("📭 Collection vide. Indexez des documents dans l'onglet Classification.")
                
        except Exception as e:
            st.error(f"Erreur : {e}")

if __name__ == "__main__":
    main()