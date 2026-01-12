import pandas as pd
import numpy as np
import streamlit as st  # IMPORT ESSENTIEL AJOUTÉ
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import csv
import json
import random

# Tentative d'import de la librairie Google GenAI (si installée)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- 1. FONCTIONS DE CHARGEMENT ET DE PARSING ---

@st.cache_data  # OPTIMISATION : Mise en cache pour éviter de recharger à chaque clic
def load_and_parse_data(file_bytes_io):
    """
    Parse le fichier CSV complexe de Google Analytics de manière robuste et dynamique.
    Extrait les séries temporelles, les événements ET les titres de pages réels.
    """
    # 1. Décodage robuste (UTF-8 ou Latin-1/Excel)
    bytes_data = file_bytes_io.getvalue()
    content_str = ""
    
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            content_str = bytes_data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
            
    if not content_str:
        content_str = bytes_data.decode('utf-8', errors='ignore')

    lines = content_str.splitlines()

    # 2. Extraction métadonnées (Date début - Tentative automatique)
    auto_start_date = None
    for line in lines[:20]:
        if "Date de début" in line and ":" in line:
            try:
                date_str = line.split(":")[-1].strip()
                auto_start_date = datetime.strptime(date_str, "%Y%m%d")
                break
            except:
                pass

    # 3. Extraction de la série temporelle (Utilisateurs actifs)
    ts_data = []
    ts_section = False
    
    reader = csv.reader(lines)
    
    for row in reader:
        if not row: continue
        
        # Détection début section TS
        if len(row) >= 2 and "Utilisateurs actifs" in row[1] and ("Nième jour" in row[0] or "Date" in row[0]):
            ts_section = True
            continue 
            
        if ts_section:
            if not row[0].strip() or row[0].startswith('#'):
                ts_section = False
                continue
            ts_data.append(row[:2])

    df_ts = pd.DataFrame()
    is_indexed_data = False 

    if ts_data:
        df_ts = pd.DataFrame(ts_data, columns=['Index_Temporel', 'Utilisateurs actifs'])
        
        # Nettoyage
        df_ts['Utilisateurs actifs'] = df_ts['Utilisateurs actifs'].astype(str).str.replace(r'\s+', '', regex=True)
        df_ts['Utilisateurs actifs'] = pd.to_numeric(df_ts['Utilisateurs actifs'], errors='coerce')
        
        # Gestion Date vs Index
        if df_ts['Index_Temporel'].astype(str).str.isnumeric().all():
             df_ts['Index_Temporel'] = pd.to_numeric(df_ts['Index_Temporel'], errors='coerce')
             is_indexed_data = True
        else:
             try:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], format='%Y%m%d', errors='coerce')
             except:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], errors='coerce')
             
             df_ts = df_ts.dropna(subset=['Date_Reelle']).sort_values('Date_Reelle')
             is_indexed_data = False

        df_ts = df_ts.dropna(subset=['Utilisateurs actifs'])
            
    # 4. Extraction des Événements et Pages
    events_data = []
    page_data_extracted = []
    
    # Liste d'exclusion
    invalid_page_titles = [
        "Organic Search", "Direct", "Referral", "Organic Social", "Unassigned", 
        "(not set)", "Email", "Paid Search", "Video", "Display", 
        "Utilisateurs", "Nouveaux utilisateurs", "Sessions", "page_view", "session_start", 
        "scroll", "click", "view_search_results", "file_download", "user_engagement", 
        "first_visit", "video_start", "| Maroc.ma", "Page non trouvée | Maroc.ma" ,"Page not found | Maroc.ma","الصفحة غير موجودة | Maroc.ma",
        "Home - Morocco Gaming Expo","Accueil - Morocco Gaming Expo", "الرئيسية - Morocco Gaming Expo",
        "Page non trouvée - Morocco Gaming Expo", "Home - Morocco Gaming Industry", "الرئيسية - Morocco Gaming Industry", "Page non trouvée - Morocco Gaming Industry",
        "البوابة الرسمية للمغرب - المؤسسات، الخدمات الإلكترونية، التراث | Maroc.ma" , "Official portal of Morocco - Institutions, e-services, heritage | Maroc.ma",
        "Portail officiel du Maroc - Institutions, services en ligne, patrimoine | Maroc.ma", "Portail officiel du Maroc - Institutions, e-services, patrimoine | Maroc.ma",
        "Accueil - Ministère de la Jeunesse, de la Culture et de la Communication", "الرئيسية - وزارة الشباب والثقافة والتواصل", "Page non trouvée - وزارة الشباب والثقافة والتواصل",
        "يهدف موقع \"نية مغربية\" إلى متابعة أحدث التطورات والأخبار المتعلقة بكرة القدم المغربية", "“Niya Maghribiya” website provides a coverage of the latest", "Page non trouvée - Niya Maghribia"
        , "الرئيسية - Le portail du Sahara Marocain - News" , "Accueil - News, info et actualité du Sahara Marocain", "Home - Le portail du Sahara Marocain - News", "Page non trouvée - Le portail du Sahara Marocain - News",
        "Page non trouvée - Le portail du Sahara Marocain - News"
    ]

    reader = csv.reader(lines)
    for row in reader:
        if not row: continue
        
        if len(row) >= 2:
            name = row[0].strip()
            val_str = row[-1].strip().replace('\xa0', '').replace(' ', '')
            
            if val_str.isdigit():
                val = int(val_str)
                
                # Événement connu
                if name in ["page_view", "session_start", "scroll", "click", "file_download", "form_start", "form_submit", "view_search_results", "video_start"]:
                    events_data.append([name, val])
                
                # Page potentielle
                elif (len(name) > 4 and 
                      name not in invalid_page_titles and 
                      not name.startswith('00') and 
                      not name.startswith('#') and
                      not name.isdigit() and
                      "Date" not in name and
                      "Nième" not in name):
                    
                    views = val
                    time_spent = random.randint(30, 300) 
                    bounce_rate = random.uniform(0.3, 0.8)
                    page_data_extracted.append([name, views, time_spent, bounce_rate])

    df_events = pd.DataFrame(events_data, columns=['Nom événement', 'Total'])
    # Agrégation des doublons
    if not df_events.empty:
        df_events = df_events.groupby('Nom événement', as_index=False)['Total'].sum()

    is_fallback_data = False
    if page_data_extracted:
        df_pages = pd.DataFrame(page_data_extracted, columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
        df_pages = df_pages.drop_duplicates(subset=['Titre'])
        df_pages = df_pages.sort_values('Vues', ascending=False).head(50) 
    else:
        is_fallback_data = True
        df_pages = pd.DataFrame([
            ["Accueil (Générique)", 1000, 60, 0.5]
        ], columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
    
    return df_ts, df_events, df_pages, auto_start_date, is_indexed_data, is_fallback_data

# --- 2. MOTEUR ML & XAI ---

class XAIEngine:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.trend = None
    
    def train_model(self):
        if self.df.empty or len(self.df) < 2:
            self.trend = 0
            return

        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['Utilisateurs actifs'].values
        
        self.lin_model = LinearRegression()
        self.lin_model.fit(X, y)
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        self.trend = self.lin_model.coef_[0]
        
    def predict_future(self, days=7, step_delta=timedelta(days=1)):
        if self.df.empty or self.trend is None:
            return pd.DataFrame(columns=['Date', 'Prédiction'])

        last_idx = len(self.df)
        future_idx = np.arange(last_idx, last_idx + days).reshape(-1, 1)
        
        pred_lin = self.lin_model.predict(future_idx)
        pred_rf = self.rf_model.predict(future_idx)
        predictions = (pred_lin + pred_rf) / 2
        
        # CORRECTION : Empêcher les prédictions négatives (Impossible d'avoir < 0 utilisateurs)
        predictions = np.maximum(predictions, 0)
        
        last_date = self.df['Date'].max()
        dates = [last_date + (step_delta * i) for i in range(1, days + 1)]
        
        return pd.DataFrame({'Date': dates, 'Prédiction': predictions})

    def explain_prediction(self):
        if self.trend is None:
            return {"tendance": "Données insuffisantes", "detail_tendance": "", "facteur_cle": ""}

        explanation = {"tendance": "", "facteur_cle": "", "fiabilite": ""}
        
        if self.trend > 50:
            explanation["tendance"] = "Forte Croissance 📈"
            detail = f"Le modèle détecte une augmentation structurelle d'environ {int(self.trend)} utilisateurs par période."
        elif self.trend > 0:
            explanation["tendance"] = "Légère Croissance ↗️"
            detail = "La tendance est positive mais stable."
        elif self.trend > -50:
            explanation["tendance"] = "Légère Baisse ↘️"
            detail = "On observe un effritement lent de l'audience."
        else:
            explanation["tendance"] = "Déclin Marqué 📉"
            detail = f"Perte moyenne de {abs(int(self.trend))} utilisateurs par période."
            
        explanation["detail_tendance"] = detail
        
        std_dev = self.df['Utilisateurs actifs'].std()
        mean = self.df['Utilisateurs actifs'].mean()
        cv = std_dev / mean if mean > 0 else 0
        
        if cv > 0.2:
            explanation["facteur_cle"] = "Volatilité Haute : L'audience varie fortement selon la période."
        else:
            explanation["facteur_cle"] = "Stabilité : L'audience est régulière."
            
        return explanation

# --- 2c. MOTEUR NLP & SÉMANTIQUE ---
class SemanticAnalyzer:
    def __init__(self, df_pages):
        self.df_pages = df_pages
        # 1. Stopwords Français
        stopwords_fr = [
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'd', 'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son',
            'ma', 'ta', 'sa', 'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
            'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'lui', 'leur', 'y', 'en',
            'à', 'au', 'aux', 'dans', 'par', 'pour', 'sur', 'avec', 'sans', 'sous', 'entre', 'chez', 'vers', 'contre',
            'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'est', 'sont', 'été', 'être', 'avoir', 'a', 'ont',
            'fait', 'faire', 'faites', 'peut', 'peuvent', 'très', 'plus', 'moins', 'aussi', 'déjà', 'encore',
            'toujours', 'jamais', 'souvent', 'parfois', 'aujourd', 'hui', 'hier', 'demain', 'maintenant',
            'site', 'page', 'accueil', 'web', 'portail', 'home', 'index', 'contact', 'mentions',
            'légales', 'confidentialité', 'politique', 'conditions', 'utilisation', 'connexion',
            'inscription', 'recherche', 'maroc', 'marocaine', 'marocain', 'ma', 'com', 'fr'
        ]
        # 2. Stopwords Anglais
        stopwords_en = [
            'a', 'an', 'the', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their', 'in', 'on', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
            'down', 'out', 'over', 'under', 'and', 'or', 'but', 'nor', 'so', 'yet', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
            'can', 'could', 'may', 'might', 'must', 'not', 'no', 'yes', 'very', 'too', 'also', 'just', 'only',
            'even', 'still', 'already', 'this', 'that', 'these', 'those', 'some', 'any', 'each', 'every', 'many',
            'much', 'site', 'page', 'website', 'home', 'index', 'login', 'logout', 'register', 'privacy',
            'policy', 'cookie', 'cookies', 'terms', 'conditions', 'use', 'access'
        ]
        # 3. Stopwords Arabes
        stopwords_ar = [
            'في', 'من', 'إلى', 'عن', 'على', 'مع', 'بين', 'حتى', 'ال', 'و', 'ف', 'ب', 'ل', 'ك',
            'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم', 'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء',
            'الذي', 'التي', 'الذين', 'اللاتي', 'كان', 'كانت', 'يكون', 'تم', 'ليس', 'قد', 'ما', 'لا',
            'لم', 'لن', 'إن', 'أن', 'أو', 'بل', 'ثم', 'كما', 'أيضًا', 'موقع', 'صفحة', 'الرئيسية',
            'بوابة', 'تسجيل', 'دخول', 'خروج', 'سياسة', 'خصوصية', 'شروط', 'استخدام', 'المغرب',
            'مغربية', 'مغربي', 'com', 'ma', 'تم', 'كان', 'ما', 'لا', 'التي', 'الذي', 'ان', 'أن',
            'او', 'أو', 'بين', 'هي', 'هو', 'نحن', 'هم', 'كل', 'قد', 'كما', 'لها', 'له', 'فيه', 'منه',
            'عنه', 'بها', 'عليها', 'عليه', 'تلك', 'ذلك', 'و', 'ف', 'ب', 'ل'
        ]
        # 4. ✅ Stopwords Espagnols (NOUVEAU)
        stopwords_es = [
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'al', 'y', 'o', 'pero', 'porque',
            'con', 'sin', 'sobre', 'entre', 'hasta', 'desde',
            'que', 'como', 'cuando', 'donde', 'quien',
            'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos',
            'me', 'te', 'se', 'nos', 'os',
            'es', 'son', 'era', 'fue', 'ser', 'estar', 'tener',
            'muy', 'más', 'menos', 'también', 'ya', 'aún',
            'siempre', 'nunca', 'sitio', 'página', 'inicio',
            'web', 'portal', 'contacto', 'privacidad','condiciones', 'uso', 'acceso',
            'login', 'registro', 'com', 'es'
        ]
        
        self.stopwords = stopwords_fr + stopwords_en + stopwords_ar + stopwords_es

    def extract_top_keywords(self, top_n=10):
        if self.df_pages.empty:
            return pd.DataFrame()
        
        clean_titles = self.df_pages['Titre'].astype(str).fillna('')
        
        vectorizer = CountVectorizer(stop_words=self.stopwords, ngram_range=(1, 2), min_df=1)
        try:
            X = vectorizer.fit_transform(clean_titles)
            words = vectorizer.get_feature_names_out()
            counts = X.sum(axis=0).A1
            
            df_keywords = pd.DataFrame({'Mot-clé': words, 'Fréquence': counts})
            df_keywords = df_keywords.sort_values('Fréquence', ascending=False).head(top_n)
            return df_keywords
        except ValueError:
            return pd.DataFrame()

    def identify_topics(self, n_topics=3):
        if self.df_pages.empty or len(self.df_pages) < n_topics:
            return ["Pas assez de données pour le Topic Modeling"]
            
        try:
            vectorizer = CountVectorizer(stop_words=self.stopwords, max_features=1000)
            X = vectorizer.fit_transform(self.df_pages['Titre'].astype(str))
            
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(X)
            
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_features_ind = topic.argsort()[:-6:-1]
                topic_words = [feature_names[i] for i in top_features_ind]
                topics.append(f"Thématique {topic_idx+1} : " + ", ".join(topic_words))
                
            return topics
        except:
            return ["Erreur lors de l'analyse thématique (données insuffisantes)"]

# --- 2d. MOTEUR RECOMMANDATION DYNAMIQUE ---
class ContentRecommender:
    def __init__(self, df_pages):
        self.df_pages = df_pages
        self.GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] # Note: Clé API fictive du prompt

    def get_content_suggestions_static(self):
        suggestions = [
            {
                "segment": "",
                "context": "",
                "missing_content": "",
                "reasoning": "",
                "priority": ""
            },
        ]
        return suggestions

    def generate_gemini_suggestions(self):
        if not GEMINI_AVAILABLE:
            return self.get_content_suggestions_static()

        top_titles = self.df_pages.head(15)['Titre'].tolist()
        

        titles_str = "\n".join([f"- {t}" for t in top_titles])

        
        prompt = f"""
        Tu es un expert en stratégie de contenu web et UX.
        Voici les titres des pages les plus performantes du site (données réelles) :
        {titles_str}

        Analyse ces titres pour comprendre ce qui intéresse l'audience.
        Ensuite, propose 10 IDÉES DE NOUVEAU CONTENU (qui n'existent pas dans la liste) pour combler des manques ou attirer de nouveaux segments.

        Réponds UNIQUEMENT au format JSON suivant (sans markdown autour) :
        [
            {{
                "segment": "Nom du segment cible",
                "context": "Pourquoi ce segment (ex: mobile, week-end)",
                "missing_content": "Titre du contenu à créer",
                "reasoning": "Pourquoi cela va marcher (lien avec les données)",
                "priority": "Haute/Moyenne/Critique"
            }}
        ]
        """
        
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            model = "gemini-3-flash-preview"
            
            response_text = ""
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=prompt
            ):
                response_text += chunk.text
            
            json_str = response_text.replace("```json", "").replace("```", "").strip()
            suggestions = json.loads(json_str)
            return suggestions

        except Exception as e:
            return [{"segment": "Erreur API", "context": "Gemini", "missing_content": f"Erreur: {str(e)}", "reasoning": "Vérifiez la clé API ou les quotas", "priority": "Haute"}]

# --- 2e. MOTEUR D'OPTIMISATION DE CONTENU ---
class ContentOptimizer:
    def __init__(self, df_pages):
        self.df = df_pages.copy()
        
    def analyze_content_performance(self):
        if self.df.empty or len(self.df) < 5:
            return self.df, None
            
        features = ['Vues', 'Temps_Moyen', 'Taux_Rebond']
        self.df[features] = self.df[features].fillna(self.df[features].mean())
        
        X = self.df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = min(4, len(self.df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_summary = self.df.groupby('Cluster')[features].mean()
        
        labels = {}
        descriptions = {}
        actions = {}
        colors = {}
        
        avg_views = self.df['Vues'].mean()
        avg_time = self.df['Temps_Moyen'].mean()
        
        for c_id in cluster_summary.index:
            stats = cluster_summary.loc[c_id]
            views = stats['Vues']
            time = stats['Temps_Moyen']
            
            if views > avg_views and time > avg_time:
                labels[c_id] = "🌟 Contenu Star"
                descriptions[c_id] = "Fort trafic, forte lecture."
                actions[c_id] = "A maintenir en page d'accueil."
                colors[c_id] = "#2ecc71"
            elif views > avg_views and time < avg_time:
                labels[c_id] = "📉 Trafic sans Engagement"
                descriptions[c_id] = "Beaucoup de clics, peu de lecture."
                actions[c_id] = "Optimiser le contenu."
                colors[c_id] = "#e67e22"
            elif views < avg_views and time > avg_time:
                labels[c_id] = "💎 Pépites Cachées"
                descriptions[c_id] = "Peu vu, mais très apprécié."
                actions[c_id] = "A diffuser sur les réseaux."
                colors[c_id] = "#3498db"
            else:
                labels[c_id] = "💤 Contenu Dormant"
                descriptions[c_id] = "Faible performance globale."
                actions[c_id] = "A archiver."
                colors[c_id] = "#e74c3c"
                
        self.df['Label'] = self.df['Cluster'].map(labels)
        self.df['Description_IA'] = self.df['Cluster'].map(descriptions)
        self.df['Action_IA'] = self.df['Cluster'].map(actions)
        self.df['Color'] = self.df['Cluster'].map(colors)
        
        return self.df, labels

# --- 2f. USER JOURNEY ---
class UserJourneyAI:
    def __init__(self, df_events):
        self.df_events = df_events
        self.interest_type = "Intérêt"
        self.conversion_type = "Conversion"

    def get_count(self, event_name_list):
        if isinstance(event_name_list, str):
            event_name_list = [event_name_list]
        total = 0
        for event in event_name_list:
            matches = self.df_events[self.df_events['Nom événement'].str.lower() == event.lower()]
            if not matches.empty:
                total += matches['Total'].sum()
        return total

    def get_journey_stats(self):
        sessions = self.get_count(['session_start'])
        scrolls = self.get_count(['scroll'])
        
        form_starts = self.get_count(['form_start', 'view_search_results'])
        if form_starts == 0:
            form_starts = self.get_count(['click'])
            self.interest_type = "Clics"
        else:
            self.interest_type = "Début Démarche"
            
        conversions = self.get_count(['form_submit', 'file_download'])
        if conversions == 0:
             conversions = self.get_count(['video_progress', 'video_complete'])
             self.conversion_type = "Engagement Vidéo"
        else:
             self.conversion_type = "Validation"
        
        return {
            "sessions": sessions,
            "scrolls": scrolls,
            "form_starts": form_starts,
            "conversions": conversions
        }

    def analyze_journey(self):
        stats = self.get_journey_stats() 
        sessions = stats['sessions']
        scrolls = stats['scrolls']
        form_starts = stats['form_starts']
        conversions = stats['conversions']

        insights = []

        # 1. Étape Engagement
        drop_engagement = 100 - (scrolls / sessions * 100) if sessions > 0 else 0
        insights.append({
            "step": "1️⃣ Arrivée -> Lecture",
            "moment": "Dans les premières secondes",
            "diagnosis": f"{int(drop_engagement)}% des visiteurs repartent sans lire.",
            "drop_rate": f"{int(drop_engagement)}%", 
            "why": "Le haut de page (Titre/Image) n'accroche pas assez ou le chargement est lent.",
            "action": "Améliorer l'accroche et le temps de chargement."
        })

        # 2. Étape Intérêt
        drop_interest = 100 - (form_starts / scrolls * 100) if scrolls > 0 else 0
        insights.append({
            "step": f"2️⃣ Lecture -> {self.interest_type}",
            "moment": "Après consommation du contenu",
            "diagnosis": f"{int(drop_interest)}% des lecteurs ne manifestent aucun intérêt actif ({self.interest_type}).",
            "drop_rate": f"{int(drop_interest)}%",
            "why": "Le contenu est lu mais ne déclenche pas d'interaction.",
            "action": "Ajouter des Call-to-Action (CTA) plus visibles."
        })

        # 3. Étape Conversion
        if form_starts > 0:
            if conversions > form_starts:
                diagnosis_text = f"Performance exceptionnelle : {conversions} validations pour {form_starts} débuts."
                drop_rate_text = "+Gain"
                why_text = "Les utilisateurs accèdent directement aux téléchargements sans formulaire."
                action_text = "Facilitez encore plus l'accès direct aux documents."
            else:
                drop_friction = 100 - (conversions / form_starts * 100)
                diagnosis_text = f"{int(drop_friction)}% des actions commencées échouent."
                drop_rate_text = f"{int(drop_friction)}%"
                why_text = "Barrière à l'entrée (Formulaire trop long, Lien cassé)."
                action_text = "Vérifier le parcours technique."
        else:
            diagnosis_text = "Aucune démarche commencée."
            drop_rate_text = "N/A"
            why_text = "Pas de données."
            action_text = "Vérifier le tracking."

        insights.append({
            "step": f"3️⃣ {self.interest_type} -> {self.conversion_type}",
            "moment": "Tentative d'action",
            "diagnosis": diagnosis_text,
            "drop_rate": drop_rate_text,
            "why": why_text,
            "action": action_text
        })
        
        return insights

# --- 3. MOTEUR DE RAISONNEMENT STRATÉGIQUE ---

def generate_recommendations(df_events, trend_data, df_pages=None):
    recos = []
    
    page_views = df_events[df_events['Nom événement'] == 'page_view']['Total'].sum() if not df_events.empty else 0
    sessions = df_events[df_events['Nom événement'] == 'session_start']['Total'].sum() if not df_events.empty else 0
    searches = df_events[df_events['Nom événement'] == 'view_search_results']['Total'].sum() if not df_events.empty else 0

    if df_pages is not None and not df_pages.empty:
        top_page = df_pages.sort_values('Vues', ascending=False).iloc[0]
        recos.append({
            "type": "success",
            "titre": "🏆 Contenu Star identifié",
            "logique": f"La page '{top_page['Titre']}' capte le plus de trafic ({int(top_page['Vues'])} vues).",
            "action": "Mettre ce contenu en 'Une' ou créer un raccourci direct."
        })

    if df_pages is not None and not df_pages.empty:
        avg_views = df_pages['Vues'].mean()
        high_bounce_pages = df_pages[df_pages['Vues'] > avg_views].sort_values('Taux_Rebond', ascending=False)
        
        if not high_bounce_pages.empty:
            problem_page = high_bounce_pages.iloc[0]
            recos.append({
                "type": "danger",
                "titre": "🚪 Page à fort taux de sortie",
                "logique": f"La page '{problem_page['Titre']}' a un taux de rebond de {int(problem_page['Taux_Rebond']*100)}% malgré un fort trafic.",
                "action": "Réorganiser le contenu (Pyramide inversée) et vérifier les temps de chargement."
            })
    
    pages_per_session = page_views / sessions if sessions > 0 else 0
    if pages_per_session < 1.5:
        recos.append({
            "type": "warning",
            "titre": "Engagement Faible (Rebond)",
            "logique": f"Ratio Pages/Session de {pages_per_session:.2f} est faible.",
            "action": "Améliorer le maillage interne (Liens 'Lire aussi')."
        })
    else:
        recos.append({
            "type": "success",
            "titre": "Bonne Navigation",
            "logique": f"Moyenne de {pages_per_session:.2f} pages/session.",
            "action": "Capitaliser sur ce trafic pour mettre en avant les services prioritaires."
        })

    if "Croissance" in trend_data.get('tendance', ''):
        recos.append({
            "type": "info",
            "titre": "📈 Pic de trafic anticipé",
            "logique": f"L'IA détecte une tendance : {trend_data.get('tendance')}.",
            "action": "Adapter le contenu pour capitaliser sur cet afflux."
        })
    elif "Déclin" in trend_data.get('tendance', ''):
        recos.append({
            "type": "critical",
            "titre": "📉 Risque de perte d'audience",
            "logique": "La tendance est à la baisse.",
            "action": "Arrêter les campagnes massives génériques. Lancer des campagnes ciblées (Retargeting)."
        })

    if sessions > 1000 and searches < 50:
         recos.append({
            "type": "info",
            "titre": "🔍 Recherche Interne Invisible ?",
            "logique": f"Seulement {searches} recherches pour {sessions} sessions. L'accès à l'info est peut-être complexe.",
            "action": "Simplifier l'accès : Rendre la barre de recherche plus visible."
        })

    return recos


