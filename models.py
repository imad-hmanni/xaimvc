import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import csv
import json

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- 1. FONCTIONS DE CHARGEMENT ET DE PARSING ---

@st.cache_data
def load_and_parse_data(file_bytes_io):
    bytes_data = file_bytes_io.getvalue()
    content_str = ""
    
    encodings = ['utf-8-sig', 'utf-16', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            content_str = bytes_data.decode(encoding)
            if "Utilisateurs" in content_str or "Vues" in content_str or "page_" in content_str:
                break
        except UnicodeDecodeError:
            continue
            
    if not content_str:
        content_str = bytes_data.decode('utf-8', errors='ignore')

    lines = content_str.splitlines()

    auto_start_date = None
    auto_end_date = None
    
    for line in lines[:20]:
        if "Date de début" in line and ":" in line:
            try:
                date_str = line.split(":")[-1].strip()
                auto_start_date = datetime.strptime(date_str, "%Y%m%d")
            except: pass
        if "Date de fin" in line and ":" in line:
            try:
                date_str = line.split(":")[-1].strip()
                auto_end_date = datetime.strptime(date_str, "%Y%m%d")
            except: pass

    file_format = "snapshot"
    header_map = {}
    data_start_idx = 0
    
    for i, line in enumerate(lines[:50]):
        if ("Chemin de la page" in line or "Titre de la page" in line) and "Vues" in line:
            file_format = "detailed_report"
            reader = csv.reader([line])
            headers = next(reader)
            for idx, col in enumerate(headers):
                c = col.lower().strip()
                if "chemin" in c or "titre" in c: header_map['title'] = idx
                elif "vues" in c and "utilisateur" not in c: header_map['views'] = idx
                elif "durée" in c and "engagement" in c: header_map['time'] = idx
                elif "rebond" in c: header_map['bounce'] = idx
                elif "vues par utilisateur" in c: header_map['views_per_user'] = idx
            data_start_idx = i + 1
            break

    df_ts = pd.DataFrame()
    df_events = pd.DataFrame()
    df_pages = pd.DataFrame()
    
    if file_format == "detailed_report":
        page_data_extracted = []
        reader = csv.reader(lines[data_start_idx:])
        
        for row in reader:
            if not row or len(row) < 2: continue
            try:
                name = row[header_map['title']].strip() if 'title' in header_map else "Inconnu"
                
                views = 0
                if 'views' in header_map:
                    v_str = row[header_map['views']].replace('\xa0', '').replace(' ', '')
                    if v_str.isdigit(): views = int(v_str)
                
                time_spent = 0
                if 'time' in header_map:
                    t_str = row[header_map['time']].strip()
                    try:
                        if '.' in t_str or (',' in t_str and ':' not in t_str):
                            time_spent = float(t_str.replace(',', '.'))
                        elif 'm' in t_str and 's' in t_str: 
                            parts = t_str.replace('s','').split('m')
                            time_spent = int(parts[0])*60 + int(parts[1])
                        elif ':' in t_str:
                            parts = t_str.split(':')
                            if len(parts) == 3: time_spent = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                            elif len(parts) == 2: time_spent = int(parts[0])*60 + int(parts[1])
                    except:
                        time_spent = 0
                
                bounce_rate = 0
                if 'bounce' in header_map:
                    try:
                        b_str = row[header_map['bounce']].strip().replace('%', '').replace(',', '.')
                        bounce_rate = float(b_str)
                        if bounce_rate <= 1 and bounce_rate > 0: bounce_rate = bounce_rate * 100
                    except:
                        bounce_rate = 0
                else:
                    views_per_user = 1.0
                    if 'views_per_user' in header_map:
                        try:
                            vpu_str = row[header_map['views_per_user']].strip().replace(',', '.')
                            views_per_user = float(vpu_str)
                        except: pass
                    
                    if views_per_user >= 1:
                        bounce_rate = max(0.1, min(0.9, 1 / views_per_user))
                        
                    if time_spent > 30:
                        bounce_rate *= 0.7 
                    elif time_spent < 5:
                        bounce_rate = max(bounce_rate, 0.9)
                    
                    bounce_rate = bounce_rate * 100

                page_data_extracted.append([name, views, time_spent, bounce_rate])
            except (IndexError, ValueError):
                continue

        if page_data_extracted:
            df_pages = pd.DataFrame(page_data_extracted, columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
            df_pages = df_pages.sort_values('Vues', ascending=False).head(100) 
            
            dates = []
            if auto_start_date and auto_end_date:
                delta_days = (auto_end_date - auto_start_date).days
                if delta_days >= 0:
                    dates = [auto_start_date + timedelta(days=i) for i in range(delta_days + 1)]
            
            if not dates:
                base_date = datetime.now()
                dates = [base_date - timedelta(days=i) for i in range(30)]
                dates.reverse() 
                
            df_ts = pd.DataFrame({
                'Date_Reelle': dates,
                'Utilisateurs actifs': [0] * len(dates) 
            })
            
            total_views = df_pages['Vues'].sum()
            df_events = pd.DataFrame([['page_view', total_views]], columns=['Nom événement', 'Total'])
            
            return df_ts, df_events, df_pages, auto_start_date, 1, False

    ts_data = []
    ts_section = False
    time_step = 1 
    
    events_data = []
    page_data_raw = []
    
    reader = csv.reader(lines)
    
    invalid_page_titles = [
        "Organic Search", "Direct", "Referral", "Organic Social", "Unassigned", 
        "(not set)", "Email", "Paid Search", "Video", "Display", 
        "Utilisateurs", "Nouveaux utilisateurs", "Sessions", "page_view", "session_start", 
        "scroll", "click", "view_search_results", "file_download", "user_engagement", 
        "first_visit", "video_start"
    ]

    for row in reader:
        if not row: continue
        
        if len(row) >= 2:
            col0_lower = row[0].strip().lower()
            col1_lower = row[1].strip().lower()
            
            if any(k in col1_lower for k in ["utilisateurs", "users"]) and any(kw in col0_lower for kw in ["nième", "nth", "date", "jour", "day", "semaine", "week", "mois", "month"]):
                ts_section = True
                if "semaine" in col0_lower or "week" in col0_lower: time_step = 7
                elif "mois" in col0_lower or "month" in col0_lower: time_step = 30
                else: time_step = 1
                continue 
            
        if ts_section:
            if not row[0].strip() or row[0].startswith('#'):
                ts_section = False
            else:
                ts_data.append(row[:2])
                continue

        if len(row) >= 2:
            name = row[0].strip().replace('\xa0', ' ')
            val_str = row[-1].strip().replace('\xa0', '').replace(' ', '')
            
            if val_str.isdigit():
                val = int(val_str)
                
                if name in ["page_view", "session_start", "scroll", "click", "file_download", "form_start", "form_submit", "view_search_results", "video_start", "user_engagement", "first_visit"]:
                    events_data.append([name, val])
                
                elif (len(name) > 4 and 
                      name not in invalid_page_titles and 
                      not name.startswith('00') and 
                      not name.startswith('#') and
                      not name.isdigit() and
                      "Date" not in name and
                      "Nième" not in name):
                    
                    page_data_raw.append({'row': row, 'name': name, 'views': val})

    if ts_data:
        df_ts = pd.DataFrame(ts_data, columns=['Index_Temporel', 'Utilisateurs actifs'])
        df_ts['Utilisateurs actifs'] = pd.to_numeric(df_ts['Utilisateurs actifs'].astype(str).str.replace(r'\s+', '', regex=True), errors='coerce')
        if df_ts['Index_Temporel'].astype(str).str.isnumeric().all():
             df_ts['Index_Temporel'] = pd.to_numeric(df_ts['Index_Temporel'], errors='coerce')
        else:
             try:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], format='%Y%m%d', errors='coerce')
             except:
                df_ts['Date_Reelle'] = pd.to_datetime(df_ts['Index_Temporel'], errors='coerce')
             df_ts = df_ts.dropna(subset=['Date_Reelle']).sort_values('Date_Reelle')
             time_step = 0 
        df_ts = df_ts.dropna(subset=['Utilisateurs actifs'])

    df_events = pd.DataFrame(events_data, columns=['Nom événement', 'Total'])
    if not df_events.empty:
        df_events = df_events.groupby('Nom événement', as_index=False)['Total'].sum()

    global_bounce_rate = 50.0 
    if not df_events.empty:
        total_sessions = df_events[df_events['Nom événement'] == 'session_start']['Total'].sum()
        total_scrolls = df_events[df_events['Nom événement'] == 'scroll']['Total'].sum()
        if total_sessions > 0 and total_scrolls > 0:
            bounce_val = max(0, min(1, 1 - (total_scrolls / total_sessions)))
            global_bounce_rate = bounce_val * 100

    page_data_processed = []
    for item in page_data_raw:
        row = item['row']
        name = item['name']
        views = item['views']
        time_spent = 0
        bounce_rate = 0
        
        if len(row) > 2:
             for col in row[1:]:
                 col_str = col.strip()
                 if ':' in col_str or ('m' in col_str and 's' in col_str):
                     try:
                         if 'm' in col_str and 's' in col_str:
                             parts = col_str.replace('s','').split('m')
                             time_spent = int(parts[0])*60 + int(parts[1])
                         elif ':' in col_str:
                             parts = col_str.split(':')
                             if len(parts) == 3: time_spent = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                             elif len(parts) == 2: time_spent = int(parts[0])*60 + int(parts[1])
                     except: pass
                 elif '%' in col_str:
                     try:
                         val_pct = float(col_str.replace('%', '').replace(',', '.').strip())
                         if val_pct <= 1: val_pct = val_pct * 100
                         bounce_rate = val_pct
                     except: pass
        
        if bounce_rate == 0: bounce_rate = global_bounce_rate
        page_data_processed.append([name, views, time_spent, bounce_rate])

    is_fallback_data = False
    if page_data_processed:
        df_pages = pd.DataFrame(page_data_processed, columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
        df_pages = df_pages.drop_duplicates(subset=['Titre'])
        df_pages = df_pages.sort_values('Vues', ascending=False).head(50) 
    else:
        is_fallback_data = True
        df_pages = pd.DataFrame([["Accueil (Générique)", 1000, 60, 50.0]], columns=['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond'])
    
    return df_ts, df_events, df_pages, auto_start_date, time_step, is_fallback_data

# --- 2. MOTEUR ML & XAI ---

class XAIEngine:
    def __init__(self, df):
        self.df = df if (df is not None and not df.empty and 'Utilisateurs actifs' in df.columns) else pd.DataFrame()
        self.model = None
        self.trend = None
    
    def is_valid(self):
        return not self.df.empty and len(self.df) >= 2 and 'Utilisateurs actifs' in self.df.columns

    def train_model(self):
        if not self.is_valid():
            self.trend = None
            return

        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['Utilisateurs actifs'].values
        
        self.lin_model = LinearRegression()
        self.lin_model.fit(X, y)
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        
        self.trend = self.lin_model.coef_[0]
        
    def predict_future(self, days=90, step_delta=timedelta(days=1)):
        if not self.is_valid() or self.trend is None:
            return pd.DataFrame(columns=['Date', 'Prédiction'])

        step_days = step_delta.days
        steps = int(days / step_days) if step_days > 0 else days
        if steps < 1: steps = 1

        last_idx = len(self.df)
        future_idx = np.arange(last_idx, last_idx + steps).reshape(-1, 1)
        
        pred_lin = self.lin_model.predict(future_idx)
        pred_rf = self.rf_model.predict(future_idx)
        predictions = (pred_lin + pred_rf) / 2
        
        predictions = np.maximum(predictions, 0)
        
        last_date = self.df['Date'].max()
        dates = [last_date + (step_delta * i) for i in range(1, steps + 1)]
        
        return pd.DataFrame({'Date': dates, 'Prédiction': predictions})

    def get_historical_trend(self):
        if not self.is_valid() or self.trend is None:
            return pd.DataFrame(columns=['Date', 'Tendance_IA'])
        X = np.arange(len(self.df)).reshape(-1, 1)
        pred_lin = self.lin_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        pred_trend = (pred_lin + pred_rf) / 2
        
        return pd.DataFrame({'Date': self.df['Date'], 'Tendance_IA': pred_trend})

    def explain_prediction(self):
        if not self.is_valid() or self.trend is None:
            return {"tendance": "Données insuffisantes ou absentes", "detail_tendance": "Le fichier exporté ne contient pas de données temporelles reconnues.", "facteur_cle": ""}

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
        self.GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") 
        self.stopwords = [
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'd', 'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son',
            'ma', 'ta', 'sa', 'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos', 'vos', 'leurs',
            'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'lui', 'leur', 'y', 'en',
            'à', 'au', 'aux', 'dans', 'par', 'pour', 'sur', 'avec', 'sans', 'sous', 'entre', 'chez', 'vers', 'contre',
            'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'est', 'sont', 'été', 'être', 'avoir', 'a', 'ont',
            'fait', 'faire', 'faites', 'peut', 'peuvent', 'très', 'plus', 'moins', 'aussi', 'déjà', 'encore',
            'toujours', 'jamais', 'souvent', 'parfois', 'aujourd', 'hui', 'hier', 'demain', 'maintenant',
            'site', 'page', 'accueil', 'web', 'portail', 'home', 'index', 'contact', 'mentions',
            'légales', 'confidentialité', 'politique', 'conditions', 'utilisation', 'connexion',
            'inscription', 'recherche', 'maroc', 'marocaine', 'marocain', 'ma', 'com', 'fr',
            'a', 'an', 'the', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their', 'in', 'on', 'of', 'at', 'by', 'for', 'with', 'about',
            'في', 'من', 'إلى', 'عن', 'على', 'مع', 'بين', 'حتى', 'ال', 'و', 'ف', 'ب', 'ل', 'ك',
            'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم', 'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء'
        ]

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

    def identify_topics(self, n_topics=5):
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

    def explain_topics(self, topics_list):
        if not GEMINI_AVAILABLE or not self.GEMINI_API_KEY:
            return {}
        
        topics_str = "\n".join(topics_list)
        prompt = f"""
        Tu es un expert en analyse sémantique et comportement utilisateur.
        Voici des "Thématiques" (clusters de mots-clés) extraites des pages les plus visitées d'un portail web :
        {topics_str}

        Pour chaque thématique, analyse les mots-clés et rédige un petit paragraphe (2 à 3 lignes) expliquant clairement de quel sujet concret il s'agit et ce qui intéresse vraiment les visiteurs derrière ces mots.

        Réponds STRICTEMENT au format JSON suivant :
        {{
            "Thématique 1": "Ton explication de 2-3 lignes ici...",
            "Thématique 2": "Ton explication de 2-3 lignes ici..."
        }}
        """
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            model = "gemini-2.5-flash"
            response = client.models.generate_content(model=model, contents=prompt)
            json_str = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception as e:
            return {}

    # --- NOUVELLE FONCTION : GÉNÉRATION DE MOTS-CLÉS PUISSANTS ---
    def generate_powerful_keywords(self, top_keywords, topics_list):
        if not GEMINI_AVAILABLE or not self.GEMINI_API_KEY:
            return []
        
        kw_str = ", ".join(top_keywords) if isinstance(top_keywords, list) else str(top_keywords)
        topics_str = "\n".join(topics_list)
        
        prompt = f"""
        Tu es un expert SEO et stratégie de contenu digital.
        
        Voici les mots-clés organiques qui performent déjà très bien sur le site : 
        [{kw_str}]
        
        Voici les grandes thématiques qui intéressent l'audience actuelle : 
        {topics_str}
        
        En te basant sur ce contexte existant, propose 30 NOUVEAUX mots-clés très puissants que le site devrait impérativement cibler dans ses prochains contenus pour générer encore plus de trafic ciblé.
        CONTRAINTE STRICTE : Chaque mot-clé généré ne doit absolument PAS dépasser 3 mots au total (ex: "mot1 mot2 mot3").
        
        Réponds STRICTEMENT au format JSON (liste d'objets) suivant, sans markdown autour :
        [
            {{
                "mot_cle": "Le mot clé puissant proposé",
                "raison": "Pourquoi c'est pertinent et pourquoi ça va marcher",
                "potentiel": "Élevé ou Très Élevé"
            }}
        ]
        """
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            model = "gemini-2.5-flash"
            response = client.models.generate_content(model=model, contents=prompt)
            json_str = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception as e:
            return []

# --- 2d. MOTEUR RECOMMANDATION DYNAMIQUE ---
class ContentRecommender:
    def __init__(self, df_pages):
        self.df_pages = df_pages
        self.GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") 

    def get_content_suggestions_static(self):
        return [{"segment": "", "context": "", "missing_content": "", "reasoning": "", "priority": ""}]

    def generate_gemini_suggestions(self):
        if not GEMINI_AVAILABLE:
            return self.get_content_suggestions_static()

        top_titles = self.df_pages.head(15)['Titre'].tolist()
        titles_str = "\n".join([f"- {t}" for t in top_titles])
        prompt = f"""Tu es un expert en stratégie de contenu web.
        Titres : {titles_str}
        Propose 10 IDÉES. Réponds UNIQUEMENT au format JSON :
        [{{ "segment": "", "context": "", "missing_content": "", "reasoning": "", "priority": "Haute" }}]"""
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            model = "gemini-2.5-flash"
            response = client.models.generate_content(model=model, contents=prompt)
            json_str = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception as e:
            return [{"segment": "Erreur", "context": "", "missing_content": f"{str(e)}", "reasoning": "", "priority": ""}]

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
        labels = {}; descriptions = {}; actions = {}; colors = {}
        avg_views = self.df['Vues'].mean()
        avg_time = self.df['Temps_Moyen'].mean()
        for c_id in cluster_summary.index:
            stats = cluster_summary.loc[c_id]
            if stats['Vues'] > avg_views and stats['Temps_Moyen'] > avg_time:
                labels[c_id] = "🌟 Contenu Star"; descriptions[c_id] = "Fort trafic, forte lecture."; actions[c_id] = "A maintenir en page d'accueil."; colors[c_id] = "#2ecc71"
            elif stats['Vues'] > avg_views and stats['Temps_Moyen'] < avg_time:
                labels[c_id] = "📉 Trafic sans Engagement"; descriptions[c_id] = "Beaucoup de clics, peu de lecture."; actions[c_id] = "Optimiser le contenu."; colors[c_id] = "#e67e22"
            elif stats['Vues'] < avg_views and stats['Temps_Moyen'] > avg_time:
                labels[c_id] = "💎 Pépites Cachées"; descriptions[c_id] = "Peu vu, mais très apprécié."; actions[c_id] = "A diffuser sur les réseaux."; colors[c_id] = "#3498db"
            else:
                labels[c_id] = "💤 Contenu Dormant"; descriptions[c_id] = "Faible performance globale."; actions[c_id] = "A archiver."; colors[c_id] = "#e74c3c"
        self.df['Label'] = self.df['Cluster'].map(labels)
        self.df['Description_IA'] = self.df['Cluster'].map(descriptions)
        self.df['Action_IA'] = self.df['Cluster'].map(actions)
        self.df['Color'] = self.df['Cluster'].map(colors)
        return self.df, labels

# --- 2f. USER JOURNEY ---
class UserJourneyAI:
    def __init__(self, df_events):
        self.df_events = df_events
        self.interest_type = "Début Formulaire"
        self.conversion_type = "Formulaire Validé"

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
        
        form_starts = self.get_count(['form_start'])
        if form_starts == 0:
            form_starts = self.get_count(['click'])
            self.interest_type = "Clics d'intérêt"
        else:
            self.interest_type = "Début Formulaire"
            
        conversions = self.get_count(['form_submit'])
        if conversions == 0:
             conversions = self.get_count(['video_progress', 'video_complete'])
             self.conversion_type = "Engagement Vidéo"
        else:
             self.conversion_type = "Formulaire Validé"
             
        downloads = self.get_count(['file_download'])
        
        return {
            "sessions": sessions,
            "scrolls": scrolls,
            "form_starts": form_starts,
            "conversions": conversions,
            "downloads": downloads
        }

    def analyze_journey(self):
        stats = self.get_journey_stats() 
        sessions = stats['sessions']
        scrolls = stats['scrolls']
        form_starts = stats['form_starts']
        conversions = stats['conversions']

        insights = []

        if sessions > 0:
            if scrolls > sessions:
                diagnosis_text = f"Engagement Fort : les utilisateurs naviguent sur plusieurs pages."
                drop_rate_text = "✅ Gain"
                why_text = "Les visiteurs sont captifs et naviguent sur plusieurs pages (Multi-page journey)."
                action_text = "Insérer des liens croisés (Cross-linking) pour maintenir cette dynamique."
            else:
                drop_engagement = 100 - (scrolls / sessions * 100)
                diagnosis_text = f"{int(drop_engagement)}% des visiteurs repartent sans lire."
                drop_rate_text = f"{int(drop_engagement)}%" 
                why_text = "Le haut de page (Titre/Image) n'accroche pas assez ou le chargement est lent."
                action_text = "Améliorer l'accroche et le temps de chargement."
        else:
             diagnosis_text = "Pas de données."
             drop_rate_text = "-"
             why_text = ""
             action_text = ""

        insights.append({
            "step": "1️⃣ Arrivée -> Lecture",
            "moment": "Dans les premières secondes",
            "diagnosis": diagnosis_text,
            "drop_rate": drop_rate_text, 
            "why": why_text,
            "action": action_text
        })

        drop_interest = 100 - (form_starts / scrolls * 100) if scrolls > 0 else 0
        insights.append({
            "step": f"2️⃣ Lecture -> {self.interest_type}",
            "moment": "Après consommation du contenu",
            "diagnosis": f"{int(drop_interest)}% des lecteurs n'entament aucune démarche active.",
            "drop_rate": f"{int(drop_interest)}%",
            "why": "Le contenu est lu mais ne déclenche pas d'interaction.",
            "action": "Ajouter des Call-to-Action (Boutons) plus visibles."
        })

        if form_starts > 0:
            if conversions >= form_starts:
                diagnosis_text = f"Performance exceptionnelle : {conversions} validations pour {form_starts} débuts."
                drop_rate_text = "+Gain"
                why_text = "Le formulaire est extrêmement simple, ou des utilisateurs valident sans que le début soit tracé."
                action_text = "Capitalisez sur ce modèle de formulaire."
            else:
                drop_friction = 100 - (conversions / form_starts * 100)
                diagnosis_text = f"{int(drop_friction)}% des formulaires commencés sont abandonnés."
                drop_rate_text = f"{int(drop_friction)}%"
                why_text = "Barrière à l'entrée (Formulaire trop long, pièce jointe manquante)."
                action_text = "Simplifiez le formulaire et ajoutez des indications d'aide."
        else:
            diagnosis_text = "Aucune démarche commencée."
            drop_rate_text = "N/A"
            why_text = "Pas de données."
            action_text = "Vérifier le tracking."

        insights.append({
            "step": f"3️⃣ {self.interest_type} -> {self.conversion_type}",
            "moment": "Tentative de validation",
            "diagnosis": diagnosis_text,
            "drop_rate": drop_rate_text,
            "why": why_text,
            "action": action_text
        })
        
        return insights


# --- 4. NOUVEAU MOTEUR : CHAT AVEC LES DONNÉES ---
class DataChatBot:
    def __init__(self, df_ts, df_events, df_pages):
        self.df_ts = df_ts
        self.df_events = df_events
        self.df_pages = df_pages
        self.GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

    def get_data_context(self):
        context = "Voici le contexte analytique actuel du site web (Google Analytics):\n\n"
        if not self.df_ts.empty and 'Date' in self.df_ts.columns and 'Utilisateurs actifs' in self.df_ts.columns:
            best_day = self.df_ts.loc[self.df_ts['Utilisateurs actifs'].idxmax()]
            context += f"--- AUDIENCE GLOBALE ---\n"
            context += f"Période analysée : du {self.df_ts['Date'].min().strftime('%Y-%m-%d')} au {self.df_ts['Date'].max().strftime('%Y-%m-%d')}\n"
            context += f"Total des utilisateurs actifs : {self.df_ts['Utilisateurs actifs'].sum()}\n"
            context += f"Le meilleur moment (pic) a été le {best_day['Date'].strftime('%Y-%m-%d')} avec {int(best_day['Utilisateurs actifs'])} utilisateurs actifs.\n\n"
        
        if not self.df_events.empty:
            context += "--- COMPORTEMENT ET ÉVÉNEMENTS ---\n"
            for _, row in self.df_events.iterrows():
                context += f"- {row['Nom événement']} : {row['Total']} occurrences\n"
            context += "\n"
            
        if not self.df_pages.empty:
            context += "--- DONNÉES DES PAGES ANALYSÉES ---\n"
            # On retire le .head(15) pour envoyer toutes les pages extraites à l'IA
            for _, row in self.df_pages.iterrows():
                context += f"- Titre: {row['Titre']} | Vues: {row['Vues']} | Temps moyen: {int(row['Temps_Moyen'])}s | Taux de rebond: {row['Taux_Rebond']:.1f}%\n"
        
        return context

    def ask_question(self, question, history_str=""):
        if not GEMINI_AVAILABLE or not self.GEMINI_API_KEY:
            return "⚠️ L'API Gemini n'est pas configurée ou la librairie `google-genai` est manquante. Veuillez vérifier votre environnement."
        
        data_context = self.get_data_context()
        
        prompt = f"""Tu es un Data Analyst expert et francophone qui aide un décideur à comprendre les performances de son site web.
        {data_context}
        
        Historique de la conversation récente :
        {history_str}
        
        Question du décideur : {question}
        
        Règles de réponse :
        1. Réponds de manière claire, concise et professionnelle.
        2. Base-toi EXCLUSIVEMENT sur les données fournies ci-dessus.
        3. Si la réponse n'est pas contenue dans les données (par exemple s'il demande la météo ou une donnée de l'année dernière non présente), dis-le honnêtement sans inventer de chiffres.
        4. N'hésite pas à donner un mini-conseil d'optimisation basé sur ton constat.
        """
        
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Erreur lors de la communication avec l'IA : {str(e)}"
