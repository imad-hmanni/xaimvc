import streamlit as st
import io
import os
from datetime import datetime, timedelta
import pandas as pd

# Import des modules MVC
import models
import views

# Configuration de la page
st.set_page_config(page_title="XAI - Dashboard Standard", page_icon="🇲🇦", layout="wide")

def main():
    # 1. Initialisation de la Vue
    views.inject_custom_css()
    st.title("🤖 Dashboard Stratégique (XAI)")
    st.markdown("Combinaison : **Google Analytics** + **Apprentissage Automatique** + **Raisonnement** + **Explicabilité**")
    
    # 2. Gestion des entrées (Sidebar) avec Paramètres
    site_name, file_ts_upload, file_pages_upload, prediction_days = views.render_sidebar()
    
    # Initialisation des dataframes vides pour éviter les crashs
    df_ts_final = pd.DataFrame()
    df_events_final = pd.DataFrame()
    df_pages_final = pd.DataFrame()
    auto_start_date_final = None
    time_step = 1 
    is_fallback = True 

    has_instantane_data = False
    has_pages_data = False

    try:
        # A. Traitement du Fichier Instantané
        if file_ts_upload:
            df_ts_1, df_events_1, df_pages_1, date_1, step_1, fb_1 = models.load_and_parse_data(file_ts_upload)
            df_ts_final = df_ts_1
            df_events_final = df_events_1
            auto_start_date_final = date_1
            time_step = step_1 
            
            if not file_pages_upload:
                df_pages_final = df_pages_1
                is_fallback = fb_1
            
            has_instantane_data = True
            st.sidebar.success("✅ Rapport 'Instantané' chargé.")

        # B. Traitement du Fichier Pages
        if file_pages_upload:
            df_ts_2, df_events_2, df_pages_2, date_2, step_2, fb_2 = models.load_and_parse_data(file_pages_upload)
            df_pages_final = df_pages_2
            is_fallback = False 
            
            if df_ts_final.empty:
                df_ts_final = df_ts_2 
                df_events_final = df_events_2 
                auto_start_date_final = date_2
                time_step = step_2
            
            has_pages_data = True
            st.sidebar.success("✅ Rapport 'Pages et écrans' chargé.")

        # C. Fallback local (Démo)
        if not file_ts_upload and not file_pages_upload:
            local_path = 'Instantané_des_rapports.csv'
            if os.path.exists(local_path):
                with open(local_path, 'rb') as f:
                    file_bytes = io.BytesIO(f.read())
                    df_ts_final, df_events_final, df_pages_final, auto_start_date_final, time_step, is_fallback = models.load_and_parse_data(file_bytes)
                st.sidebar.info(f"📂 Mode Démo : Fichier local utilisé")
                has_instantane_data = True 
            else:
                st.info("👋 Veuillez uploader vos fichiers CSV pour commencer (Barre latérale).")
                return

        # 4. Vérification post-chargement
        if df_ts_final.empty and df_pages_final.empty:
            st.error("⚠️ Aucun fichier valide détecté ou impossible de lire le format.")
            return

        # --- Préparation des Dates Intelligente (Jour/Semaine/Mois) ---
        start_date_user = auto_start_date_final if auto_start_date_final else datetime(2025, 1, 1)
        
        if 'time_step' not in locals() or type(time_step) is not int or time_step < 0:
            time_step = 1
            
        delta = timedelta(days=time_step if time_step > 0 else 1)
        
        # Sécurisation stricte de la création des dates
        if not df_ts_final.empty:
            if time_step > 0 and 'Index_Temporel' in df_ts_final.columns:
                 df_ts_final['Date'] = [datetime.combine(start_date_user, datetime.min.time()) + (delta * int(x)) for x in df_ts_final['Index_Temporel']]
            elif 'Date_Reelle' in df_ts_final.columns:
                 df_ts_final['Date'] = df_ts_final['Date_Reelle']
            else:
                 df_ts_final['Date'] = [start_date_user + (delta * i) for i in range(len(df_ts_final))]
            df_ts_final = df_ts_final.sort_values('Date')

        # Préparation Training set
        df_for_training = df_ts_final.copy()
        if not df_ts_final.empty and 'Date' in df_ts_final.columns:
            last_date = df_ts_final['Date'].max()
            # MODIFICATION ICI : On utilise prediction_days au lieu de 30
            start_training = last_date - timedelta(days=prediction_days)
            df_for_training = df_ts_final[df_ts_final['Date'] >= start_training]
            if len(df_for_training) < 5:
                df_for_training = df_ts_final.copy()

        # 5. Instanciation des Modèles IA
        xai_engine = models.XAIEngine(df_for_training)
        nlp_engine = models.SemanticAnalyzer(df_pages_final)
        recommender = models.ContentRecommender(df_pages_final)
        content_optimizer = models.ContentOptimizer(df_pages_final)
        journey_ai = models.UserJourneyAI(df_events_final)
        
        # Assistant IA Chatbot
        chat_engine = models.DataChatBot(df_ts_final, df_events_final, df_pages_final)

        # --- CONSTRUCTION DYNAMIQUE DES ONGLETS ---
        tabs_config = []

        if has_instantane_data:
            tabs_config.extend([
                ("📊 Analyse & KPI", lambda: views.render_kpi_tab(site_name, df_ts_final, df_events_final)),
                ("🔮 Prédictions & XAI", lambda: render_predictions_wrapper(xai_engine, df_ts_final, df_for_training, delta, prediction_days)),
                ("🧠 Analyse Sémantique (NLP)", lambda: views.render_nlp_tab(nlp_engine, is_fallback)),
                ("📍 Parcours & Churn", lambda: views.render_journey_tab(journey_ai))
            ])

        if has_pages_data:
            tabs_config.extend([
                ("⚙️ Audit Contenu", lambda: render_audit_wrapper(content_optimizer, is_fallback)),
                ("🎨 Personnalisation (IA)", lambda: render_personalization_wrapper(recommender, is_fallback))
            ])

        if not tabs_config:
            st.warning("Aucune donnée exploitable trouvée pour afficher les onglets.")
        else:
            tab_labels = [t[0] for t in tabs_config]
            tabs_objects = st.tabs(tab_labels)
            for i, tab_obj in enumerate(tabs_objects):
                with tab_obj:
                    tabs_config[i][1]()
                    
        # --- LE CHATBOT EST APPELÉ ICI POUR ÊTRE TOUJOURS FLOTTANT ---
        if has_instantane_data or has_pages_data:
            views.render_floating_chat(chat_engine)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement : {e}")

# --- Wrappers (Lazy Loading) ---

def render_predictions_wrapper(engine, df_ts, df_train, delta, days_to_predict):
    engine.train_model()
    df_future = engine.predict_future(days=days_to_predict, step_delta=delta)
    df_trend = engine.get_historical_trend()
    xai_data = engine.explain_prediction()
    views.render_prediction_tab(df_ts, df_train, df_future, xai_data, df_trend=df_trend)

def render_audit_wrapper(optimizer, is_fallback):
    df_opt, labels = optimizer.analyze_content_performance()
    views.render_audit_tab(df_opt, labels, is_fallback)

def render_personalization_wrapper(recommender, is_fallback):
    if 'gemini_suggestions' not in st.session_state:
        st.session_state.gemini_suggestions = None
        
    views.render_personalization_tab(recommender, is_fallback)

if __name__ == "__main__":
    main()
