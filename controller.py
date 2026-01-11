import streamlit as st
import io
import os
from datetime import datetime, timedelta

# Import des modules MVC
import models
import views

# Configuration de la page (Doit être la première commande Streamlit)
st.set_page_config(page_title="XAI - Dashboard Standard", page_icon="🇲🇦", layout="wide")

def main():
    # 1. Initialisation de la Vue
    views.inject_custom_css()
    st.title("🤖 Dashboard Stratégique (XAI)")
    st.markdown("Combinaison : **Google Analytics** + **Apprentissage Automatique** + **Raisonnement** + **Explicabilité**")
    
    # 2. Gestion des entrées (Sidebar) via la Vue
    site_name, uploaded_file = views.render_sidebar()
    
    # 3. Logique de chargement des données (Contrôleur -> Modèle)
    file_to_process = None
    source_type = ""
    
    if uploaded_file is not None:
        file_to_process = uploaded_file
        source_type = "upload"
    else:
        # Fallback fichier local
        local_path = 'Instantané_des_rapports.csv'
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                file_bytes = io.BytesIO(f.read())
                file_to_process = file_bytes
                source_type = "local"
        else:
            st.info("👋 Veuillez uploader un fichier CSV Google Analytics pour commencer.")
            return

    # 4. Traitement des données (Modèle)
    if file_to_process:
        try:
            df_ts_raw, df_events, df_pages, auto_start_date, is_indexed, is_fallback = models.load_and_parse_data(file_to_process)
            
            if df_ts_raw.empty:
                st.error("⚠️ Le fichier CSV semble vide ou illisible.")
                return

            if source_type == "upload":
                st.sidebar.success(f"✅ Fichier uploadé analysé !")
            else:
                st.sidebar.info(f"📂 Mode Démo : Utilisation du fichier local '{local_path}'")
            
            st.sidebar.caption(f"{len(df_ts_raw)} jours de données détectés.")
            if is_fallback:
                st.sidebar.warning("⚠️ Titres de pages non détectés. Mode dégradé activé.")

            # Préparation des dates
            start_date_user = auto_start_date if auto_start_date else datetime(2025, 1, 1)
            delta = timedelta(days=1)
            
            if is_indexed:
                df_ts_raw['Date'] = [datetime.combine(start_date_user, datetime.min.time()) + (delta * int(x)) for x in df_ts_raw['Index_Temporel']]
            else:
                df_ts_raw['Date'] = df_ts_raw['Date_Reelle']

            df_ts = df_ts_raw.sort_values('Date')
            
            # Préparation Training set
            df_for_training = df_ts.copy()
            if not df_ts.empty:
                last_date = df_ts['Date'].max()
                start_training = last_date - timedelta(days=30)
                df_for_training = df_ts[df_ts['Date'] >= start_training]
                if len(df_for_training) < 5:
                    df_for_training = df_ts.copy()

            # 5. Instanciation des Modèles IA
            xai_engine = models.XAIEngine(df_for_training)
            nlp_engine = models.SemanticAnalyzer(df_pages)
            recommender = models.ContentRecommender(df_pages)
            content_optimizer = models.ContentOptimizer(df_pages)
            journey_ai = models.UserJourneyAI(df_events)

            # --- Affichage des Onglets (Orchestration Vue) ---
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Analyse & KPI", "🔮 Prédictions & XAI", "🧠 Recommandations Stratégiques", 
                "🧠 Analyse Sémantique (NLP)", "🎨 Personnalisation (IA)", "⚙️ Audit Contenu", "📍 Parcours & Churn"
            ])
            
            # Tab 1: KPI
            with tab1:
                views.render_kpi_tab(site_name, df_ts, df_events)
            
            # Tab 2: Prédictions
            with tab2:
                xai_engine.train_model()
                df_future = xai_engine.predict_future(days=30, step_delta=delta)
                xai_data = xai_engine.explain_prediction()
                views.render_prediction_tab(df_ts, df_for_training, df_future, xai_data)
                
            # Tab 3: Stratégie
            with tab3:
                # Calcul XAI nécessaire ici s'il n'a pas été fait en Tab 2 (Streamlit exécute séquentiellement)
                if xai_engine.trend is None:
                    xai_engine.train_model()
                    xai_data = xai_engine.explain_prediction()
                recommendations = models.generate_recommendations(df_events, xai_data, df_pages)
                views.render_strategy_tab(recommendations)
                
            # Tab 4: NLP
            with tab4:
                views.render_nlp_tab(nlp_engine, is_fallback)
                
            # Tab 5: Personnalisation (Avec callback pour Gemini)
            with tab5:
                # Gestion de l'état du bouton dans le contrôleur (Streamlit Session State)
                if 'gemini_suggestions' not in st.session_state:
                    st.session_state.gemini_suggestions = None
                
                def gemini_callback():
                    with st.spinner("Analyse des pages et génération des idées..."):
                        st.session_state.gemini_suggestions = recommender.generate_gemini_suggestions()
                
                # Si pas de suggestions en session, on prend les statiques du modèle
                current_suggestions = st.session_state.gemini_suggestions if st.session_state.gemini_suggestions else recommender.get_content_suggestions_static()
                
                views.render_personalization_tab(current_suggestions, is_fallback, gemini_callback)
                
            # Tab 6: Audit
            with tab6:
                df_optimized, cluster_labels = content_optimizer.analyze_content_performance()
                views.render_audit_tab(df_optimized, cluster_labels, is_fallback)
                
            # Tab 7: Journey
            with tab7:
                views.render_journey_tab(journey_ai)

        except Exception as e:
            st.error(f"Une erreur est survenue lors de l'analyse du fichier : {e}")
            st.error("Veuillez vérifier le format du fichier ou la présence du fichier local.")
