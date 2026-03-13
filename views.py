import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import textwrap

# --- Configuration CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #c0392b;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .xai-explanation {
            background-color: #e8f8f5;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #1abc9c;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stProgress > div > div > div > div {
            background-color: #e74c3c;
        }
        .step-card {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-bottom: 4px solid #ddd;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .step-arrow {
            text-align: center;
            font-size: 24px;
            color: #7f8c8d;
            margin-top: 30px;
            font-weight: bold;
        }
        .download-card {
            background-color: #e1f5fe;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 5px solid #0288d1;
            margin-top: 20px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Composants d'Affichage ---

def render_sidebar():
    st.sidebar.header("📂 Configuration des Données")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("1. Données Temporelles")
    st.sidebar.caption("Fichier 'Instantané' (Séries, User Journey)")
    file_ts = st.sidebar.file_uploader("Upload Instantané.csv", type=['csv'], key="file_ts")
    
    st.sidebar.subheader("2. Données Contenu")
    st.sidebar.caption("Fichier 'Pages et écrans' (Temps, Engagement)")
    file_pages = st.sidebar.file_uploader("Upload Pages.csv", type=['csv'], key="file_pages")
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("3. Paramètres IA")
    prediction_horizon = st.sidebar.slider("Horizon de prédiction (Jours)", min_value=7, max_value=365, value=90, step=1)
    
    st.sidebar.markdown("---")
    site_name = st.sidebar.text_input("Nom du Site / Portail", value="Site Web")
    
    return site_name, file_ts, file_pages, prediction_horizon

def render_kpi_tab(site_name, df_ts, df_events):
    st.subheader(f"Vue d'ensemble : {site_name}")
    col1, col2, col3, col4 = st.columns(4)
    
    total_users = df_ts['Utilisateurs actifs'].sum() if not df_ts.empty and 'Utilisateurs actifs' in df_ts.columns else 0
    avg_users = df_ts['Utilisateurs actifs'].mean() if not df_ts.empty and 'Utilisateurs actifs' in df_ts.columns else 0
    total_views = df_events[df_events['Nom événement'] == 'page_view']['Total'].sum() if not df_events.empty else 0
    data_points = len(df_ts) if not df_ts.empty else 0
    
    col1.metric("Total Utilisateurs", f"{total_users:,.0f}")
    col2.metric("Moyenne / Période", f"{avg_users:,.0f}")
    col3.metric("Pages Vues", f"{total_views:,.0f}")
    col4.metric("Données analysées", data_points)
    
    if not df_ts.empty and 'Date' in df_ts.columns and 'Utilisateurs actifs' in df_ts.columns:
        fig = px.line(df_ts, x='Date', y='Utilisateurs actifs', title='Évolution Historique', markers=True, line_shape='spline')
        fig.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
        if df_ts['Date'].notna().any():
            st.caption(f"Période affichée : du {df_ts['Date'].min().strftime('%d/%m/%Y')} au {df_ts['Date'].max().strftime('%d/%m/%Y')}")
    else:
        st.info("Graphique temporel indisponible (Données temporelles manquantes ou format non reconnu).")

def render_prediction_tab(df_ts, df_for_training, df_future, xai_data, df_trend=None):
    st.subheader(f"Prédiction & Explicabilité")
    
    # Conversion des prédictions en entiers pour un affichage propre (sans virgule)
    if not df_future.empty:
        df_future['Prédiction'] = df_future['Prédiction'].round().astype(int)
    if df_trend is not None and not df_trend.empty:
        df_trend['Tendance_IA'] = df_trend['Tendance_IA'].round().astype(int)
        
    col_pred, col_xai = st.columns([2, 1])
    
    with col_pred:
        if not df_ts.empty and 'Date' in df_ts.columns and 'Utilisateurs actifs' in df_ts.columns:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Utilisateurs actifs'], mode='lines', name='Historique Réel', line=dict(color='#2980b9', width=1)))
            if df_trend is not None and not df_trend.empty:
                fig_pred.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Tendance_IA'], mode='lines', name='Tendance IA (Modèle)', line=dict(color='#f39c12', dash='solid', width=2)))
            if not df_future.empty:
                fig_pred.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Prédiction'], mode='lines+markers', name='Prédiction Futur', line=dict(color='#e74c3c', dash='dot', width=3)))
            
            fig_pred.update_layout(title="Trajectoire IA : Passé (Tendance) et Futur (Prévision)", hovermode="x unified")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # --- NOUVEAU BLOC : Affichage des métriques sous le graphique ---
            if not df_future.empty:
                total_pred = int(df_future['Prédiction'].sum())
                avg_pred = int(df_future['Prédiction'].mean())
                
                st.markdown("#### 📊 Résumé de la Prévision")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Total Utilisateurs (Prévus)", value=f"{total_pred:,.0f}")
                with metric_col2:
                    st.metric(label="Moyenne par Période (Prévue)", value=f"{avg_pred:,.0f}")
                    
        else:
            st.warning("Pas assez de données historiques pour afficher une prédiction fiable.")
        
    with col_xai:
        st.markdown("### 🔍 Pourquoi cette prédiction ?")
        with st.container(border=True):
            st.markdown(f"**1. Tendance Globale :**\n\n{xai_data['tendance']}")
            st.caption("*(La ligne orange montre la direction générale calculée par l'IA, sans le bruit quotidien.)*")
            
            st.markdown("---")
            
            st.markdown(f"**2. Analyse du Modèle :**\n\n{xai_data['detail_tendance']}")
            
            st.markdown("---")
            
            st.markdown(f"**3. Caractéristique des données :**\n\n{xai_data['facteur_cle']}")

def render_nlp_tab(nlp_engine, is_fallback):
    st.subheader("🧠 Analyse Sémantique")
    if is_fallback:
        st.warning("⚠️ Les titres de pages réels n'ont pas été détectés. Analyse basée sur des données génériques.")
    
    col_keywords, col_topics = st.columns(2)
    
    df_kw = nlp_engine.extract_top_keywords()
    topics = nlp_engine.identify_topics()
    
    with col_keywords:
        st.markdown("#### 🔑 Mots-clés les plus performants")
        if not df_kw.empty:
            fig_kw = px.bar(df_kw, x='Fréquence', y='Mot-clé', orientation='h', color='Fréquence', color_continuous_scale='Viridis')
            fig_kw.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Pas assez de données textuelles.")
            
    with col_topics:
        st.markdown("#### 📚 Thématiques Identifiées")
        
        if st.button("✨ Expliquer le sens caché (IA)", key="explain_topics_btn"):
            with st.spinner("L'IA analyse le sens des thématiques..."):
                st.session_state.topic_explanations = nlp_engine.explain_topics(topics)

        explanations = st.session_state.get('topic_explanations', {})

        for t in topics:
            parts = t.split(' : ')
            title = parts[0].strip()
            content = parts[1] if len(parts) > 1 else ""
            
            explanation_html = ""
            if explanations and title in explanations:
                exp_text = explanations[title]
                explanation_html = f"""
                <div style='margin-top: 12px; padding: 10px; background-color: #fdf2e9; border-left: 3px solid #e67e22; border-radius: 4px; font-size: 0.95em; color: #333;'>
                    <strong>🤖 Explication IA :</strong><br>{exp_text}
                </div>
                """
            
            st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #3498db; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <h5 style="margin-top: 0; margin-bottom: 5px; color: #2c3e50;">{title}</h5>
                    <span style="color: #7f8c8d; font-size: 0.9em;"><em>Mots-clés bruts : {content}</em></span>
                    {explanation_html}
                </div>
            """, unsafe_allow_html=True)
            
    # --- BLOC : GÉNÉRATION DE MOTS CLÉS PUISSANTS ---
    st.markdown("---")
    st.markdown("### 🚀 Opportunités SEO & Mots-clés")
    st.write("Demandez à l'Intelligence Artificielle d'étudier vos mots-clés actuels et vos thématiques pour vous suggérer de nouveaux mots-clés performants à intégrer dans vos futurs contenus.")
    
    if st.button("💡 Propose-moi des mots-clés puissants pour mon site", key="generate_kw_btn"):
        with st.spinner("Génération des mots-clés SEO en cours..."):
            top_kw_list = df_kw['Mot-clé'].tolist() if not df_kw.empty else []
            st.session_state.powerful_keywords = nlp_engine.generate_powerful_keywords(top_kw_list, topics)
            
    if 'powerful_keywords' in st.session_state and st.session_state.powerful_keywords:
        st.success("🎯 Voici les recommandations stratégiques générées par Gemini :")
        
        # Affichage stylisé des mots clés
        cols = st.columns(2)
        for idx, kw in enumerate(st.session_state.powerful_keywords):
            target_col = cols[idx % 2]
            with target_col:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #2ecc71; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="font-size: 1.1em; color: #2c3e50;">🔑 {kw.get('mot_cle', '')}</strong> 
                        <span style="background-color: #e8f8f5; color: #27ae60; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold;">{kw.get('potentiel', 'Élevé')}</span>
                    </div>
                    <span style="color: #555; font-size: 0.9em; line-height: 1.4;">{kw.get('raison', '')}</span>
                </div>
                """, unsafe_allow_html=True)


def render_personalization_tab(recommender, is_fallback):
    st.subheader("🎨 Personnalisation Dynamique : Suggestions de Contenu (IA)")
    st.markdown("L'IA analyse vos titres de pages réels et utilise **(IA DeepMind)** pour inventer des opportunités.")
    
    if is_fallback:
        st.warning("⚠️ Analyse basée sur des données génériques.")
        
    if st.button("✨ Lancer la Suggestion (IA)"):
        with st.spinner("Analyse des pages et génération des idées..."):
            st.session_state.gemini_suggestions = recommender.generate_gemini_suggestions()

    suggestions = st.session_state.get('gemini_suggestions')

    if not suggestions: 
         st.caption("Cliquez sur le bouton pour utiliser Gemini.")
    else:
        for s in suggestions:
            if not s.get('missing_content'): continue 
            with st.container(border=True):
                col_seg, col_prio = st.columns([3, 1])
                with col_seg:
                    st.subheader(f"🎯 Pour : {s.get('segment', 'Segment')}")
                    st.markdown(f"**Contexte observé :** *{s.get('context', '')}*")
                with col_prio:
                    prio = s.get('priority', 'Moyenne')
                    color = "red" if prio in ["Critique", "Haute"] else "blue"
                    st.markdown(f":{color}[**Priorité : {prio}**]")
                
                st.info(f"💡 **Idée de Contenu à Créer :**\n\n### {s.get('missing_content', '')}")
                st.success(f"🧠 **Pourquoi ? (XAI) :** {s.get('reasoning', '')}")

def render_audit_tab(df_optimized, cluster_labels, is_fallback):
    st.subheader("⚙️ Audit de Contenu Automatisé (IA)")
    if is_fallback:
        st.warning("⚠️ Données simulées pour l'audit.")
        
    if cluster_labels:
        groups = df_optimized.groupby('Label')
        display_order = ["🌟 Contenu Star", "💎 Pépites Cachées", "📉 Trafic sans Engagement", "💤 Contenu Dormant"]
        col1, col2 = st.columns(2)
        
        for i, label in enumerate(display_order):
            if label in groups.groups:
                group_data = groups.get_group(label)
                first_item = group_data.iloc[0]
                count = len(group_data)
                color = first_item['Color']
                
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    st.markdown(f"""
                    <div style="border: 1px solid {color}; border-radius: 10px; padding: 20px; margin-bottom: 20px; background-color: white; border-top: 5px solid {color};">
                        <h3 style="color: {color}; margin-top:0;">{label}</h3>
                        <p style="font-size: 1.2em; font-weight: bold;">{count} pages concernées</p>
                        <p style="font-style: italic; color: #555;">"{first_item['Description_IA']}"</p>
                        <hr style="margin: 10px 0;">
                        <p><strong>👉 Action Recommandée :</strong></p>
                        <p style="background-color: {color}20; padding: 10px; border-radius: 5px; color: {color}; font-weight: bold;">{first_item['Action_IA']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander(f"Voir les pages '{label}'"):
                        st.dataframe(group_data[['Titre', 'Vues', 'Temps_Moyen', 'Taux_Rebond']].sort_values('Vues', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Pas assez de données pour segmenter le contenu.")

def render_journey_tab(journey_ai):
    st.subheader("📍 Analyse des Points de Rupture (User Journey)")
    stats = journey_ai.get_journey_stats()
    
    if stats['sessions'] > 0:
        st.markdown("#### 📉 Entonnoir de Conversion (Formulaires strictement)")
        col_s1, col_a1, col_s2, col_a2, col_s3, col_a3, col_s4 = st.columns([2,0.5,2,0.5,2,0.5,2])
        with col_s1: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #3498db;'><h4 style='margin:0; color:#3498db;'>1. Arrivée</h3><h2 style='margin:10px 0;'>{int(stats['sessions'])}</h2></div>", unsafe_allow_html=True)
        with col_a1: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
        with col_s2: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #f1c40f;'><h4 style='margin:0; color:#f1c40f;'>2. Lecture</h3><h2 style='margin:10px 0;'>{int(stats['scrolls'])}</h2></div>", unsafe_allow_html=True)
        with col_a2: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
        with col_s3: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #e67e22;'><h4 style='margin:0; color:#e67e22;'>3. {journey_ai.interest_type}</h4><h2 style='margin:10px 0;'>{stats['form_starts']}</h2></div>", unsafe_allow_html=True)
        with col_a3: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
        with col_s4: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #2ecc71;'><h4 style='margin:0; color:#2ecc71;'>4. {journey_ai.conversion_type}</h4><h2 style='margin:10px 0;'>{stats['conversions']}</h2></div>", unsafe_allow_html=True)

        if stats.get('downloads', 0) > 0:
            st.markdown(f"""
            <div class="download-card">
                <h3 style="margin:0; color: #01579b;">📥 Fichiers Téléchargés</h3>
                <p style="font-size: 1.2em; margin-top:5px; margin-bottom: 0;">En parallèle, vos utilisateurs ont généré <strong>{stats['downloads']} téléchargements</strong> de documents PDF ou autres.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("#### 🕵️‍♂️ Diagnostic des Pertes (IA)")
        insights = journey_ai.analyze_journey()
        for insight in insights:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.subheader(f"{insight['step']}")
                    st.markdown(f"**⏰ Moment :** *{insight['moment']}*")
                    st.warning(f"**⚠️ Problème :** {insight['diagnosis']}")
                with c2:
                    st.metric("Fuite (Drop-off)", insight['drop_rate'])
                st.success(f"**💡 Cause Probable (XAI) :** {insight['why']}")
                st.info(f"**🛠️ Action Recommandée :** {insight['action']}")
    else:
        st.warning("Données insuffisantes pour tracer le parcours utilisateur (Besoin du fichier Instantané/Séries).")


# --- CHAT FLOTTANT ÉPURÉ ---
def render_floating_chat(chat_engine):
    st.markdown("""
    <style>
        /* Conteneur principal figé en bas à droite */
        div[data-testid="stPopover"] {
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            z-index: 999999 !important;
            width: fit-content !important; 
            height: fit-content !important;
        }
        
        /* Bulle de chat parfaitement ronde */
        div[data-testid="stPopover"] > button {
            background-color: #e74c3c !important;
            border-radius: 50% !important;
            width: 65px !important;
            height: 65px !important;
            min-width: 65px !important;
            min-height: 65px !important;
            border: 3px solid white !important;
            box-shadow: 0 6px 16px rgba(0,0,0,0.3) !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: transform 0.3s ease, background-color 0.3s ease !important;
        }
        
        /* L'émoji dans le bouton natif Streamlit */
        div[data-testid="stPopover"] > button p {
            font-size: 32px !important;
            margin: 0 !important;
            line-height: 1 !important;
        }
        
        /* Animation au survol */
        div[data-testid="stPopover"] > button:hover {
            transform: scale(1.1) !important;
            background-color: #c0392b !important;
        }
        
        /* Fenêtre de dialogue fixe (Taille unique) */
        div[data-testid="stPopoverBody"] {
            position: fixed !important; 
            bottom: 110px !important; /* Apparait juste au-dessus du bouton */
            right: 30px !important;
            width: 360px !important;
            height: 520px !important;
            border-radius: 15px !important;
            padding: 15px !important;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation de la conversation
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "Bonjour ! Je suis l'IA Explicatif. Posez-moi vos questions sur vos données analytiques."}]

    # La bulle flottante (L'argument est l'icône native Streamlit)
    with st.popover("💬", use_container_width=False):
        
        # En-tête simple
        st.markdown("<h4 style='margin-top: 0; color: #2c3e50;'>🤖 IA Explicatif</h4>", unsafe_allow_html=True)
            
        # Zone d'affichage des messages (Taille fixe)
        messages_container = st.container(height=350)
        with messages_container:
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Formulaire d'envoi
        with st.form("chat_form", clear_on_submit=True, border=False):
            # Colonnes pour bien séparer le texte du bouton d'envoi
            col1, col2 = st.columns([5, 1]) 
            with col1:
                prompt = st.text_input("Message", label_visibility="collapsed", placeholder="Ex: Quel est le pic de trafic ?")
            with col2:
                # L'icône de la fusée pour valider l'envoi
                submitted = st.form_submit_button("🚀")
            
            if submitted and prompt:
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                
                with st.spinner("Analyse..."):
                    recent_history = [f"{m['role']}: {m['content']}" for m in st.session_state.chat_messages[-5:-1]]
                    history_str = "\n".join(recent_history)
                    response = chat_engine.ask_question(prompt, history_str)
                
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
