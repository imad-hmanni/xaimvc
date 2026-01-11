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
    </style>
    """, unsafe_allow_html=True)

# --- Composants d'Affichage ---

def render_sidebar():
    st.sidebar.header("📂 Configuration des Données")
    site_name = st.sidebar.text_input("Nom du Site / Portail", value="Site Web")
    uploaded_file = st.sidebar.file_uploader("Fichier CSV Google Analytics", type=['csv'])
    return site_name, uploaded_file

def render_kpi_tab(site_name, df_ts, df_events):
    st.subheader(f"Vue d'ensemble : {site_name}")
    col1, col2, col3, col4 = st.columns(4)
    total_users = df_ts['Utilisateurs actifs'].sum()
    avg_users = df_ts['Utilisateurs actifs'].mean()
    total_views = df_events[df_events['Nom événement'] == 'page_view']['Total'].sum() if not df_events.empty else 0
    
    col1.metric("Total Utilisateurs", f"{total_users:,.0f}")
    col2.metric("Moyenne / Période", f"{avg_users:,.0f}")
    col3.metric("Pages Vues", f"{total_views:,.0f}")
    col4.metric("Points de données", len(df_ts))
    
    fig = px.line(df_ts, x='Date', y='Utilisateurs actifs', title='Évolution Historique', markers=True, line_shape='spline')
    fig.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Période affichée : du {df_ts['Date'].min().strftime('%d/%m/%Y')} au {df_ts['Date'].max().strftime('%d/%m/%Y')}")

def render_prediction_tab(df_ts, df_for_training, df_future, xai_data, use_recent_only=True):
    st.subheader(f"Prédiction & Explicabilité")
    col_pred, col_xai = st.columns([2, 1])
    
    with col_pred:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Utilisateurs actifs'], mode='lines', name='Historique Complet', line=dict(color='#2980b9', width=1)))
        if use_recent_only:
            fig_pred.add_trace(go.Scatter(x=df_for_training['Date'], y=df_for_training['Utilisateurs actifs'], mode='lines', name='Zone Apprentissage (30j)', line=dict(color='#2ecc71', width=3)))
        if not df_future.empty:
            fig_pred.add_trace(go.Scatter(x=df_future['Date'], y=df_future['Prédiction'], mode='lines+markers', name='Prédiction IA', line=dict(color='#e74c3c', dash='dot', width=3)))
        
        fig_pred.update_layout(title="Trajectoire prédite (Mois Prochain)", hovermode="x unified")
        st.plotly_chart(fig_pred, use_container_width=True)
        
    with col_xai:
        st.markdown("### 🔍 Pourquoi cette prédiction ?")
        st.markdown(textwrap.dedent(f"""
            <div class="xai-explanation">
                <strong>Tendance Globale :</strong><br>{xai_data['tendance']}<br><br>
                <strong>Analyse du Modèle :</strong><br>{xai_data['detail_tendance']}<br><br>
                <strong>Caractéristique des données :</strong><br>{xai_data['facteur_cle']}
            </div>
        """), unsafe_allow_html=True)

def render_strategy_tab(recommendations):
    st.subheader("Aide à la Décision Stratégique")
    if not recommendations:
        st.warning("Pas assez de données pour générer des recommandations.")
    
    for rec in recommendations:
        type_to_emoji = {'success': '✅', 'warning': '⚠️', 'danger': '🚫', 'info': 'ℹ️', 'critical': '📉'}
        emoji = type_to_emoji.get(rec['type'], 'ℹ️')
        
        with st.container(border=True):
            st.subheader(f"{emoji} {rec['titre']}")
            st.write(f"**Analyse :** {rec['logique']}")
            
            if rec['type'] == 'success':
                st.success(f"**Action :** {rec['action']}")
            elif rec['type'] == 'warning':
                st.warning(f"**Action :** {rec['action']}")
            elif rec['type'] == 'danger' or rec['type'] == 'critical':
                st.error(f"**Action :** {rec['action']}")
            else:
                st.info(f"**Action :** {rec['action']}")

def render_nlp_tab(nlp_engine, is_fallback):
    st.subheader("🧠 Analyse Sémantique")
    if is_fallback:
        st.warning("⚠️ Les titres de pages réels n'ont pas été détectés. Analyse basée sur des données génériques.")
    
    col_keywords, col_topics = st.columns(2)
    
    with col_keywords:
        st.markdown("#### 🔑 Mots-clés les plus performants")
        df_kw = nlp_engine.extract_top_keywords()
        if not df_kw.empty:
            fig_kw = px.bar(df_kw, x='Fréquence', y='Mot-clé', orientation='h', color='Fréquence', color_continuous_scale='Viridis')
            fig_kw.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Pas assez de données textuelles.")
            
    with col_topics:
        st.markdown("#### 📚 Thématiques Identifiées")
        topics = nlp_engine.identify_topics()
        for t in topics:
            parts = t.split(':')
            title = parts[0]
            content = parts[1] if len(parts) > 1 else ""
            st.markdown(f"""
                <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #3498db;">
                    <strong>{title}</strong><br><em>{content}</em>
                </div>
            """, unsafe_allow_html=True)

def render_personalization_tab(suggestions, is_fallback, gemini_callback):
    st.subheader("🎨 Personnalisation Dynamique : Suggestions de Contenu (IA)")
    st.markdown("L'IA analyse vos titres de pages réels et utilise **(IA DeepMind)** pour inventer des opportunités.")
    
    if is_fallback:
        st.warning("⚠️ Analyse basée sur des données génériques.")
        
    if st.button("✨ Lancer la Suggestion (IA)"):
        gemini_callback()

    if not suggestions: # Cas initial ou erreur
         st.caption("Cliquez sur le bouton pour utiliser Gemini.")
    
    # Si suggestions est une liste vide (car pas encore chargé), on n'affiche rien de plus
    # Si c'est rempli, on boucle
    if suggestions:
        for s in suggestions:
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
                        st.dataframe(group_data[['Titre', 'Vues', 'Temps_Moyen']].sort_values('Vues', ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Pas assez de données pour segmenter le contenu.")

def render_journey_tab(journey_ai):
    st.subheader("📍 Analyse des Points de Rupture (User Journey)")
    st.markdown("#### 📉 Entonnoir de Conversion Simplifié")
    stats = journey_ai.get_journey_stats()
    
    if stats['sessions'] > 0:
        col_s1, col_a1, col_s2, col_a2, col_s3, col_a3, col_s4 = st.columns([2,0.5,2,0.5,2,0.5,2])
        with col_s1: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #3498db;'><h4 style='margin:0; color:#3498db;'>1. Arrivée</h3><h2 style='margin:10px 0;'>{int(stats['sessions'])}</h4></div>", unsafe_allow_html=True)
        with col_a1: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
        with col_s2: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #f1c40f;'><h4 style='margin:0; color:#f1c40f;'>2. Lecture</h3><h2 style='margin:10px 0;'>{int(stats['scrolls'])}</h4></div>", unsafe_allow_html=True)
        with col_a2: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
        with col_s3: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #e67e22;'><h4 style='margin:0; color:#e67e22;'>3. {journey_ai.interest_type}</h4><h2 style='margin:10px 0;'>{stats['form_starts']}</h2></div>", unsafe_allow_html=True)
        with col_a3: st.markdown("<div class='step-arrow'>➔</div>", unsafe_allow_html=True)
        with col_s4: st.markdown(f"<div class='step-card' style='border-bottom: 4px solid #2ecc71;'><h4 style='margin:0; color:#2ecc71;'>4. {journey_ai.conversion_type}</h4><h2 style='margin:10px 0;'>{stats['conversions']}</h2></div>", unsafe_allow_html=True)

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
        st.warning("Données insuffisantes pour tracer le parcours utilisateur.")