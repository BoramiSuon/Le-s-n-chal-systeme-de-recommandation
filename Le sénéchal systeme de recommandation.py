import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import time
from st_clickable_images import clickable_images
import streamlit_clickable_images as stci
from IPython.display import display, HTML
nltk.download('wordnet')

# configuration de la page 
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.image(r"C:\Users\borami\Downloads\Sans titre-1.png") 
st.markdown('</div>', unsafe_allow_html=True)
# Fonction qui charge notre dataframe avec cache pour eviter de charger à chaque fois 
@st.cache_data
def load_data():
    url = r"G:\Mon Drive\df_film_ML_V6_trailer+sysnopsisFR.csv"
    df = pd.read_csv(url)
    df['poster_path'] = 'https://image.tmdb.org/t/p/w500' + df['poster_path']
    df.fillna('', inplace=True)
    return df

df = load_data()

# Fonction de lemmatisation
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join(lemmatizer.lemmatize(word) for word in words)

# Fonction pour calculer la similarité avec cache
@st.cache_resource
def compute_similarity_matrix(dataframe):
    text_columns = ['genres_x', 'synopsis', 'keyword', 'top_actors', 'directors', 'producers', 'genre_id tsv.Main_Category', 'studios']
    weights = {'genres_x': 4, 'synopsis': 1, 'keyword': 1, 'top_actors': 2, 'directors': 2, 'producers': 1, 'genre_id tsv.Main_Category': 4, 'studios': 2}
    
    dataframe['combined_text'] = dataframe.apply(lambda row: " ".join((row[col] + " ") * weights[col] for col in text_columns), axis=1)
    dataframe['combined_text'] = dataframe['combined_text'].apply(lemmatize_text)
    
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,1), strip_accents='unicode', min_df=2)
    tfidf_matrix = tfidf.fit_transform(dataframe['combined_text'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity_matrix(df)

# Fonction de recommandation
@st.cache_data
def recommandation_films(title, num_recommendations=8):
    # Vérifier si le titre existe dans le DataFrame
    if title not in df['Titre'].values:
        raise ValueError(f"Le titre '{title}' n'existe pas dans les données.")

    # Récupérer l'index du film à recommander
    idx = df[df['Titre'] == title].index[0]

    # Calculer les similarités cosinus
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Trier les films similaires par score de similarité (du plus similaire au moins similaire)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Récupérer les indices des films recommandés (en excluant le film lui-même)
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]

    # Filtrer les films recommandés pour ne garder que ceux avec le statut 'Released'
    films_reco = df.iloc[movie_indices]
    films_reco = films_reco[films_reco['status'] == 'Released']

    # Ajouter les scores de similarité aux films recommandés
    films_reco['sim_score'] = [sim_scores[i + 1][1] for i in range(len(films_reco))]

    # Fonction pour générer un avis de recommandation basé sur le score de similarité
    def avis_reco(sim_score):
        if sim_score <= 0.10:
            return "Éviter à tout prix"
        elif sim_score <= 0.15:
            return "Pas convaincant"
        elif sim_score <= 0.30:
            return "Pas mal, mais à voir"
        elif sim_score <= 0.45:
            return "À regarder les yeux fermés"
        else:
            return "Incontournable"

    # Ajouter une colonne 'reco' avec l'avis de recommandation
    films_reco['reco'] = films_reco['sim_score'].apply(avis_reco)

    return films_reco
@st.cache_data
def recommandation_films_planned(title):
    # Vérifier si le titre existe dans le DataFrame
    if title not in df['Titre'].values:
        raise ValueError(f"Le titre '{title}' n'existe pas dans les données.")

    # Récupérer l'index du film à recommander
    idx = df[df['Titre'] == title].index[0]

    # Calculer les similarités cosinus
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Trier les films similaires par score de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Prendre les 5 films les plus similaires

    movie_indices = [i[0] for i in sim_scores]

    # Retourner les films recommandés
    films_reco_planned = df.iloc[movie_indices]
    films_reco_planned = films_reco_planned[~films_reco_planned['status'].isin(['Released', 'NaN'])]
    films_reco_planned = films_reco_planned[1:4]


    # Ajouter les scores de similarité aux films recommandés
    films_reco_planned['sim_score'] = [sim_scores[i+1][1] for i in range(len(films_reco_planned))]

    def avis_reco(sim_score):
      if sim_score <= 0.10:
          return "va prendre l'air plutôt"
      if sim_score <= 0.15:
          return "Pas convaincant"
      if sim_score <= 0.30:
          return "Pas mal, mais à voir"
      if sim_score <= 0.45:
          return "A regarder les yeux fermés"
      else:
          return "Nous ne pouvons pas nous tromper"

    films_reco_planned['reco'] = films_reco_planned['sim_score'].apply(avis_reco)

    return films_reco_planned
@st.cache_data
def recommandation_films_affiche(title):
    # Vérifier si le titre existe dans le DataFrame
    if title not in df['Titre'].values:
        raise ValueError(f"Le titre '{title}' n'existe pas dans les données.")
    
    idx = df[df['Titre'] == title].index[0] # Récupérer l'index du film à recommander
    
    sim_scores = list(enumerate(cosine_sim[idx])) # Calculer les similarités cosinus
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Trier les films similaires par score de similarité

    movie_indices = [i[0] for i in sim_scores]

    films_reco_affiche = df.iloc[movie_indices] # Retourner les films recommandés

    films_reco_affiche = films_reco_affiche[films_reco_affiche['status']== 'Au cinéma']
    films_reco_affiche = films_reco_affiche[:1]
    
    films_reco_affiche['sim_score'] = [sim_scores[i+1][1] for i in range(len(films_reco_affiche))] # Ajouter les scores de similarité aux films recommandés

    def avis_reco(sim_score):
      if sim_score <= 0.10:
          return "Éviter à tout prix"
      if sim_score <= 0.15:
          return "Pas convaincant"
      if sim_score <= 0.30:
          return "Pas mal, mais à voir"
      if sim_score <= 0.45:
          return "A regarder les yeux fermés"
      else:
          return "Nous ne pouvons pas nous tromper"

    films_reco_affiche['reco'] = films_reco_affiche['sim_score'].apply(avis_reco)

    return films_reco_affiche


def afficher_films_alignes(movies_sample, category, cols_count=4):
    if len(movies_sample) == 0:
        return None
    affiche_clickable = [movie['poster_path'] for _, movie in movies_sample.iterrows()]
    selected_index = clickable_images(
        affiche_clickable,
        titles=[f"{movie['Titre']} ({movie['note_IMDB']}⭐)" for _, movie in movies_sample.iterrows()],
        div_style={"display": "flex", "flex-wrap": "wrap", "justify-content": "center", "gap": "10px"},
        img_style={"width": "175px", "border-radius": "10px", "transition": "transform 0.3s"}
    )
    if selected_index is not None and selected_index < len(movies_sample):
        st.session_state[f"selected_movie_{category}"] = movies_sample.iloc[selected_index]

films_affiche = df[df['status'] == 'Au cinéma']

# Initialisation de session_state
if "movies_defilement" not in st.session_state:
    if len(films_affiche) >= 9:
        st.session_state.movies_defilement = films_affiche.sample(9).reset_index(drop=True)
    else:
        st.session_state.movies_defilement = films_affiche.reset_index(drop=True)  # Prend tous les films si < 9

# Initialisation de l'index de sélection
if "index" not in st.session_state:
    st.session_state.index = 0

# Fonction d'affichage des films secondaires
def afficher_films_secondaires(movies_defilement):
    cols = st.columns(len(movies_defilement))  # Adapter le nombre de colonnes
    
    if not movies_defilement.empty and all(col in movies_defilement.columns for col in ['poster_path', 'Titre', 'note_IMDB']):
        for i, movie in enumerate(movies_defilement.itertuples()):
            with cols[i]:  
                img_size = "350px"

                # CSS pour l'effet de survol
                st.markdown(f"""
                <style>
                    .movie-{i} img {{
                        width: {img_size};
                        transition: transform 0.3s;
                    }}
                    .movie-{i} img:hover {{
                        transform: scale(1.1);
                    }}
                </style>
                """, unsafe_allow_html=True)

                # Affichage de l'image et du score IMDB
                st.markdown(
                    f'<div class="movie-{i}"><img src="{movie.poster_path}" width="300" alt="{movie.Titre}"><br><b>{movie.note_IMDB}</b></div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("Le DataFrame ne contient pas les données nécessaires.")
# Affichage de la page d'accueil
def afficher_page_accueil():
# Centrer le titre avec du CSS via st.markdown
    st.markdown("""
    <style>
        .title {
            text-align: center;
            margin-bottom: 20px;  /* Ajouter de l'espace sous le titre */
        }
        .subtitle {
            text-align: center;
            font-size: 35px;
            margin-top: 20px;  /* Ajouter de l'espace au-dessus du sous-titre */
            margin-bottom: 30px;  /* Ajouter de l'espace sous le sous-titre */
        }
        .description {
            text-align: center;
            font-size: 25px;
            margin-bottom: 40px;  /* Ajouter de l'espace sous la description */
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="title">🎟️ Bienvenue sur votre système de recommandation 🎟️</h1>', unsafe_allow_html=True)

# Sous-titre centré avec espace autour
    st.markdown('<h2 class="subtitle">🍿Découvrez notre sélection de films🍿</h2>', unsafe_allow_html=True)

# Paragraphe avec texte plus grand et aéré
    st.markdown("""
    <p style="text-align: center; font-size: 25px;">
        Le cinéma Le Sénéchale vous propose un nouveau système de recommandation, car nous savons que chacun a une vision, une approche et une sensibilité uniques face au septième art. 
        C'est pourquoi nous vous offrons la possibilité de choisir votre film favori, et nous vous recommanderons ensuite des films capables de susciter les mêmes émotions.
        Une collaboration avec l'équipe de data analyst de la Wild Code School !
    </p>
    """, unsafe_allow_html=True)

# Ajout de sauts de ligne supplémentaires via <br>
    st.markdown("<br>", unsafe_allow_html=True)  # Deux sauts de ligne pour plus d'aération

# Sous-titre supplémentaire ou autre contenu
    st.markdown("""
    <p style='text-align: center; font-size: 35px;'>
        🎥 <b>Explorez le monde du cinéma avec nous !</b>🎥
    </p>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 30px;'>
        🎞️ <b>Actuellement de votre cinéma !</b>🎞️
    </p>
    """, unsafe_allow_html=True)
    afficher_films_secondaires(st.session_state.movies_defilement)

# Ajout de "Aucun film sélectionné" au selectbox pour l'état initial

# Affichage de la selectbox
Films_utilisateur = st.selectbox("🎥 Film que vous avez aimé", options=["Aucun film sélectionné"] + df['Titre'].tolist(), index=0)

# Si aucun film n'est sélectionné, afficher la page d'accueil
if Films_utilisateur == "Aucun film sélectionné":
    afficher_page_accueil()

else:
    st.title("🎬 Nos recommandations parmi :")

    with st.sidebar:
        st.write("### 📌 Détails du Film Sélectionné")
        
        # Récupérer les détails du film sélectionné
        selected_movie = df[df['Titre'] == Films_utilisateur].iloc[0]

        # Affichage des détails du film
        st.markdown(
            f'<div style="text-align: center;"><img src="{selected_movie["poster_path"]}" width="200"></div>',
            unsafe_allow_html=True
        )
        st.write(f"**Titre :** {selected_movie['Titre']}")
        st.write(f"**Note IMDB :** {selected_movie['note_IMDB']}⭐")
        st.write(f"**Résumé :** {selected_movie['SynopsisFR']}")
        st.write(f"**Acteurs :** {selected_movie['top_actors']}")
        st.write(f"**Studio :** {selected_movie['studios']}")
        st.write(f"**Mot clé :** {selected_movie['keyword']}")

        # Bande-annonce
        if selected_movie['trailer_url'] and isinstance(selected_movie['trailer_url'], str) and selected_movie['trailer_url'].startswith("http"):
            st.video(selected_movie['trailer_url'])
        else:
            st.write("Trailer non disponible")

    # Exécuter les recommandations
    def afficher_details_film(film, categorie):
        if film is not None:
            with st.expander(f"{categorie} : {film['Titre']} ({film['note_IMDB']}⭐)", expanded=True):
                st.markdown(f'<div style="text-align: center;"><img src="{film["poster_path"]}" width="150"></div>', unsafe_allow_html=True)
                st.write(f"**Résumé :** {film['SynopsisFR']}")
                st.write(f"**Date de sortie :** {film['release_date']}")
                st.write(f"**Acteurs :** {film['top_actors']}")
                st.write(f"**Studio :** {film['studios']}")
                st.write(f"**Disponible sur :** {film['streaming_platforms']}")
                st.write(f"**Genre :** {film['genres_x']}")
                if film['trailer_url'] and isinstance(film['trailer_url'], str) and film['trailer_url'].startswith("http"):
                    st.video(film['trailer_url'])
                else:
                    st.write("Trailer non disponible")

    try:
        films_recommandes = recommandation_films(Films_utilisateur)
    except ValueError as e:
        st.error(str(e))
        films_recommandes = pd.DataFrame()

    try:
        films_a_venir = recommandation_films_planned(Films_utilisateur)
    except ValueError:
        films_a_venir = pd.DataFrame()

    try:
        films_affiche = recommandation_films_affiche(Films_utilisateur)
    except ValueError:
        films_affiche = pd.DataFrame()

# Création des trois colonnes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("## 🎟️Les films déja sortis")
        if not films_recommandes.empty:
            afficher_films_alignes(films_recommandes, "reco", cols_count=4)
        else:
            st.write("Aucune recommandation trouvée.")
        afficher_details_film(st.session_state.get("selected_movie_reco"), "🎟️ Film recommandé")

    with col2:
        st.write("## 🔜 Films à venir")
        if not films_a_venir.empty:
            afficher_films_alignes(films_a_venir, "planned")
        else:
            st.write("Aucun film à venir trouvé.")
        afficher_details_film(st.session_state.get("selected_movie_planned"), "🔜 Film à venir")

    with col3:
        st.write("## 🎬 Films à l'affiche dans votre cinéma")
        if not films_affiche.empty:
            afficher_films_alignes(films_affiche, "affiche")
        else:
            st.write("Aucun film à l'affiche trouvé.")
        afficher_details_film(st.session_state.get("selected_movie_affiche"), "🎬 Film à l'affiche")


#streamlit run "C:\Users\borami\Desktop\projet  wild\systeme reco\Le sénéchal systeme de recommandation.py"