# This script creates a comprehensive recipe recommendation system using Streamlit.
# It includes user management, a recipe dataset, a calorie calculator,
# and both filter-based and content-based recommendation features.

import streamlit as st
import pandas as pd
import bcrypt
import os
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- USER MANAGEMENT FUNCTIONS ---
# This section handles user login and registration.

# Path for user credentials file
credentials_file = 'data/users.csv'


# Function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# Function to verify password
def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())


# Function to load user credentials
def load_users():
    if os.path.exists(credentials_file):
        return pd.read_csv(credentials_file)
    else:
        # Create the file if it doesn't exist
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(credentials_file, index=False)
        return df


# Function to save a new user
def save_user(username, password):
    users_df = load_users()
    new_user = pd.DataFrame({"username": [username], "password": [hash_password(password)]})
    updated_users = pd.concat([users_df, new_user], ignore_index=True)
    updated_users.to_csv(credentials_file, index=False)


# Login Function
def login_user(username, password):
    users_df = load_users()
    user_record = users_df[users_df['username'] == username]
    if not user_record.empty and verify_password(password, user_record.iloc[0]['password']):
        return True
    else:
        return False


# Registration Function
def register_user(username, password):
    users_df = load_users()
    if username in users_df['username'].values:
        st.warning("Username already taken. Please choose another.")
    else:
        save_user(new_username, new_password)
        st.success("Registration successful! You can now log in.")


# Initialize session state for login status and current user
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# --- GLOBAL DATA LOADING & PREPROCESSING ---
# This section loads the dataset and standardizes column names.
# It is placed outside the login block so the data is always available.
try:
    recipes_13k_df = pd.read_csv('data/updated_recipes_with_nutrition.csv')    # Strip whitespace from column names
    recipes_13k_df.columns = recipes_13k_df.columns.str.strip()
    # Convert all column names to lowercase to prevent KeyErrors
    recipes_13k_df.columns = recipes_13k_df.columns.str.lower()

    # Fill missing values and ensure consistent data types using lowercase names
    recipes_13k_df['cleaned_ingredients'] = recipes_13k_df['cleaned_ingredients'].fillna('')
    recipes_13k_df['title'] = recipes_13k_df['title'].fillna('')
    recipes_13k_df['instructions'] = recipes_13k_df['instructions'].fillna('')

    # Handle boolean columns and fill NaNs
    dietary_columns = ['vegan', 'vegetarian', 'gluten_free', 'lactose_free', 'non_vegetarian']
    for col in dietary_columns:
        recipes_13k_df[col] = recipes_13k_df[col].apply(lambda x: True if str(x).lower() == 'true' else False)
    recipes_13k_df[dietary_columns] = recipes_13k_df[dietary_columns].fillna(False)

except FileNotFoundError:
    st.error("Error: '13k-recipes-complete-updated.csv' not found. Please ensure the file is in the same directory.")
    st.stop()
except KeyError as e:
    st.error(f"Error: Missing column in the CSV file. Please check for a column named {e}.")
    st.stop()


# --- HELPER FUNCTIONS FOR NEW FEATURES ---
# Calorie Calculator
def calculate_tdee(age, height, weight, gender, activity_level):
    if gender == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Activity multipliers
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }
    tdee = bmr * activity_multipliers[activity_level]
    return bmr, tdee


# Content-Based Recommender
def get_recipe_recommendations(title, df, cosine_sim, indices, top_n=5):
    if title not in indices:
        return pd.DataFrame()  # Return empty if title not found

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar recipes, excluding the recipe itself
    sim_scores = sim_scores[1:top_n + 1]

    # Get the recipe indices
    recipe_indices = [i[0] for i in sim_scores]

    # Return the top N most similar recipes
    return df.iloc[recipe_indices]


# Pre-calculate TfidfVectorizer and cosine similarity for content-based recommendations
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(recipes_13k_df['cleaned_ingredients'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(recipes_13k_df.index, index=recipes_13k_df['title']).drop_duplicates()

# --- APP LAYOUT AND LOGIN/REGISTER FORM ---
st.sidebar.title("Nutrimate Login")
login_tab, register_tab = st.sidebar.tabs(["Login", "Register"])

with login_tab:
    st.write("Log in to access Nutrimate.")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    login_button = st.button("Log in")

    if login_button:
        if login_user(username, password):
            st.success(f"Welcome, {username}!")
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.rerun()  # Force rerun to show the main content
        else:
            st.error("Invalid username or password.")

with register_tab:
    st.write("Create a new Nutrimate account.")
    new_username = st.text_input("New Username", key="register_username")
    new_password = st.text_input("New Password", type="password", key="register_password")
    register_button = st.button("Register")

    if register_button:
        if new_username and new_password:
            register_user(new_username, new_password)
        else:
            st.warning("Please fill in both fields.")

# --- INGREDIENT CATEGORY DATA ---
ingredient_categories = {
    'meat_products': ['chicken', 'beef', 'pork', 'lamb', 'bacon', 'sausage', 'fish', 'seafood', 'shrimp', 'turkey',
                      'ham', 'veal', 'duck', 'goat', 'crab', 'lobster', 'tuna', 'salmon', 'anchovy', 'prosciutto',
                      'salami', 'chorizo'],
    'dairy_products': ['milk', 'cheese', 'cream', 'butter', 'yogurt', 'whey', 'ricotta', 'mozzarella', 'parmesan',
                       'cheddar', 'buttermilk', 'sour cream', 'mascarpone', 'ghee', 'half and half', 'heavy cream'],
    'gluten_products': ['wheat', 'flour', 'pasta', 'bread', 'barley', 'rye', 'couscous', 'semolina', 'farro',
                        'breadcrumbs', 'noodles', 'spaghetti', 'macaroni', 'crackers'],
    'eggs': ['egg', 'eggs', 'egg white', 'egg yolk', 'egg whites', 'egg yolks'],
    'seafood': ['fish', 'shrimp', 'crab', 'lobster', 'clam', 'mussel', 'oyster', 'scallop', 'squid', 'octopus'],
    'nuts_seeds': ['almond', 'walnut', 'pecan', 'cashew', 'pistachio', 'peanut', 'sesame', 'pine nut', 'chia',
                   'flaxseed'],
    'vegetables': ['onion', 'garlic', 'tomato', 'carrot', 'celery', 'pepper', 'lettuce', 'spinach', 'broccoli',
                   'cauliflower'],
    'fruits': ['apple', 'banana', 'orange', 'lemon', 'lime', 'berry', 'strawberry', 'blueberry', 'raspberry', 'grape'],
    'grains': ['rice', 'quinoa', 'oat', 'corn', 'millet', 'buckwheat', 'amaranth', 'wild rice'],
    'legumes': ['bean', 'lentil', 'chickpea', 'pea', 'soybean', 'tofu', 'tempeh'],
    'herbs_spices': ['basil', 'oregano', 'thyme', 'rosemary', 'cumin', 'coriander', 'paprika', 'cinnamon', 'nutmeg',
                     'ginger'],
    'sweeteners': ['sugar', 'honey', 'maple syrup', 'agave', 'stevia', 'molasses', 'corn syrup'],
    'oils': ['olive oil', 'vegetable oil', 'coconut oil', 'sesame oil', 'canola oil', 'sunflower oil'],
    'alcoholic': ['wine', 'beer', 'vodka', 'rum', 'whiskey', 'brandy', 'sherry', 'cognac']
}


# Function to filter recipes based on dietary preferences, allergens, and ingredient categories
def filter_by_preferences(df, dietary_preferences, allergens, include_category, exclude_category):
    if dietary_preferences.get('Vegan', False):
        df = df[~df['cleaned_ingredients'].str.contains('|'.join(ingredient_categories['dairy_products']), case=False,
                                                        na=False)]
        df = df[df['vegan'] == True]

    if dietary_preferences.get('Vegetarian', False):
        df = df[df['vegetarian'] == True]

    if dietary_preferences.get('Gluten-Free', False):
        df = df[df['gluten_free'] == True]

    if dietary_preferences.get('Lactose-Free', False):
        df = df[df['lactose_free'] == True]

    if dietary_preferences.get('Non-Vegetarian', False):
        df = df[(df['vegetarian'] == False) & (df['vegan'] == False)]

    for allergen in allergens:
        df = df[~df['cleaned_ingredients'].str.contains(allergen, case=False, na=False)]

    if include_category:
        included_ingredients = []
        for category in include_category:
            included_ingredients.extend(ingredient_categories[category])
        included_ingredients = [ingredient.lower() for ingredient in included_ingredients]
        df = df[df['cleaned_ingredients'].apply(
            lambda x: any(ingredient in x.lower() for ingredient in included_ingredients))]

    if exclude_category:
        excluded_ingredients = []
        for category in exclude_category:
            excluded_ingredients.extend(ingredient_categories[category])
        excluded_ingredients = [ingredient.lower() for ingredient in excluded_ingredients]
        df = df[~df['cleaned_ingredients'].apply(
            lambda x: any(ingredient in x.lower() for ingredient in excluded_ingredients))]

    return df


# --- MAIN APP CONTENT (ONLY FOR LOGGED-IN USERS) ---
st.markdown("""
    <style>
    .main-title {
        color: #F4A300;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subheader {
        color: #2E8B57;
        font-size: 24px;
        text-align: center;
    }
    .button {
        background-color: #F4A300;
        color: white;
        font-size: 18px;
    }
    .button:hover {
        background-color: #FF6F00;
    }
    body {
        background-color: #F0F8FF;
    }
    .recommendation-card {
        background-color: #FFF3E0;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 10px;
        font-family: "Comic Sans MS", cursive, sans-serif;
    }
    .recipe-title {
        font-size: 24px;
        color: #FF6F00;
        font-weight: bold;
    }
    .recipe-ingredients, .recipe-instructions, .recipe-nutrition {
        font-size: 18px;
        color: #555555;
    }
    </style>
""", unsafe_allow_html=True)

if st.session_state.logged_in:
    st.markdown('<p class="main-title">🍽 Nutrimate - Recipe Recommender 🍴</p>', unsafe_allow_html=True)
    st.image("images/logo.jpg", width=200)


    # --- Calorie Calculator Section ---
    st.markdown("---")
    st.subheader("Calorie Calculator")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
        activity_level = st.selectbox("Activity Level",
                                      ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("Calculate Daily Calorie Needs"):
        bmr, tdee = calculate_tdee(age, height, weight, gender, activity_level)
        st.success(f"Your Basal Metabolic Rate (BMR) is approximately {bmr:.2f} kcal.")
        st.success(f"Your Total Daily Energy Expenditure (TDEE) is approximately {tdee:.2f} kcal.")

    # --- FAQ Chatbot Section ---
    st.markdown("---")
    st.subheader("Nutritional FAQ Chatbot")
    # Initialize AIML kernel and learn the FAQ file
    kernal = aiml.Kernel()
    faq_file = "chatbot/faq.aiml"
    if not os.path.exists(faq_file):
        st.error(f"AIML file {faq_file} not found!")
    else:
        kernal.learn(faq_file)

    category = st.selectbox(
        "Select a Category",
        ("Weight Loss", "Weight Gain", "Fitness", "General Health")
    )

    if category == "Weight Loss":
        query = "IMPORTANT INGREDIENTS FOR WEIGHT LOSS"
    elif category == "Weight Gain":
        query = "IMPORTANT INGREDIENTS FOR WEIGHT GAIN"
    elif category == "Fitness":
        query = "IMPORTANT INGREDIENTS FOR FITNESS"
    elif category == "General Health":
        query = "IMPORTANT INGREDIENTS FOR GENERAL HEALTH"

    response = kernal.respond(query)
    st.subheader(f"FAQ: {category}")
    st.write(response)

    # --- Recipe Search Section ---
    st.markdown("---")
    st.subheader("Search Recipes")
    search_query = st.text_input("Search for a recipe (e.g., 'pasta', 'chicken soup', 'vegan curry')")
    if st.button("Search"):
        if search_query:
            search_results = recipes_13k_df[
                recipes_13k_df['title'].str.contains(search_query, case=False, na=False) |
                recipes_13k_df['cleaned_ingredients'].str.contains(search_query, case=False, na=False) |
                recipes_13k_df['instructions'].str.contains(search_query, case=False, na=False)
                ]
            if not search_results.empty:
                st.subheader("Search Results:")
                for index, row in search_results.iterrows():
                    st.markdown(
                        f'<div class="recommendation-card">'
                        f'<p class="recipe-title">{row["title"]}</p>'
                        f'<p class="recipe-ingredients">Ingredients: {row["cleaned_ingredients"]}</p>'
                        f'<p class="recipe-instructions">Instructions: {row["instructions"]}</p>'
                        f'<p class="recipe-nutrition">Estimated Calories: {row["estimated_calories"]} kcal</p>'
                        f'<p class="recipe-nutrition">Healthiness Score: {row["healthiness_score"]}</p>'
                        f'</div>', unsafe_allow_html=True)
            else:
                st.warning("No recipes found matching your search query.")
        else:
            st.warning("Please enter a search query.")

    # --- Filter-Based Recommender Section ---
    st.markdown("---")
    st.subheader("Filter Recipes")
    preferred_ingredient = st.text_input("Preferred ingredient (e.g., chicken, spinach, etc.)")
    st.subheader("Ingredient Category Filters")
    include_category = st.multiselect("Select Categories to Include", list(ingredient_categories.keys()))
    exclude_category = st.multiselect("Select Categories to Exclude", list(ingredient_categories.keys()))

    dietary_preferences = {
        'Vegan': st.checkbox('Vegan'),
        'Vegetarian': st.checkbox('Vegetarian'),
        'Gluten-Free': st.checkbox('Gluten-Free'),
        'Lactose-Free': st.checkbox('Lactose-Free'),
        'Non-Vegetarian': st.checkbox('Non-Vegetarian')
    }

    allergens = st.multiselect('Select allergens to avoid', ['Nuts', 'Dairy', 'Eggs', 'Soy', 'Wheat'])

    # Button to generate recommendations
    if st.button("Get Filtered Recipes", key="filter_recommend"):
        try:
            # Filter by preferred ingredient
            if preferred_ingredient:
                filtered_df = recipes_13k_df[
                    recipes_13k_df['cleaned_ingredients'].str.contains(preferred_ingredient, case=False, na=False)]
            else:
                filtered_df = recipes_13k_df

            # Filter by dietary preferences, allergens, and ingredient categories
            filtered_df = filter_by_preferences(filtered_df, dietary_preferences, allergens, include_category,
                                                exclude_category)

            if filtered_df.empty:
                st.error("No suitable recipes found based on your inputs.")
            else:
                st.subheader("Filtered Recipes:")
                for index, row in filtered_df.iterrows():
                    st.markdown(
                        f'<div class="recommendation-card">'
                        f'<p class="recipe-title">{row["title"]}</p>'
                        f'<p class="recipe-ingredients">Ingredients: {row["cleaned_ingredients"]}</p>'
                        f'<p class="recipe-instructions">Instructions: {row["instructions"]}</p>'
                        f'<p class="recipe-nutrition">Estimated Calories: {row["estimated_calories"]} kcal</p>'
                        f'<p class="recipe-nutrition">Healthiness Score: {row["healthiness_score"]}</p>'
                        f'</div>', unsafe_allow_html=True)
        except KeyError as e:
            st.error(
                f"An error occurred: Missing expected column {e}. Please ensure your CSV has all required columns.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # --- Content-Based Recommender Section ---
    st.markdown("---")
    st.subheader("Get Similar Recipes")
    recipe_titles = recipes_13k_df['title'].tolist()
    selected_recipe = st.selectbox("Select a recipe you like:", recipe_titles)

    if st.button("Find Similar Recipes"):
        similar_recipes = get_recipe_recommendations(selected_recipe, recipes_13k_df, cosine_sim, indices)
        if not similar_recipes.empty:
            st.subheader(f"Recipes similar to '{selected_recipe}':")
            for index, row in similar_recipes.iterrows():
                st.markdown(
                    f'<div class="recommendation-card">'
                    f'<p class="recipe-title">{row["title"]}</p>'
                    f'<p class="recipe-ingredients">Ingredients: {row["cleaned_ingredients"]}</p>'
                    f'<p class="recipe-instructions">Instructions: {row["instructions"]}</p>'
                    f'<p class="recipe-nutrition">Estimated Calories: {row["estimated_calories"]} kcal</p>'
                    f'<p class="recipe-nutrition">Healthiness Score: {row["healthiness_score"]}</p>'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.warning("No similar recipes found or the selected recipe could not be processed.")


else:
    st.warning("Please log in to access the Nutrimate app.")
