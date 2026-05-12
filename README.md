# Nutrimate️

Live Demo
https://nutrimateproject-hnkwjuf4kbsvhh5hdyybnr.streamlit.app/

Nutrimate is an AI-powered nutrition and recipe recommendation system built using Streamlit and machine learning techniques. The application provides personalized recipe recommendations based on dietary preferences, ingredients, and nutritional goals.

## Features

* User authentication system with secure password hashing using bcrypt
* Recipe recommendation using TF-IDF vectorization and cosine similarity
* Dietary filtering for Vegan, Vegetarian, Gluten-Free, Lactose-Free, and Non-Vegetarian users
* Calorie calculator using BMR and TDEE calculations
* Nutritional FAQ chatbot using AIML
* Ingredient-based recipe search and filtering
* Healthiness score and calorie estimation for recipes

## Tech Stack

* Python
* Streamlit
* Pandas
* Scikit-learn
* AIML
* bcrypt

## Machine Learning Concepts Used

* TF-IDF Vectorization
* Cosine Similarity
* Content-Based Recommendation System

## Dataset

The application uses a recipe dataset containing 13k+ recipes with nutritional and ingredient information.

## Project Structure

```text
NutrimateProject/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── updated_recipes_with_nutrition.csv
│   └── users.csv
│
├── chatbot/
│   └── faq.aiml
│
├── images/
│   └── logo.jpg
```

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/NutrimateProject.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Future Improvements

* Cloud deployment using AWS or Render
* Docker containerization
* Advanced nutrition analytics
* User profile personalization
* Database integration

## Author

Diya Mareeza Doyle
