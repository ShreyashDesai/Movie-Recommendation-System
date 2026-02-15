# 🎬 Movie Recommendation System

## 📌 Overview
This project suggests movies based on user ratings and genres.  
It implements:
- **Content-Based Filtering** (using movie genres)
- **Collaborative Filtering** (using user ratings)

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/YourUsername/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
▶️ Usage
Run the script:

bash
python main.py
📊 Example Output
Code
Content-based recommendations for 'Toy Story (1995)':
['Jumanji (1995)', 'Grumpier Old Men (1995)', 'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)', 'Heat (1995)']

Collaborative recommendations for user 1:
['GoldenEye (1995)', 'Four Rooms (1995)', 'Get Shorty (1995)', 'Copycat (1995)', 'Casino (1995)']
🚀 Future Work
Add hybrid recommendation (combine content + collaborative).

Use deep learning (Autoencoders, Neural Collaborative Filtering).

Deploy as a web app with Streamlit.

📂 Project Structure
Code
movie-recommendation-system/
│── main.py              # Core script
│── requirements.txt     # Dependencies
│── README.md            # Documentation
│── movies.csv           # Movie metadata
│── ratings.csv          # User ratings
📬 Contact
