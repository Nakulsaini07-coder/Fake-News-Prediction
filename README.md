# 📰 Fake News Prediction using Machine Learning  

## 📌 Project Overview  
This project aims to build a machine learning model that can detect whether a news article is **real** or **fake**.  
We use Natural Language Processing (NLP) techniques to preprocess text data and a **Logistic Regression** classifier to make predictions.  

---

## 📂 Dataset  
The dataset consists of news articles with the following fields:  

- **id** → unique id for a news article  
- **title** → title of the news article  
- **author** → author of the news article  
- **text** → main content of the article (could be incomplete)  
- **label** → target variable  
  - `1` → Fake news  
  - `0` → Real news  

---

## ⚙️ Tech Stack  
- **Programming Language**: Python 🐍  
- **Libraries**:  
  - `numpy`, `pandas` → data manipulation  
  - `nltk` → natural language processing (stopwords removal, stemming)  
  - `scikit-learn` → TF-IDF vectorization, model training, evaluation  

---

## 🧹 Data Preprocessing  
- Lowercasing text  
- Removing punctuation & special characters  
- Removing stopwords using **NLTK**  
- Stemming words with **PorterStemmer**  
- Converting text into numerical features using **TF-IDF Vectorizer**  

---

## 🏗️ Model Building  
- Train-test split: 90% training, 10% testing  
- Classifier: **Logistic Regression**  
- Evaluation metric: **Accuracy Score**  

---

## 📊 Results  
The Logistic Regression model achieved:  

- **Training Accuracy**: ~99%  
- **Testing Accuracy**: ~98%  

(*numbers inferred from typical runs — replace with your actual outputs if needed*)  

---

## 🚀 How to Run  

### 1. Clone the Repository  
```bash
git clone https://github.com/yourusername/Fake-News-Prediction.git
cd Fake-News-Prediction
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook  
Open Jupyter Notebook and run:  
```bash
jupyter notebook Fake_News_Prediction.ipynb
```

---

## 📌 Example Usage  
Once trained, the model can predict new samples:  

```python
input_data = ["Breaking news example text goes here"]
prediction = model.predict(tfidf_vectorizer.transform(input_data))

if prediction[0] == 0:
    print("✅ Real News")
else:
    print("❌ Fake News")
```

---

## 📜 License  
This project is open-source under the MIT License.  
