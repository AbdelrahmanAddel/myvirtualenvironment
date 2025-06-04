import joblib
from flask import Flask, request, jsonify

# تحميل الموديلات
svm_classifier = joblib.load("best_svm_classifier.joblib")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.joblib")

# دالة معالجة النصوص
def preprocess_text(text):
    text = text.lower().strip()
    return text

# دالة التنبؤ بالمرض
def predict_disease_for_symptoms(symptom_text):
    processed_symptom = preprocess_text(symptom_text)
    symptom_vectorized = tfidf_vectorizer.transform([processed_symptom]).toarray()
    predicted_disease = svm_classifier.predict(symptom_vectorized)
    return predicted_disease[0]

# إنشاء Flask API
app = Flask(__name__)
# تعديل المسار من هنا يا مصعب 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات من الطلب
        data = request.json
        symptom_text = data.get('text', '')

        if not symptom_text:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # التنبؤ بالمرض
        predicted_disease = predict_disease_for_symptoms(symptom_text)

        return jsonify({'prediction': predicted_disease})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


    


