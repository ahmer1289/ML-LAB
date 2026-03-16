from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path
from models import service

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/image')
def image_page():
    return render_template('image.html')


@app.route('/text_gen')
def text_page():
    return render_template('text_gen.html')


@app.route('/translate')
def translate_page():
    return render_template('translate.html')

@app.route('/sentiment')
def sentiment_page():
    return render_template('sentiment.html')


@app.route('/qa')
def qa_page():
    return render_template('qa.html')

@app.route('/docs')
def docs_page():
    return render_template('docs.html')


@app.route('/kmeans')
def kmeans():
    return render_template('kmeans.html')

@app.route('/apriori')
def apriori():
    return render_template('apriori.html')

@app.route('/dbscan')
def dbscan():
    return render_template('dbscan.html')

@app.route('/zeroshot')
def zeroshot():
    return render_template('zeroshot.html')


@app.route('/api/image_classify', methods=['POST'])
def api_image_classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        f = request.files['image']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        filename = secure_filename(f.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        path = UPLOAD_DIR / save_name
        f.save(path)
        result = service.classify_image(str(path))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/text_generate', methods=['POST'])
def api_text_generate():
    try:
        data = request.json or {}
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        result = service.generate_text(prompt)
        return jsonify({'text': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/translate', methods=['POST'])
def api_translate():
    try:
        data = request.json or {}
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = service.translate_en_to_ur(text)
        return jsonify({'text': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sentiment_voice', methods=['POST'])
def api_sentiment_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        f = request.files['audio']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        filename = secure_filename(f.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        path = UPLOAD_DIR / save_name
        f.save(path)
        text = service.speech_to_text(str(path))
        sentiment = service.sentiment_from_text(text)
        return jsonify({'text': text, 'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/qa_voice', methods=['POST'])
def api_qa_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        f = request.files['audio']
        if f.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

      
        context = request.form.get('context')
        if context and context.strip() == "":
            context = None

        filename = secure_filename(f.filename)
        save_name = f"{uuid.uuid4().hex}_{filename}"
        path = UPLOAD_DIR / save_name
        f.save(path)

        question = service.speech_to_text(str(path))
        answer = service.answer_question(question, context=context)
        audio_path = service.text_to_speech(answer)
        return send_file(audio_path, as_attachment=True, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route('/api/zero-shot-classify', methods=['POST'])
def zero_shot_classify():
    try:
        
        
        data = request.json
        text = data.get('text', '')
        labels = [label.strip() for label in data.get('labels', '').split(',')]
        
        if not text or not labels:
            return jsonify({'error': 'Text and labels are required'}), 400
        
      
        result = classifier(text, labels, multi_class=True)
        
        predictions = []
        for label, score in zip(result['labels'], result['scores']):
            predictions.append({
                'label': label,
                'confidence': round(score * 100, 2)
            })
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/kmeans-cluster', methods=['POST'])
def kmeans_cluster():
    try:
        import pandas as pd
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        
        if 'dataset' not in request.files:
            return jsonify({'error': 'No dataset provided'}), 400
        
        file = request.files['dataset']
        df = pd.read_csv(file)
        
        n_clusters = int(request.form.get('n_clusters', 3))
        
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return jsonify({'error': 'No numeric columns found in dataset'}), 400
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        df['Cluster'] = clusters
        
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i]
            cluster_stats.append({
                'cluster': i,
                'size': len(cluster_data),
                'mean_values': cluster_data[numeric_df.columns].mean().to_dict()
            })
        
      
        return jsonify({
            'clusters': cluster_stats,
            'inertia': float(kmeans.inertia_),
            'data': df.to_dict(orient='records')[:100]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dbscan-cluster', methods=['POST'])
def dbscan_cluster():
    try:
        import pandas as pd
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        if 'dataset' not in request.files:
            return jsonify({'error': 'No dataset provided'}), 400
        
        file = request.files['dataset']
        df = pd.read_csv(file)
        
        eps = float(request.form.get('eps', 0.5))
        min_samples = int(request.form.get('min_samples', 5))
        
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return jsonify({'error': 'No numeric columns found in dataset'}), 400
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(scaled_data)
        
        df['Cluster'] = clusters
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        cluster_stats = []
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i]
            cluster_stats.append({
                'cluster': i,
                'size': len(cluster_data),
                'mean_values': cluster_data[numeric_df.columns].mean().to_dict()
            })
        
        return jsonify({
            'clusters': cluster_stats,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'data': df.to_dict(orient='records')[:100]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/apriori-rules', methods=['POST'])
def apriori_rules():
    try:
        import pandas as pd
        import numpy as np
        from mlxtend.frequent_patterns import apriori, association_rules
        
        if 'dataset' not in request.files:
            return jsonify({'error': 'No dataset provided'}), 400
        
        file = request.files['dataset']
        df = pd.read_csv(file)
        
        min_support = float(request.form.get('min_support', 0.1))
        min_confidence = float(request.form.get('min_confidence', 0.5))
        
        if df.select_dtypes(include=[bool, 'int']).shape[1] > 0:
            item_set = df.select_dtypes(include=[bool, 'int'])
        else:
            item_set = (df > 0).astype(int)
        
        frequent_itemsets = apriori(item_set, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) == 0:
            return jsonify({'error': 'No frequent itemsets found. Try lower min_support.'}), 400
        
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        if len(rules) == 0:
            return jsonify({'error': 'No rules generated. Try lower min_confidence.'}), 400
        
        rules_list = []
        for idx, rule in rules.iterrows():
            rules_list.append({
                'antecedent': str(list(rule['antecedents'])),
                'consequent': str(list(rule['consequents'])),
                'support': float(rule['support']),
                'confidence': float(rule['confidence']),
                'lift': float(rule['lift'])
            })
        
        return jsonify({
            'rules': rules_list,
            'total_rules': len(rules_list)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True, use_reloader=False)