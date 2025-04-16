from flask import Flask, request, render_template_string
from app.doc_processor import generate_query, process_documents

def create_app():
    app = Flask(__name__)

    # A minimal HTML form using a route
    @app.route('/', methods=['GET', 'POST'])
    def home():
        if request.method == 'POST':
            user_question = request.form.get('domanda', '')
            if not user_question:
                return render_template_string(form_template, answer="fai una domanda")
            
            # Generate the Solr query from user question
            solr_query = generate_query(user_question)

            # Process documents & get answer
            answer = process_documents(user_question, solr_query)
            return render_template_string(form_template, answer=answer)
        return render_template_string(form_template, answer="")

    return app

form_template = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG</title>
</head>
<body>
    <h1>fai una domanda</h1>
    <form method="POST" action="/">
        <label for="domanda">Domanda:</label><br>
        <textarea name="domanda" rows="4" cols="50"></textarea><br><br>
        <button type="submit">Ask</button>
    </form>
    {% if answer %}
        <h2>Risposta:</h2>
        <div style="white-space: pre-wrap;">{{ answer }}</div>
    {% endif %}
</body>
</html>
"""