from flask import Flask, request, render_template_string
from app.doc_processor import generate_query, process_documents

def create_app():
    app = Flask(__name__)

    # A minimal HTML form using a route
    @app.route('/', methods=['GET', 'POST'])
    def home():
        if request.method == 'POST':
            user_question = request.form.get('question', '')
            if not user_question:
                return render_template_string(form_template, answer="Please enter a question.")
            
            # Generate the Solr query from user question
            solr_query = generate_query(user_question)

            # Process documents & get answer
            answer = process_documents(user_question, solr_query)
            return render_template_string(form_template, answer=answer)
        return render_template_string(form_template, answer="")

    return app

# Basic HTML template with a form
form_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Legal RAG App</title>
</head>
<body>
    <h1>Ask a Legal Question</h1>
    <form method="POST" action="/">
        <label for="question">Question:</label><br>
        <textarea name="question" rows="4" cols="50"></textarea><br><br>
        <button type="submit">Ask</button>
    </form>
    {% if answer %}
        <h2>Answer:</h2>
        <div style="white-space: pre-wrap;">{{ answer }}</div>
    {% endif %}
</body>
</html>
"""