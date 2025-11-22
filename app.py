print('hey')

import requests
import json
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

USERNAME = 'manvendra'
PASSWORD = '12345'

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def do_login():
    username = request.form['username']
    password = request.form['password']
    if username == USERNAME and password == PASSWORD:
        session['logged_in'] = True
        return redirect(url_for('upload'))
    else:
        return "Wrong username or password!"

@app.route('/upload')
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    file = request.files['file']
    if not file:
        return "No file uploaded!"

    try:
        import os
        filename=file.filename.lower()
        if filename.endswith('.csv'):
            df=pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            df=pd.read_excel(file)
        else:
            return "Unsupported File. Please upload CSV or Excel file."

        if 'comment' not in df.columns:
            return "CSV must have a column named 'comment'"

        comments = df['comment'].tolist()

        results = []
        for comment in comments:
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze this customer comment: '{comment}'. Respond ONLY with a valid JSON object containing keys: sentiment, category, and themes (as a list). Do not include any explanation or markdown."
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 300
            }

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }

            response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(payload))
            print("Raw response:", response.text)

            try:
                data = response.json()
                if "choices" in data and data["choices"]:
                    raw_text = data["choices"][0]["message"]["content"]
                    try:
                        parsed = json.loads(raw_text)
                        analysis = parsed
                    except:
                        analysis = {
                            "sentiment": "Error",
                            "category": "Error",
                            "themes": [raw_text]
                        }
                else:
                    analysis = {
                        "sentiment": "Missing",
                        "category": "Missing",
                        "themes": ["Unexpected format"]
                    }
            except Exception as e:
                analysis = {
                    "sentiment": "Error",
                    "category": "Error",
                    "themes": [str(e)]
                }

            results.append({
                "comment": comment,
                "analysis": analysis
            })

        html = """
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; font-family: Arial;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th>Comment</th>
                    <th>Sentiment</th>
                    <th>Category</th>
                    <th>Themes</th>
                </tr>
            </thead>
            <tbody>
        """

        for r in results:
            html += f"""
                <tr>
                    <td>{r['comment']}</td>
                    <td>{r['analysis'].get('sentiment', '')}</td>
                    <td>{r['analysis'].get('category', '')}</td>
                    <td>{', '.join(r['analysis'].get('themes', []))}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        <br><a href="/download"><button>Download CSV</button></a>
        """

        session['results'] = results  # ✅ Store results for download
        return html

    except Exception as e:
        return f"Error processing file: {str(e)}"

@app.route('/download')
def download():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    results = session.get('results')
    if not results:
        return "No results to download."

    df = pd.DataFrame([{
        "Comment": r['comment'],
        "Sentiment": r['analysis'].get('sentiment', ''),
        "Category": r['analysis'].get('category', ''),
        "Themes": ', '.join(r['analysis'].get('themes', []))
    } for r in results])

    csv_data = df.to_csv(index=False)

    return (
        csv_data,
        200,
        {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename="analysis_results.csv"'
        }
    )

if __name__ == '__main__':
    app.run(debug=True)