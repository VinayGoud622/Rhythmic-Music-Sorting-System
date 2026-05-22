from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
from werkzeug.utils import secure_filename
from model import process_music_data

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-this-in-production')

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB limit

REQUIRED_COLUMNS = {'tempo', 'energy', 'danceability'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/process', methods=['POST'])
def process():
    # --- File presence check ---
    if 'file' not in request.files:
        flash('No file part in the request.', 'error')
        return redirect(url_for('upload'))

    file = request.files['file']

    if file.filename == '':
        flash('No file selected. Please choose a CSV file.', 'error')
        return redirect(url_for('upload'))

    # --- File type check ---
    if not allowed_file(file.filename):
        flash('Invalid file type. Only .csv files are allowed.', 'error')
        return redirect(url_for('upload'))

    # --- Secure filename ---
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # --- Column validation ---
    import pandas as pd
    try:
        df = pd.read_csv(filepath)
    except Exception:
        flash('Could not read the CSV file. Make sure it is a valid, non-empty CSV.', 'error')
        os.remove(filepath)
        return redirect(url_for('upload'))

    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        flash(f'CSV is missing required columns: {", ".join(missing)}. '
              f'Required: tempo, energy, danceability.', 'error')
        os.remove(filepath)
        return redirect(url_for('upload'))

    if df.empty:
        flash('The uploaded CSV file is empty.', 'error')
        os.remove(filepath)
        return redirect(url_for('upload'))

    # --- Processing ---
    try:
        output_file, plot_file, stats = process_music_data(filepath)
    except Exception as e:
        flash(f'An error occurred during processing: {str(e)}', 'error')
        return redirect(url_for('upload'))

    return render_template('result.html',
                           output_file=output_file,
                           plot_file=plot_file,
                           stats=stats)


@app.route('/download/<filename>')
def download(filename):
    safe_name = secure_filename(filename)
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], safe_name)

    if not os.path.exists(file_path):
        flash('File not found.', 'error')
        return redirect(url_for('index'))

    return send_file(file_path, as_attachment=True)


# --- Error handlers ---
@app.errorhandler(413)
def file_too_large(e):
    flash('File is too large. Maximum allowed size is 5 MB.', 'error')
    return redirect(url_for('upload'))


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=False)
