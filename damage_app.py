import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory, make_response
from werkzeug.utils import secure_filename
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from damage_predictor import DamagePredictor
import pdfkit
import time
import io
from datetime import datetime

# Create Flask app
app = Flask(__name__)
app.secret_key = 'car_damage_assessment_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Initialize the damage predictor
predictor = DamagePredictor(model_path="models/fast_model.h5", currency="INR")

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the home page"""
    return render_template('damage_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Process the uploaded file and make a prediction"""
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user didn't select a file
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Get car information from form
    car_value_inr = float(request.form.get('car_value', 1875000))
    # Convert INR to USD (since predictor expects USD)
    car_value_usd = car_value_inr / 75.0
    
    car_make = request.form.get('car_make', '')
    car_model = request.form.get('car_model', '')
    car_year_str = request.form.get('car_year', '')
    car_year = int(car_year_str) if car_year_str.isdigit() else None
    
    # Create car info dictionary
    car_info = None
    if car_make or car_model or car_year:
        car_info = {}
        if car_make:
            car_info["make"] = car_make
        if car_model:
            car_info["model"] = car_model
        if car_year:
            car_info["year"] = car_year
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the image with the predictor (passing USD value)
            result = predictor.process_image(file_path, car_value=car_value_usd, car_info=car_info)
            
            # Save the visualization
            vis_filename = f"result_{filename}"
            vis_path = os.path.join(app.config['RESULTS_FOLDER'], vis_filename)
            
            if "visualization" in result["damage_assessment"]:
                vis = result["damage_assessment"]["visualization"]
                cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                
                # Remove visualization from JSON
                result_copy = result.copy()
                if "visualization" in result_copy["damage_assessment"]:
                    del result_copy["damage_assessment"]["visualization"]
                
                # Save JSON result
                json_filename = f"result_{os.path.splitext(filename)[0]}.json"
                json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)
                
                with open(json_path, 'w') as f:
                    json.dump(result_copy, f, indent=2)
                
                # Redirect to results page
                return redirect(url_for('result', filename=filename))
            else:
                flash('Error: Could not generate visualization')
                return redirect(url_for('index'))
            
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg, gif')
        return redirect(url_for('index'))

@app.route('/result/<filename>')
def result(filename):
    """Display the assessment result"""
    # Get paths to result files
    vis_filename = f"result_{filename}"
    vis_path = os.path.join(app.config['RESULTS_FOLDER'], vis_filename)
    
    json_filename = f"result_{os.path.splitext(filename)[0]}.json"
    json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)
    
    # Check if files exist
    if not os.path.exists(vis_path) or not os.path.exists(json_path):
        flash('Result not found')
        return redirect(url_for('index'))
    
    # Load assessment data
    with open(json_path, 'r') as f:
        assessment = json.load(f)
    
    # Get the original uploaded image
    uploaded_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    return render_template(
        'damage_result.html',
        vis_image=f"../results/{vis_filename}",
        uploaded_image=f"../uploads/{filename}",
        assessment=assessment,
        filename=filename
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded file"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve the result file"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/generate_pdf/<filename>')
def generate_pdf(filename):
    """Generate a PDF report for the given result"""
    # Get paths to result files
    json_filename = f"result_{os.path.splitext(filename)[0]}.json"
    json_path = os.path.join(app.config['RESULTS_FOLDER'], json_filename)
    
    # Check if files exist
    if not os.path.exists(json_path):
        flash('Result not found')
        return redirect(url_for('index'))
    
    # Load assessment data
    with open(json_path, 'r') as f:
        assessment = json.load(f)
    
    # Generate PDF content
    pdf_content = generate_pdf_content(assessment)
    
    # Create response with PDF content
    response = make_response(pdf_content)
    
    # Determine content type based on content (default to PDF)
    content_type = 'application/pdf'
    file_ext = 'pdf'
    
    # If content is text (fallback method), serve as text file
    if pdf_content.startswith(b'Car Damage Assessment Report') or pdf_content.startswith(b'\n            Car Damage'):
        content_type = 'text/plain'
        file_ext = 'txt'
    
    # Set response headers
    response.headers['Content-Type'] = content_type
    response.headers['Content-Disposition'] = f'attachment; filename=damage_report_{os.path.splitext(filename)[0]}.{file_ext}'
    
    return response

def generate_pdf_content(assessment):
    """Generate PDF content based on the assessment data"""
    # Create HTML for the PDF
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Car Damage Assessment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .report-section {{ margin-bottom: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .damage-normal {{ color: green; font-weight: bold; }}
            .damage-minor {{ color: #FFC107; font-weight: bold; }}
            .damage-moderate {{ color: #FF9800; font-weight: bold; }}
            .damage-severe {{ color: #F44336; font-weight: bold; }}
            .cost {{ font-size: 18px; color: #e74c3c; font-weight: bold; }}
            .timestamp {{ font-size: 12px; color: #7f8c8d; text-align: right; margin-top: 30px; }}
            .logo {{ text-align: center; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Car Damage Assessment Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="report-section">
            <h2>Car Details</h2>
            <table>
                <tr>
                    <th>Make/Model/Year:</th>
                    <td>
                        {f"{assessment['car_info']['year']} " if assessment['car_info'] and 'year' in assessment['car_info'] else ""}
                        {f"{assessment['car_info']['make']} " if assessment['car_info'] and 'make' in assessment['car_info'] else ""}
                        {f"{assessment['car_info']['model']}" if assessment['car_info'] and 'model' in assessment['car_info'] else "Not specified"}
                    </td>
                </tr>
                <tr>
                    <th>Car Value:</th>
                    <td>INR {assessment['car_value_inr']:.2f} (${assessment['car_value']:.2f})</td>
                </tr>
            </table>
        </div>
        
        <div class="report-section">
            <h2>Damage Assessment</h2>
            <table>
                <tr>
                    <th>Damage Level:</th>
                    <td class="{
                        'damage-normal' if assessment['damage_assessment']['predicted_class'] == 'normal'
                        else 'damage-minor' if assessment['damage_assessment']['predicted_class'] == 'minor damaged car'
                        else 'damage-moderate' if assessment['damage_assessment']['predicted_class'] == 'moderate car damaged'
                        else 'damage-severe'
                    }">
                        {assessment['damage_assessment']['predicted_class']}
                    </td>
                </tr>
                <tr>
                    <th>Confidence:</th>
                    <td>{assessment['damage_assessment']['confidence'] * 100:.1f}%</td>
                </tr>
                <tr>
                    <th>Damage Multiplier:</th>
                    <td>{assessment['damage_assessment']['damage_multiplier']}</td>
                </tr>
            </table>
        </div>
        
        <div class="report-section">
            <h2>Cost Estimate</h2>
            <table>
                <tr>
                    <th>Base Repair Cost:</th>
                    <td>INR {assessment['base_repair_cost_inr']:.2f} (${assessment['base_repair_cost']:.2f})</td>
                </tr>
                <tr>
                    <th>Cost Multiplier:</th>
                    <td>{assessment['cost_multiplier']}</td>
                </tr>
                <tr>
                    <th>Estimated Repair Cost:</th>
                    <td class="cost">INR {assessment['estimated_repair_cost_inr']:.2f} (${assessment['estimated_repair_cost']:.2f})</td>
                </tr>
            </table>
        </div>
        
        <div class="timestamp">
            <p>Assessment ID: {hash(str(assessment))}</p>
            <p>This is an automated assessment generated by the Car Damage AI system.</p>
        </div>
    </body>
    </html>
    """
    
    # Generate PDF using pdfkit
    try:
        # Options for PDF generation
        options = {
            'page-size': 'A4',
            'margin-top': '15mm',
            'margin-right': '15mm',
            'margin-bottom': '15mm',
            'margin-left': '15mm',
            'encoding': 'UTF-8',
        }
        
        # Generate PDF directly to bytes
        pdf_content = pdfkit.from_string(html_content, False, options=options)
        return pdf_content
    except Exception as e:
        print(f"Error generating PDF with pdfkit: {str(e)}")
        
        # Fall back to a simple text report if pdfkit fails
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from io import BytesIO
            
            # Create a PDF using ReportLab
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Title
            title_style = styles['Heading1']
            title_style.alignment = 1  # Center alignment
            elements.append(Paragraph("Car Damage Assessment Report", title_style))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Car Details
            elements.append(Paragraph("Car Details", styles['Heading2']))
            car_details = []
            car_details.append(["Make/Model/Year:", 
                              f"{assessment['car_info']['year']} {assessment['car_info']['make']} {assessment['car_info']['model']}" 
                              if assessment['car_info'] and 'year' in assessment['car_info'] and 'make' in assessment['car_info'] and 'model' in assessment['car_info']
                              else "Not specified"])
            car_details.append(["Car Value:", f"INR {assessment['car_value_inr']:.2f} (${assessment['car_value']:.2f})"])
            
            car_table = Table(car_details, colWidths=[120, 350])
            car_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(car_table)
            elements.append(Spacer(1, 20))
            
            # Damage Assessment
            elements.append(Paragraph("Damage Assessment", styles['Heading2']))
            damage_details = []
            damage_details.append(["Damage Level:", assessment['damage_assessment']['predicted_class']])
            damage_details.append(["Confidence:", f"{assessment['damage_assessment']['confidence'] * 100:.1f}%"])
            damage_details.append(["Damage Multiplier:", str(assessment['damage_assessment']['damage_multiplier'])])
            
            damage_table = Table(damage_details, colWidths=[120, 350])
            damage_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(damage_table)
            elements.append(Spacer(1, 20))
            
            # Cost Estimate
            elements.append(Paragraph("Cost Estimate", styles['Heading2']))
            cost_details = []
            cost_details.append(["Base Repair Cost:", f"INR {assessment['base_repair_cost_inr']:.2f} (${assessment['base_repair_cost']:.2f})"])
            cost_details.append(["Cost Multiplier:", str(assessment['cost_multiplier'])])
            cost_details.append(["Estimated Repair Cost:", f"INR {assessment['estimated_repair_cost_inr']:.2f} (${assessment['estimated_repair_cost']:.2f})"])
            
            cost_table = Table(cost_details, colWidths=[120, 350])
            cost_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(cost_table)
            
            # Footer
            elements.append(Spacer(1, 30))
            elements.append(Paragraph(f"Assessment ID: {hash(str(assessment))}", styles['Normal']))
            elements.append(Paragraph("This is an automated assessment generated by the Car Damage AI system.", styles['Normal']))
            
            # Build PDF
            doc.build(elements)
            return buffer.getvalue()
        except Exception as inner_e:
            print(f"Error generating PDF with reportlab: {str(inner_e)}")
            # If all else fails, return a simple text file
            text_content = f"""
            Car Damage Assessment Report
            Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            
            CAR DETAILS
            Make/Model/Year: {f"{assessment['car_info']['year']} {assessment['car_info']['make']} {assessment['car_info']['model']}" if assessment['car_info'] and 'year' in assessment['car_info'] and 'make' in assessment['car_info'] and 'model' in assessment['car_info'] else "Not specified"}
            Car Value: INR {assessment['car_value_inr']:.2f} (${assessment['car_value']:.2f})
            
            DAMAGE ASSESSMENT
            Damage Level: {assessment['damage_assessment']['predicted_class']}
            Confidence: {assessment['damage_assessment']['confidence'] * 100:.1f}%
            Damage Multiplier: {assessment['damage_assessment']['damage_multiplier']}
            
            COST ESTIMATE
            Base Repair Cost: INR {assessment['base_repair_cost_inr']:.2f} (${assessment['base_repair_cost']:.2f})
            Cost Multiplier: {assessment['cost_multiplier']}
            Estimated Repair Cost: INR {assessment['estimated_repair_cost_inr']:.2f} (${assessment['estimated_repair_cost']:.2f})
            
            Assessment ID: {hash(str(assessment))}
            This is an automated assessment generated by the Car Damage AI system.
            """
            return text_content.encode('utf-8')

if __name__ == '__main__':
    # Create template directory and simple templates
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Car Damage Assessment</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                padding-top: 2rem;
                background-color: #f8f9fa;
            }
            .card {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .header-image {
                max-height: 200px;
                object-fit: cover;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-md-8">
                    <div class="card mb-4">
                        <div class="card-header bg-primary text-white text-center">
                            <h2>Car Damage Assessment</h2>
                        </div>
                        <div class="card-body">
                            {% with messages = get_flashed_messages() %}
                                {% if messages %}
                                    <div class="alert alert-danger">
                                        <ul class="mb-0">
                                            {% for message in messages %}
                                                <li>{{ message }}</li>
                                            {% endfor %}
                                        </ul>
                                    </div>
                                {% endif %}
                            {% endwith %}
                            
                            <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data" class="mb-4">
                                <div class="mb-3">
                                    <label for="file" class="form-label">Select a car damage image:</label>
                                    <input type="file" class="form-control" id="file" name="file" accept="image/*" required>
                                </div>
                                
                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <label for="car_make" class="form-label">Car Make:</label>
                                        <input type="text" class="form-control" id="car_make" name="car_make" placeholder="e.g., Toyota, BMW">
                                    </div>
                                    <div class="col-md-6">
                                        <label for="car_model" class="form-label">Car Model:</label>
                                        <input type="text" class="form-control" id="car_model" name="car_model" placeholder="e.g., Camry, 3-Series">
                                    </div>
                                </div>
                                
                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <label for="car_year" class="form-label">Car Year:</label>
                                        <input type="number" class="form-control" id="car_year" name="car_year" placeholder="e.g., 2020" min="1900" max="2099">
                                    </div>
                                    <div class="col-md-6">
                                        <label for="car_value" class="form-label">Car Value (INR):</label>
                                        <input type="number" class="form-control" id="car_value" name="car_value" value="1875000" min="75000" max="75000000">
                                        <small class="form-text text-muted">Approx. value in INR</small>
                                    </div>
                                </div>
                                
                                <div class="d-grid">
                                    <button type="submit" class="btn btn-primary btn-lg">Analyze Damage</button>
                                </div>
                            </form>
                            
                            <div class="mt-4">
                                <h5>How it works:</h5>
                                <ol>
                                    <li>Upload a photo of the car damage</li>
                                    <li>Enter car details (make, model, year, value)</li>
                                    <li>Get instant damage assessment and repair cost estimate</li>
                                </ol>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    """
    
    # Create result.html template
    result_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Damage Assessment Result</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                padding-top: 2rem;
                background-color: #f8f9fa;
            }
            .card {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .result-image {
                max-height: 400px;
                object-fit: contain;
            }
            .damage-normal {
                color: green;
                font-weight: bold;
            }
            .damage-minor {
                color: #FFC107;
                font-weight: bold;
            }
            .damage-moderate {
                color: #FF9800;
                font-weight: bold;
            }
            .damage-severe {
                color: #F44336;
                font-weight: bold;
            }
            .usd-price {
                font-size: 0.8em;
                color: #6c757d;
            }
            .btn-pdf {
                background-color: #D32F2F;
                color: white;
            }
            .btn-pdf:hover {
                background-color: #B71C1C;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-md-10">
                    <div class="card mb-4">
                        <div class="card-header bg-primary text-white text-center">
                            <h2>Damage Assessment Result</h2>
                        </div>
                        <div class="card-body">
                            <div class="row mb-4">
                                <div class="col-12">
                                    <div class="text-center">
                                        <img src="{{ vis_image }}" class="img-fluid rounded result-image" alt="Assessment Result">
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="card mb-4">
                                        <div class="card-header bg-info text-white">
                                            <h4 class="mb-0">Damage Assessment</h4>
                                        </div>
                                        <div class="card-body">
                                            <table class="table">
                                                <tr>
                                                    <th>Damage Level:</th>
                                                    <td class="
                                                        {% if assessment.damage_assessment.predicted_class == 'normal' %}damage-normal
                                                        {% elif assessment.damage_assessment.predicted_class == 'minor damaged car' %}damage-minor
                                                        {% elif assessment.damage_assessment.predicted_class == 'moderate car damaged' %}damage-moderate
                                                        {% else %}damage-severe{% endif %}
                                                    ">
                                                        {{ assessment.damage_assessment.predicted_class }}
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <th>Confidence:</th>
                                                    <td>{{ (assessment.damage_assessment.confidence * 100) | round(1) }}%</td>
                                                </tr>
                                                <tr>
                                                    <th>Damage Multiplier:</th>
                                                    <td>{{ assessment.damage_assessment.damage_multiplier }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Other Probabilities:</th>
                                                    <td>
                                                        <ul class="list-unstyled">
                                                            {% for class_name, prob in assessment.damage_assessment.class_probabilities.items() %}
                                                                {% if class_name != assessment.damage_assessment.predicted_class %}
                                                                    <li>{{ class_name }}: {{ (prob * 100) | round(1) }}%</li>
                                                                {% endif %}
                                                            {% endfor %}
                                                        </ul>
                                                    </td>
                                                </tr>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="col-md-6">
                                    <div class="card mb-4">
                                        <div class="card-header bg-success text-white">
                                            <h4 class="mb-0">Cost Estimate</h4>
                                        </div>
                                        <div class="card-body">
                                            <table class="table">
                                                <tr>
                                                    <th>Car Details:</th>
                                                    <td>
                                                        {% if assessment.car_info %}
                                                            {% if assessment.car_info.year %}{{ assessment.car_info.year }} {% endif %}
                                                            {% if assessment.car_info.make %}{{ assessment.car_info.make }} {% endif %}
                                                            {% if assessment.car_info.model %}{{ assessment.car_info.model }}{% endif %}
                                                        {% else %}
                                                            Not specified
                                                        {% endif %}
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <th>Car Value:</th>
                                                    <td>
                                                        INR {{ assessment.car_value_inr | round(2) }}
                                                        <span class="usd-price">(${{ assessment.car_value | round(2) }})</span>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <th>Base Repair Cost:</th>
                                                    <td>
                                                        INR {{ assessment.base_repair_cost_inr | round(2) }}
                                                        <span class="usd-price">(${{ assessment.base_repair_cost | round(2) }})</span>
                                                    </td>
                                                </tr>
                                                <tr>
                                                    <th>Cost Multiplier:</th>
                                                    <td>{{ assessment.cost_multiplier }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Estimated Repair Cost:</th>
                                                    <td class="h4 text-danger">
                                                        INR {{ assessment.estimated_repair_cost_inr | round(2) }}
                                                        <div class="usd-price">(${{ assessment.estimated_repair_cost | round(2) }})</div>
                                                    </td>
                                                </tr>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="text-center mt-3">
                                <a href="{{ url_for('index') }}" class="btn btn-primary">Analyze Another Image</a>
                                <a href="{{ url_for('generate_pdf', filename=filename) }}" class="btn btn-pdf ms-2">
                                    <i class="bi bi-file-pdf"></i> Download PDF Report
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    </body>
    </html>
    """
    
    # Write templates to files
    with open(os.path.join('templates', 'damage_index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    with open(os.path.join('templates', 'damage_result.html'), 'w', encoding='utf-8') as f:
        f.write(result_html)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000) 