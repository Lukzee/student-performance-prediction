from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("student_performance.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    gpa = None
    if request.method == "POST":
        try:
            attendance = float(request.form["attendance"])
            hours_studied = float(request.form["hours_studied"])
            previous_grade = float(request.form["previous_grade"])
            assignments_completed = float(request.form["assignments_completed"])
            extra_curricular = float(request.form["extra_curricular"])

            # Prediction
            final_grade = model.predict([[attendance, hours_studied, previous_grade,
                                          assignments_completed, extra_curricular]])[0]

            # Convert to GPA (0-4 scale)
            gpa = round((final_grade / 100) * 4, 2)
            prediction = round(final_grade, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, gpa=gpa)

if __name__ == "__main__":
    app.run(debug=True)
