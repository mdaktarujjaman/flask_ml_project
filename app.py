import pandas as pd
import joblib
from flask import Flask , render_template, url_for
from forms import ImputForm



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
model = joblib.load("model.joblib")


@app.route('/')
def home():
    return render_template('home.html', title='Home')

@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/contact")
def contact():
    return render_template('contact.html', title='Contact')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = ImputForm()
    if form.validate_on_submit():
        # Here you would typically process the form data and make predictions
        x_new = pd.DataFrame(dict(
            airline=[form.airline.data],
            date_of_journey=[form.date_of_journey.data.strftime("%Y-%m-%d")],
            source=[form.source.data],
            destination=[form.destination.data],
            dep_time=[form.dep_time.data.strftime("%H:%M:%S")],
            arrival_time=[form.arrival_time.data.strftime("%H:%M:%S")],
            duration=[form.duration.data],
            total_stops=[form.total_stops.data],
            additional_info=[form.additional_info.data]
        ))
        prediction = model.predict(x_new)[0]
        message = f"The predicted price is {prediction:,.0f} INR!"
    else:
        message = "Please fill out the form to get a prediction."
        
    return render_template('predict.html', title='Predict', form=form, output=message)


if __name__ == '__main__':
    app.run(debug=True, port=5001)