from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import main
import trainer as tr
import webbrowser
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)

    print
    form.errors
    if request.method == 'POST':
        url = request.form['name']
        main.process_test_url(url, 'gui_url_features.csv')
        return_ans = tr.gui_caller('url_features.csv', 'gui_url_features.csv')
        a = str(return_ans).split()
        val = int(a[1])

        if form.validate():
            # Save the comment here.
            if val == 0 or val == 2:
                flash('URL  ' + url + '  is Safe')
                webbrowser.get("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s").open(url)
            elif val == 1:
                flash('URL  ' + url + '  is Malicious')
            else:
                flash('URL  ' + url + '  is Malware')
        else:
            flash('Error: All the form fields are required. ')

    return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run()