from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

# city mapping
cities = {
    "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946),
    "Delhi": (28.7041, 77.1025),
    "Mumbai": (19.0760, 72.8777),
    "Hyderabad": (17.3850, 78.4867),
    "Palakkad": (10.731769, 76.703273)
}

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/residential', methods=['GET', 'POST'])
def residential():
    mode = request.args.get('mode', 'custom')  # or request.form.get('mode', 'custom') in POST
    output = ""

    if request.method == 'POST':
        city = request.form.get("city")
        mode = request.form.get('mode', mode)  # preserve POST mode

        if city not in cities:
            return render_template("residential.html", output="Invalid city", cities=cities.keys(), mode=mode)

        lat, lon = cities[city]

        script_map = {
            'basic': 'selecthousebasic.py',
            'custom': 'selecthouse.py'
        }
        script = script_map.get(mode, 'selecthouse.py')

        try:
            result = subprocess.run(
                ["python3", script, str(lat), str(lon)],
                capture_output=True,
                text=True
            )
            output = result.stdout

        except Exception as e:
            output = str(e)

    return render_template("residential.html", output=output, cities=cities.keys(), mode=mode)

@app.route('/corporate', methods=['GET', 'POST'])
def corporate():
    mode = request.args.get('mode', 'custom')
    output = ""

    if request.method == 'POST':
        city = request.form.get("city")
        mode = request.form.get('mode', mode)

        if city not in cities:
            return render_template("corporate.html", output="Invalid city", cities=cities.keys(), mode=mode)

        lat, lon = cities[city]

        script_map = {
            'basic': 'corporatesimple.py',
            'custom': 'corporate.py'
        }
        script = script_map.get(mode, 'corporate.py')

        try:
            result = subprocess.run(
                ["python3", script, str(lat), str(lon)],
                capture_output=True,
                text=True
            )
            output = result.stdout

        except Exception as e:
            output = str(e)

    return render_template("corporate.html", output=output, cities=cities.keys(), mode=mode)

if __name__ == '__main__':
    app.run(debug=True)