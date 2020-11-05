### Prepare environment
1.Create virtual environment 
```
virtualenv venv
```

2. Activate virtual environment (in CMD)
```
.\venv\Scripts\Activate
```
3. Install libraries 
```
pip install -r requirements.txt
```

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command from command prompt -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://127.0.0.1:5000/ (or) http://localhost:5000

You should be able to view the homepage.
