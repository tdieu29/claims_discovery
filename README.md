## Usage

1. Create and activate virtual environment

`python -m venv venv`

`source venv/Scripts/activate`

`python -m pip install --upgrade pip setuptools wheel`

2. Install requirements

`pip install -r requirements.txt`

3. Create top level `.env` file and add `FLASK_ENV=development`

4. Run Flask app

`flask run`
