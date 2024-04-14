from flask import Flask

def create_app():
    # Create an instance of the Flask class
    app = Flask(__name__)

    # Load configuration settings from a file
    app.config.from_pyfile('config.py')

    # Import and register routes
    from .routes import api_bp
    app.register_blueprint(api_bp)

    # Additional initialization and setup can go here (e.g. connecting to a database)

    # Return the app instance
    return app
