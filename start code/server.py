from flask import Flask
from flask_login import LoginManager
from src.AuthenticationManager import AuthenticationManager
from src.client import bootstrap_system

app = Flask(__name__)
app.secret_key = 'very-secret-123'  # Used to add entropy


# Authentication manager and System setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

auth_manager = AuthenticationManager(login_manager)
system = bootstrap_system(auth_manager)

@login_manager.user_loader
def load_user(user_id):
    return system.get_user_by_id(user_id)