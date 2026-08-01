from flask import Flask
import logging
from .routes import bp

def create_app(state):
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    
    # Desativa logs de requisição para não poluir o terminal
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Registra o blueprint de rotas
    app.register_blueprint(bp)
    
    return app
