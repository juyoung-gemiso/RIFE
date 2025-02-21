from flask import Flask
from flask_restx import Api
from app.config import AppConfig
from app.logger import setup_logger


config = AppConfig()
logger = setup_logger(config)

logger.info("--------------------------------") 
logger.info("***********start server***********")
app = Flask(__name__)
api = Api(app, version='1.0', title='API 문서', description='Swagger 문서', doc="/api-docs")

from app.routes.frame_interpolation import frame_interpolation_api

api.add_namespace(frame_interpolation_api)
