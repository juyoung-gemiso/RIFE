from app import logger
from flask_restx import Resource
from app.service.load_model import *
from flask import request, jsonify, make_response
from app.utils import valid_or_return_request
from app.routes.swagger_docs.frame_interpolation_docs import *
from app.routes.schemas.frame_interpolation_schemas import *


frame_interpolation_request_schema = FrameInterpolationRequestSchema()
@frame_interpolation_api.route('/frame_interpolation', methods=['POST'])
class FrameInterpolation(Resource):
    @frame_interpolation_doc
    @logger.catch(level="INFO")
    def post(self):
        request_data = request.json
        
        if request_data is None:
            error_response = {
                "code": 400,
                "error": "Bad Request"
            }
            logger.info(f"Bad Request / return code: 400")
            return make_response(jsonify(error_response), 400)
    
        try:
            request_json = valid_or_return_request(request_data, frame_interpolation_request_schema)
        except Exception as e:
            error_response = {
                "code": 400,
                "error": str(e)
            }
            return make_response(jsonify(error_response), 400)
        
        try:
            output_video_path = rife.run(**request_json)
            return make_response(jsonify({
                "output_video_path": output_video_path
            }), 200)
        except Exception as e:
            logger.error(f"error: {e}")
            video_path = request_data["video_path"]
            return make_response(jsonify({
                "video_path": video_path,
                "error_message": str(e)
            }), 500)
