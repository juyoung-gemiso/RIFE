from flask_restx import fields, Namespace

frame_interpolation_api = Namespace(name="프레임 보간 처리", path="/")
frame_interpolation_req = frame_interpolation_api.model("프레임 보간 요청", {
    "video": fields.String(required=True, description="보간 처리할 영상 경로"),
    "temp_dir": fields.String(default=".", required=False, description="추출한 프레임을 임시 저장할 디렉토리 경로"),
    "extension": fields.String(default="mxf", required=True, description="출력 영상의 확장자명"),
    "output_fps": fields.Float(default="29.97", required=True, description="출력 영상의 fps"),
})
frame_interpolation_res = frame_interpolation_api.model("프레임 보간 처리 결과", {
    "output_video_path": "[output video path]"
})


def frame_interpolation_doc(func):
    description = """인터레이스 영상의 프레임 보간을 수행합니다.

1. 인터레이스 -> 프로그레시브(e.g. 25i -> 50p) 프레임 추출
2. 프레임 보간(e.g. 50p -> 60p) 후 보간된 프레임은 오디오 없는 영상으로 저장
3. 보간된 영상과 원본 영상의 오디오 병합

**Request**
- video: 영상 파일 경로
- temp_dir: 추출한 프레임을 임시 저장할 디렉토리 경로 (지정하지 않으면 현재 코드가 있는 경로가 설정됨)
- extension: 출력 영상의 확장자명(e.g. mxf)
- output_fps: 출력 영상의 fps(e.g. 29.97)
"""
    @frame_interpolation_api.doc(description=description)
    @frame_interpolation_api.expect(frame_interpolation_req)
    @frame_interpolation_api.response(201, "프레임 보간 처리 성공", frame_interpolation_res)
    @frame_interpolation_api.response(400, "잘못된 요청")
    @frame_interpolation_api.response(500, "서버 오류")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper