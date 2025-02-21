from marshmallow import Schema, fields, validate

class FrameInterpolationRequestSchema(Schema):
    video = fields.Str(required=True, validate=validate.Length(min=1))
    temp_dir = fields.Str(required=False, validate=validate.Length(min=1), default=".")
    extension = fields.Str(required=True, validate=validate.Length(min=1), default="mxf")
    output_fps = fields.Float(required=True, validate=validate.Range(min=20))
