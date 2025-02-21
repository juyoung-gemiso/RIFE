from app import config
from app.service.RIFE import RIFE


rife = RIFE(
    model_dir=config.rife_config.checkpoint, 
    output_base_path=config.rife_config.temp_dir
)
