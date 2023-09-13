from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image

# Init coco object
coco = Coco()

# Add categories
coco.add_category(CocoCategory(id=0, name='pole'))

# Create coco image
id = str(1652168831032115)
filename = f"data/images/{id}.jpg"
height, width = Image.open(filename).size
coco_image = CocoImage(file_name=filename, height=height, width=width)

coco_image.add_annotation(    
    CocoAnnotation(    
        bbox=[10, 20, width, height],    
        category_id=0,    
        category_name='human'    
    )
)

coco.add_image(coco_image)

save_json(data=coco.json, save_path="coco.json")
