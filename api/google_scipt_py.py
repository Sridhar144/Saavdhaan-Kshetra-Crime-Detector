import io
import os
from numpy import random
from google.cloud import vision_v1
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import cv2
import numpy as np
from .Pillow_Utility import draw_borders, Image

def object_detection(np_image):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    credential_path = os.path.join(current_directory, "peppy-citron-421415-d6e0bc494625.json")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    client = vision_v1.ImageAnnotatorClient()

    _, encoded_image = cv2.imencode('.jpg', np_image)
    content = encoded_image.tobytes()
    
    image = vision_v1.types.Image(content=content)
    response = client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations

    pillow_image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))
    df = pd.DataFrame(columns=['name', 'score'])
    for obj in localized_object_annotations:
        df = df._append(
            dict(
                name=obj.name,
                score=obj.score
            ),
            ignore_index=True)
        
        r, g, b = random.randint(150, 255), random.randint(
            150, 255), random.randint(150, 255)

        draw_borders(pillow_image, obj.bounding_poly, (r, g, b),
                    pillow_image.size, obj.name, obj.score)

    print(df)
    # pillow_image.show()
    return df, pillow_image
