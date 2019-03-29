from ppt import Presentation
import os

rootname = os.path.abspath(os.path.dirname(__file__))
dirname = os.path.join(rootname, 'images')
test_image = os.path.join(rootname, 'monty_truth.png')

ppt = Presentation(dirname,'testmonty.pptx')

ppt.add_image_slide(test_image,'Test')

ppt.save()