import sys
from PIL import Image
#import math
input_arg=sys.argv
arg1=input_arg[1]
im = Image.open(arg1)
width, height = im.size
     
for y in range(height):
    for x in range(width):
        rgb = im.getpixel((x,y))
        rgb = (rgb[0]//2,  # R
               rgb[1]//2,  # G
               rgb[2]//2,  # B
               )
        im.putpixel((x,y), rgb)
     
im.save('Q2.png')
