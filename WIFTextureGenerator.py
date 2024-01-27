from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from pyweaving import Draft, instructions
from pyweaving.wif import WIFReader, WIFWriter
from math import asin, cos, floor, sin

import os.path

from PIL import Image, ImageDraw, ImageFont
from pyweaving import WarpThread, WeftThread
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
__here__ = os.path.dirname(__file__)
font_path = os.path.join(__here__, 'pyweaving/data', 'Arial.ttf')



psi = 0.523
psi = 0.0
# u_max = 0.523 ### pi/6  30 deg
# u_max = 0.0314 ### pi/6  30 deg
u_max = 0.314 ### pi/6  30 deg
# u_max = 0.5 * 2
sin_u_max = sin(u_max)

sin_psi = sin(psi)
cos_psi = cos(psi)
pixel_scale = 4


class TextureGenerator(object):
    # TODO:
    # - Add a "drawndown only" option
    # - Add a default tag (like a small delta symbol) to signal the initial
    # shuttle direction
    # - Add option to render the backside of the fabric
    # - Add option to render a bar graph of the thread crossings along the
    # sides
    # - Add option to render 'stats table'
    #   - Number of warp threads
    #   - Number of weft threads
    #   - Number of harnesses/shafts
    #   - Number of treadles
    #   - Warp unit size / reps
    #   - Weft unit size / reps
    #   - Longest warp float
    #   - Longest weft float
    #   - Selvedge continuity
    # - Add option to rotate orientation
    # - Add option to render selvedge continuity
    # - Add option to render inset "scale view" rendering of fabric
    # - Add option to change thread spacing
    # - Support variable thickness threads
    # - Add option to render heddle count on each shaft
    def __init__(self, draft, liftplan=None, margin_pixels=20, scale=20,
                 foreground=(127, 127, 127), background=(255, 255, 255),
                 markers=(0, 0, 0), numbering=(200, 0, 0)):
        self.draft = draft

        self.liftplan = liftplan

        self.margin_pixels = margin_pixels
        self.pixels_per_square = scale

        self.background = background
        self.foreground = foreground
        self.markers = markers
        self.numbering = numbering

        self.font_size = int(round(scale * 1.2))

        self.font = ImageFont.truetype(font_path, self.font_size)

        self.output_texture_name = "out"
        self.output_normal_name = "normal_map"
        self.output_tangent_name = "tangent_map"
    
    def set_output_name(self, texName, texNormalName, texTangentName):
        self.output_texture_name = texName
        self.output_normal_name = texNormalName
        self.output_tangent_name = texTangentName

    def make_pil_image(self):
        width_squares = len(self.draft.warp)
        height_squares = len(self.draft.weft)

        # XXX Not totally sure why the +1 is needed here, but otherwise the
        # contents overflows the canvas
        width = (width_squares * self.pixels_per_square)
        height = (height_squares * self.pixels_per_square)

        im = Image.new('RGB', (width, height), self.background)
        self.normal_map = Image.new('RGB', (width, height), self.background)
        self.tangent_map = Image.new('RGB', (width, height), self.background)

        draw = ImageDraw.Draw(im) 

        self.paint_drawdown(draw)
        del draw 
        im.save(self.output_texture_name+".png")
        return im 

     
    def normal_to_rgb_int(self, normal):
        pix_normal_r = floor(127.5*(normal[0]+1))
        pix_normal_g = floor(127.5*(normal[1]+1))
        pix_normal_b = floor(127.5*(normal[2]+1))
        pix_normal_rgb = (pix_normal_r, pix_normal_g, pix_normal_b)
        return pix_normal_rgb

    def normal_to_rgb_gamma_correct(self, normal):
        pix_normal_r = ((normal[0]+1)*0.5)** ( 2.2)
        pix_normal_g = ((normal[1]+1)*0.5)** ( 2.2)
        pix_normal_b = ((normal[2]+1)*0.5)** ( 2.2) 
        pix_normal_rgb = (pix_normal_r, pix_normal_g, pix_normal_b)
        return pix_normal_rgb
 

    def paint_drawdown(self, draw):
        offsety = 0
        floats = self.draft.compute_floats()

        img_size = (self.normal_map.size[1],self.normal_map.size[0], 3)
        normal_mat = np.zeros(img_size)
        tangent_mat = np.zeros(img_size)

        normal_mat_float = np.zeros(img_size)
        tangent_mat_float = np.zeros(img_size)

        for start, end, visible, length, thread in floats:
            if visible:
                startx = start[0] * self.pixels_per_square
                starty = (start[1] * self.pixels_per_square) + offsety
                endx = (end[0] + 1) * self.pixels_per_square
                endy = ((end[1] + 1) * self.pixels_per_square) + offsety

                draw.rectangle((startx, starty, endx, endy),
                               outline=self.foreground,
                               fill=thread.color.rgb)

                center_x = (startx + endx)/2
                center_y = (starty + endy)/2                

                for x in range(startx, endx):
                    for y in range(starty, endy):
                        pix_x = x + 0.5
                        pix_y = y + 0.5
                        ratio_y_l = 2*(center_y-pix_y)/(endy-starty)
                        ratio_x_w = 2*(pix_x-center_x)/(endx-startx)
                        
                        u = asin(ratio_y_l * sin_u_max)
                        v = asin(ratio_x_w)
                        if isinstance(thread, WeftThread):
                            v = asin(ratio_x_w * sin_u_max)
                            u = asin(ratio_y_l)

                        sin_u = sin(u); sin_v = sin(v); cos_u = cos(u); cos_v = cos(v)
                        pix_normal = (sin_v, sin_u*cos_v, cos_u*cos_v)
                        pix_tangent = (
                            -cos_v*sin_psi, 
                            cos_u*cos_psi + sin_u*sin_v*sin_psi,
                            -sin_u*cos_psi + cos_u*sin_v*sin_psi

                        )
                        
                        pix_normal_rgb = self.normal_to_rgb_int(pix_normal)
                        pix_tangent_rgb = self.normal_to_rgb_int(pix_tangent) 

                        pix_normal_rgb_f = self.normal_to_rgb_gamma_correct(pix_normal)
                        pix_tangent_rgb_f = self.normal_to_rgb_gamma_correct(pix_tangent) 
                        
                        normal_mat[y,x,2] = pix_normal_rgb[0]
                        normal_mat[y,x,1] = pix_normal_rgb[1]
                        normal_mat[y,x,0] = pix_normal_rgb[2]
                        tangent_mat[y,x,2] = pix_tangent_rgb[0]
                        tangent_mat[y,x,1] = pix_tangent_rgb[1]
                        tangent_mat[y,x,0] = pix_tangent_rgb[2]

                        normal_mat_float[y,x,2] = pix_normal_rgb_f[0]
                        normal_mat_float[y,x,1] = pix_normal_rgb_f[1]
                        normal_mat_float[y,x,0] = pix_normal_rgb_f[2]
                        tangent_mat_float[y,x,2] = pix_tangent_rgb_f[0]
                        tangent_mat_float[y,x,1] = pix_tangent_rgb_f[1]
                        tangent_mat_float[y,x,0] = pix_tangent_rgb_f[2]
                        
        
        cv2.imwrite(self.output_normal_name+".png", normal_mat.astype(np.float32))
        cv2.imwrite(self.output_normal_name+".exr", normal_mat_float.astype(np.float32))

        cv2.imwrite(self.output_tangent_name+".png", tangent_mat.astype(np.float32))
        cv2.imwrite(self.output_tangent_name+".exr", tangent_mat_float.astype(np.float32))
        
    def set_pixel_resolution(self, value):
        self.pixels_per_square = value





def load_draft(infile):
    if infile.endswith('.wif'):
        return WIFReader(infile).read()
    elif infile.endswith('.json'):
        with open(infile) as f:
            return Draft.from_json(f.read())
    else:
        raise ValueError(
            "filename %r unrecognized: .wif and .json are supported" %
            infile)



def genTexture(file):
    draft = load_draft(file)
    ir = TextureGenerator(draft)
    ir.set_pixel_resolution(pixel_scale)
    ir.set_output_name("out", "normal", "tangent")
    ir.make_pil_image()

# genTexture("05sH010_linen.wif")
genTexture("123.wif")