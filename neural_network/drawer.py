def draw_image(size, data):
    from PIL import Image, ImageDraw
    image = Image.frombytes('L', size, data)
    image.save('a.png')
