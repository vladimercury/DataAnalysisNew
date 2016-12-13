from PIL import Image


def draw_image(size, data, name):
    Image.frombytes('L', size, data).save('img/' + name)


def draw_image_from_array(size, data, name):
    image = Image.frombytes('L', size, bytes(data))
    image.save('img/' + name)
