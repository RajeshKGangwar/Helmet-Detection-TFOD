import base64


def Decodetobase64(image, imageloc):
    img = base64.b64decode(image)

    with open("research/"+imageloc, 'wb') as f:
        f.write(img)
        f.close()

def encodeIntoBase64(ImagePath):
    with open(ImagePath, "rb") as f:
        return base64.b64encode(f.read())