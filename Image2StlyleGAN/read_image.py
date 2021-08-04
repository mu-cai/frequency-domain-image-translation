import PIL 
from torchvision import transforms
from PIL import Image 

def image_reader(img_path,resize=None, crop_size=None):

    with open(img_path,"rb") as f: 
        image=Image.open(f)
        image=image.convert("RGB")
    if resize!=None:
        image=image.resize((resize,resize))
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    if crop_size!=None:
        crop_size =min(image.size)
        # print(crop_size)
        transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(1024),
        transforms.ToTensor()
        ])

    image = transform(image)


    image=image.unsqueeze(0)

    return image
