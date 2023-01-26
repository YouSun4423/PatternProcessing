from torch.nn import functional
from torchvision import models, transforms
from PIL import Image
import json

# クラス一覧を読み込む。
with open("datas/imagenet_class_index.json") as f:
    label = json.load(f)
    class_names = [x[1] for x in label.values()]

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
img = Image.open("images/Mandrill.bmp")

preprocess = transforms.Compose([
    transforms.Resize(224), 
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_p = preprocess(img)
img_b = img_p[None]
model.eval()
result = model(img_b)

idx = functional.softmax(result, dim=1)
idx_probs, idx_indices = idx.sort(dim=1, descending=True)

for probs, indices in zip(idx_probs, idx_indices):
    for k in range(5):
        print(f"Top{k + 1} : {class_names[indices[k]]} {probs[k]:.2%}")

