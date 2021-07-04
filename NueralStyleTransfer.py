import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torchvision.utils import save_image
import cv2


###################################################### NST Architecture using VGG (taking features from only 5 layers) #############################
class VGG(nn.Module):
  def __init__(self):
    super(VGG,self).__init__()
    self.choosen_layers = ['0','5','10','19','28']
    self.Vgg = models.vgg19(pretrained=True).features
  
  def forward(self,x):
    features = []
    for layer_num,layer in enumerate(self.Vgg): 
      x = layer(x)
      if str(layer_num) in self.choosen_layers:
        features.append(x)
    return features

def load_image(img):
  image = Image.open(img)
  image = loader(image).unsqueeze(0) #we UNsqueeze her to add another dim for the batch size of images
  return image.to(device)

############################################################ Initialisation #############################################################################
device = torch.device("cuda" if torch.cuda.is_available else "cpu" )
image_size = 512
loader = transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()])
choosen_layers = ['0','5','10','19','28']
original_img =  load_image("/content/original_sam.jpeg")
style_img = load_image("/content/style3.jpg")
model = VGG().to(device).eval() #To freeze the weights

#generated_image = torch.randn(shape = original_img.shape,device = device,requires_grad =True)
generated_img = original_img.clone().requires_grad_(True)

############################################################ HyperParameters  #########################################################################################
total_steps = 5000
learning_rate = 0.001
alpha = 1 # for the content loss
beta = 0.01 # How much style we want in image
optimizer = optim.Adam([generated_img],lr = learning_rate)
choosen_layers = ['0','5','10','19','28']
############################################################### Generating Image ##################################################################
def NST(original_img,style_img,generated_img, model = model ,total_steps = total_steps):
    for step in range(total_steps):
    
        generated_features = model(generated_img)
        original_features = model(original_img)
        style_features = model(style_img)
        content_loss = 0
        style_loss = 0

        for original_feature,style_feature,generated_feature in zip(original_features,style_features,generated_features):
            batch_size,channel,height,width = generated_feature.shape
            content_loss = torch.mean((generated_feature - original_feature)**2)
            
            #Gram Matrices Computation
            # we multiply each pixel in a channel with all the other channels, the shape is (channel x channel)
            G = generated_feature.view(channel,height*width).mm(generated_feature.view(channel,height*width).t())
            S = style_feature.view(channel,height*width).mm(style_feature.view(channel,height*width).t())

            style_loss +=torch.mean((G  - S)**2)

    total_loss = alpha*content_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print("Steps :",step,"Total loss", total_loss,)
        save_image(generated_img,"GeneratedImage.png")
    return generated_img

Final_img =  NST(original_img,style_img,generated_img, model = model ,total_steps = total_steps))