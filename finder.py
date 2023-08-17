from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
import cv2
import torchvision.transforms as transforms
import glob
workers = 0 if os.name == 'nt' else 4
GLOBAL_I = 0
IMAGE_DIR = 'images/finder/'
REFERENCE= 'references'
TARGET = 'targets'
OUT = 'finder.png'
def collate_fn(x):
    return x[0]
def tensor_to_image(torch_image,is_negone_to_one=True):
    img = np.transpose(torch_image,(1,2,0))
    if is_negone_to_one:
        img = (img + 1)/2 #[0,1]
        img = (img * 255).round().clip(0, 255)
    else:
        img = img.round().clip(0, 255)
    return img.astype(np.uint8)[:,:,::-1]
def load_image(fdir):
    img = sorted(glob.glob(os.path.join(fdir,'*.png')))[0]
    return cv2.imread(img)
def interpolate_color(value,min_val=0.6,max_val=1.3):
    value = (value-min_val)/(max_val-min_val)
    value = max(value,0)
    value = min(value,1)
    green = int((1 - value) * 255)
    blue = 0
    red = int(value * 255)
    return (blue,green,red)
def plot_all(results):
    ref_embed = results[REFERENCE]['embeddings']
    tar_embed = results[TARGET]['embeddings']

    dists = [[torch.linalg.norm(e1 - e2).item() for e2 in ref_embed] for e1 in tar_embed]
    target_img= load_image(os.path.join(IMAGE_DIR,TARGET))

    for box,dist in zip(results['targets']['boxes'],dists):
        box = [int(round(b)) for b in box]
        color = interpolate_color(dist[0])
        # if dist[0]< 0.61:
        #     color = (0,255,0)
        # else:
        #     color = (0,0,255)
        top_left = box[:2]
        bottom_right = box[2:]
        cv2.rectangle(target_img,top_left,bottom_right,color,2)
        #print(dist)
        label_text = f"{dist[0]:0.2f}"
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = 0.6
        label_thickness = 1
        label_color = (0, 0, 0)  # (B, G, R)
        label_org = (top_left[0], top_left[1] - 10)  # Slightly above the rectangle's top-left

        cv2.putText(target_img, label_text, label_org, label_font, label_font_scale, color, label_thickness)


    cv2.imwrite(OUT,target_img)
        

def resnet_inference_aggregate_class(mtcnn,resnet,loader,dataset,device):
    """
    Returns
    {
        references : mean_embedding,
        targets : {[embedding],[boxes]},
    }
    """
    global GLOBAL_I
    memo = {}
    #results = {}
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        # Detect face
        boxes, probs = mtcnn.detect(x, landmarks=False)
        if boxes is not None and x_aligned is not None:   
            boxes = [b for b,p in zip(boxes,probs) if p > 0.8]
            #print(len(boxes),len(x_aligned))         
            name = dataset.idx_to_class[y]
            if name not in memo:
                memo[name]= {'aligneds':[x_aligned],'boxes':[boxes]}
            else:
                memo[name]['aligneds'].append(x_aligned)
                memo[name]['boxes'].append(boxes)

    for k,v in memo.items():
        v['aligneds']=    torch.concat(v['aligneds'],dim=0).to(device)
        v['boxes']=    np.vstack(v['boxes'])
        embeddings = resnet(v['aligneds']).detach().cpu()
        if k ==REFERENCE:
            v['embeddings']=embeddings.mean(axis=0)[None,:]
        else:
            v['embeddings']=embeddings
            
    
    
    return memo

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Running on device: {}'.format(device))

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device,
        keep_all=True
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


    dataset_ref = datasets.ImageFolder(IMAGE_DIR)
    dataset_ref.idx_to_class = {i:c for c, i in dataset_ref.class_to_idx.items()}
    loader = DataLoader(dataset_ref, collate_fn=collate_fn, num_workers=workers)

    results = resnet_inference_aggregate_class(mtcnn,resnet,loader,dataset_ref,device)
    plot_all(results)


if __name__=='__main__':
    main()