from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import os
import cv2
import torchvision.transforms as transforms
import glob
from scipy.optimize import linear_sum_assignment
workers = 0 if os.name == 'nt' else 4
PATH_TO_ROWS = 'images/rows'
PATH_TO_COLUMNS = 'images/columns'
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

def pad_to_square(image):
    h,w = image.shape[:2]
    dim = max(h,w)
    new = np.zeros((dim,dim,3),dtype=image.dtype)
    w_offset = (dim-w)//2
    h_offset = (dim-h)//2
    new[h_offset:h_offset+h,w_offset:w_offset+w]=image
    return new  
def add_border(image,width=1,color=(255,255,255)):
    h,w = image.shape[:2]
    new = np.zeros((h+width*2,w+width*2,3),dtype=image.dtype) + np.array(color,dtype=image.dtype)
    new[width:width+h,width:width+w]=image
    return new



def resnet_inference(mtcnn,resnet,loader,dataset,device):
    """
    Returns
    {
        name : {embedding:mean_embedding,image:first_img}
    }
    """

    memo = {}
    results = {}
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        # Detect face
        boxes, probs = mtcnn.detect(x, landmarks=False)
        if boxes is not None and x_aligned is not None:   
            #print(len(boxes),len(x_aligned))         
            name = dataset.idx_to_class[y]
            if name not in memo:
                memo[name]= {'aligneds':[x_aligned],'boxes':[boxes],'image':np.array(x)[:,:,::-1]}
            else:
                memo[name]['aligneds'].append(x_aligned)
                memo[name]['boxes'].append(boxes)

    for k,v in memo.items():
        v['aligneds']=    torch.concat(v['aligneds'],dim=0).to(device)
        v['boxes']=    np.vstack(v['boxes'])
        embeddings = resnet(v['aligneds']).detach().cpu()
        # v['embeddings']=embeddings.mean(axis=0)[None,:]    
        results[k]={
            'embeddings':embeddings.mean(axis=0)[None,:],
            'image': v['image']#tensor_to_image((v['aligneds'][0]).detach().cpu().numpy())
        }
    return results
def calc_distances(results_male,results_female):
    dists = {k1:{k2:torch.linalg.norm(v1['embeddings'] - v2['embeddings']).item() for k2,v2 in results_female.items()} for k1,v1 in results_male.items()}
    return dists
def interpolate_color(value):
    value = max(value,0)
    value = min(value,1)
    green = int((1 - value) * 255)
    blue = 0
    red = int(value * 255)
    return (blue,green,red)
def dict_to_values(dists):
    row_names = list(dists.keys())
    column_names = list(dists[row_names[0]].keys())
    return [[dists[rname][cname] for cname in column_names] for rname in row_names]

def create_image_grid(rows,columns,dists,dim=200):
    row_names = list(dists.keys())
    column_names = list(dists[row_names[0]].keys())

    vals = np.array(dict_to_values(dists))
    max_val,min_val=1.3,0.6#np.max(vals),np.min(vals)

    dtype = columns[column_names[0]]['image'].dtype
    empty = np.zeros((dim,dim,3),dtype=dtype)

    top_row = [empty]+[cv2.resize(pad_to_square(columns[k]['image']),(dim,dim)) for k in column_names]
    top_row = np.hstack([add_border(item) for item in top_row])

    all_rows = [top_row]
    for name in row_names:
        row = [cv2.resize(pad_to_square(rows[name]['image']),(dim,dim))]
        for cname in column_names:
            val = dists[name][cname]
            color = interpolate_color((val-min_val)/(max_val-min_val))
            row.append(create_text_image(f"{val:02f}",color=color))
        row = np.hstack([add_border(item) for item in row])
        all_rows.append(row)
    
    all_rows = np.vstack(all_rows)
    return all_rows
    #left_col = [cv2.resize(pad_to_square(rows[k]['image']),(dim,dim)) for k in row_names]
    #left_col = np.vstack([add_border(item) for item in left_col])
    #cv2.imwrite('tmp.png',left_col)

    # tmp = create_text_image('101.01')
    # cv2.imwrite('tmp2.png',tmp)
def create_text_image(text,dim=200,dtype=np.uint8,color=(255, 255, 255)):  
    # Create a black square image
    image = np.zeros((dim,dim, 3), dtype=dtype)
    
    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (dim - text_size[0]) // 2
    text_y = (dim + text_size[1]) // 2
    
    # Draw white text on the image
    cv2.putText(image, text, (text_x, text_y), font, 1, color , 2, cv2.LINE_AA)
    
    return image
def calc_matches(dists):
    row_names = list(dists.keys())
    column_names = list(dists[row_names[0]].keys())
    cost_matrix = np.array(dict_to_values(dists))
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    print("Calculating Optimal Matches")
    for r,c in zip(row_indices,col_indices):
        print(f"{row_names[r]} and {column_names[c]}, dist: {cost_matrix[r,c]:02f}")



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


    dataset_male = datasets.ImageFolder(PATH_TO_ROWS)
    dataset_male.idx_to_class = {i:c for c, i in dataset_male.class_to_idx.items()}
    loader_male = DataLoader(dataset_male, collate_fn=collate_fn, num_workers=workers)
    results_male = resnet_inference(mtcnn,resnet,loader_male,dataset_male,device)

    dataset_female = datasets.ImageFolder(PATH_TO_COLUMNS)
    dataset_female.idx_to_class = {i:c for c, i in dataset_female.class_to_idx.items()}
    loader_female = DataLoader(dataset_female, collate_fn=collate_fn, num_workers=workers)
    results_female = resnet_inference(mtcnn,resnet,loader_female,dataset_female,device)

    

    dists=calc_distances(results_male,results_female)
    grid=create_image_grid(results_male,results_female,dists)
    cv2.imwrite('grid.png',grid)

    calc_matches(dists)

if __name__=='__main__':
    main()