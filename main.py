from dataset.dataset import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models.ssd import *
# https://github.com/qfgaohao/pytorch-ssd
#import torchvision.transforms as transforms
DATASET_PATH = {
    'path': '/media/jake/mark-4tb3/input/kaggle_4tb/imagenet-object-localization-challenge/',
    'cifar_path' : '/media/jake/mark-4tb3/input/datasets/',
    'PennFudanPed' : '/media/jake/mark-4tb3/input/datasets/PennFudanPed/',
}

COCO = {

    'path': '/media/jake/mark-4tb3/input/datasets/coco',
    'train': '/media/jake/mark-4tb3/input/datasets/coco/train2017',
    'test': '/media/jake/mark-4tb3/input/datasets/coco/test2017',
    'path2json': '/media/jake/mark-4tb3/input/datasets/coco/instances_train2017.json',
    'save_images': '/media/jake/mark-4tb3/input/datasets/coco/images/'
}
CUDA_VISIBLE_DEVICES=1
def draw_box(img,target):
    #img,target = train[num]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    blue_color = (255, 0, 0)
    img = np.array(img)
    for i in range(len(target)):
        bbox = target[i]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x, y, w, h = int(x), int(y), int(w), int(h)
        img_bbox = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    im = Image.fromarray(img_bbox)
    im.save("./images/your_file.jpeg")

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))
# In my case, just added ToTensor

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    #custom_transforms.append(torchvision.transforms.CenterCrop())
    #custom_transforms.append(torchvision.transforms.Resize(64,64))
    return torchvision.transforms.Compose(custom_transforms)


def train(data_loader):
    # 2 classes; Only target class or background

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(1)

    num_classes = 2
    num_epochs = 10
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0
        for imgs, annotations in data_loader:
            i += 1
            #toTensor = torchvision.transforms.ToTensor()
            #imgs = list(toTensor(img).to(device) for img in imgs)
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')

def main():

    cocod_dataset = CocoDataset(COCO['train'],COCO['path2json'],transforms=get_transform())

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(cocod_dataset,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=collate_fn)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__=='__main__':
    main()


  # DataLoader is iterable over Dataset
    #for imgs, annotations in data_loader:
    #    #toTensor = torchvision.transforms.ToTensor()
    #    imgs = list(img.to(device) for img in imgs)
    #    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #    print(imgs[0].shape)
    #    print(annotations)
    #    break