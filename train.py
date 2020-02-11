from google.colab import drive
drive.mount('/content/drive')

!pip install visdom torchvision


from tqdm import tqdm
import os
import random
import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from visualizer import Visualizer
import data_utils.ext_transforms as et
from data_utils.stream_metrics import StreamSegMetrics, AverageMeter
from data_utils.data_loader import VOCSegmentation
from data_utils.utils import set_bn_momentum, Denormalize
from data_utils.scheduler import PolyLR
from network._deeplab import DeepLabHeadV3Plus, convert_to_separable_conv
from network.modeling import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet, deeplabv3plus_resnet50, deeplabv3plus_resnet101, deeplabv3plus_mobilenet


def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    save_val_results = True
    if save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    plt.imshow(pred),plt.show()
                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)
            
                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def save_ckpt(path):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)

def get_dataset(train_img_dir, train_mask_dir, valid_img_dir, valid_mask_dir,
                crop_size = 513, crop_val = False):
    img_dir = '/content/drive/My Drive/tmp/voc_jpeg'
    mask_dir = '/content/drive/My Drive/tmp/voc_anno'
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    if crop_val:
        val_transform = et.ExtCompose([
            et.ExtResize(crop_size),
            et.ExtCenterCrop(crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    train_dst = VOCSegmentation(train_img_dir, train_mask_dir, image_set='train', transform=train_transform)
    val_dst = VOCSegmentation(valid_img_dir, valid_mask_dir, image_set='val', transform=val_transform)

   
    return train_dst, val_dst



def main():
    # config
    enable_vis = False
    vis_env = 'main'
    dataset= 'voc'
    vis_port = 13570
    vis = Visualizer(port=vis_port, env=vis_env) if enable_vis else None
    os.mkdir('checkpoints')



    print('set up parameter')
    val_batch_size = 1
    num_classes = 51

    output_stride = 16
    lr = 0.01
    weight_decay = 1e-4
    total_itrs = 30e3


    print('random seed')
    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    print('setting up device')
    gpu_id = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # load dataset
    train_img_dir = 'drive/My Drive/dataset2/images_prepped_train'
    train_mask_dir = 'drive/My Drive/dataset2/annotations_prepped_train'
    valid_img_dir = 'drive/My Drive/dataset2/images_prepped_test'
    valid_mask_dir = 'drive/My Drive/dataset2/annotations_prepped_test'

    print('loading data....')
    train_dst, val_dst = get_dataset(train_img_dir, train_mask_dir, valid_img_dir, valid_mask_dir)
    train_loader = data.DataLoader(
        train_dst, batch_size=16, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=4, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %('voc', len(train_dst), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': deeplabv3_resnet50,
        'deeplabv3plus_resnet50': deeplabv3plus_resnet50,
        'deeplabv3_resnet101': deeplabv3_resnet101,
        'deeplabv3plus_resnet101': deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': deeplabv3plus_mobilenet
    }
    model = model_map['deeplabv3plus_mobilenet'](num_classes=num_classes, output_stride=output_stride)


    separable_conv = False
    if separable_conv and 'plus' in 'deeplabv3plus_mobilenet':
        convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)
    metrics = StreamSegMetrics(num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*lr},
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9, weight_decay=weight_decay)

    print('setting up model....')
    scheduler = PolyLR(optimizer, total_itrs, power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # Restore
    model = nn.DataParallel(model)
    model.to(device)

    #==========   Train Loop   ==========#
    vis_num_samples = 8
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                    np.int32) if enable_vis else None  # sample idxs for visualization
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    interval_loss = 0
    val_interval = 400
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    (cur_epochs, cur_itrs, total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % val_interval == 0:
                save_ckpt('checkpoints/latest_deeplabv3plus_os{}d.pth'.format(cur_itrs))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_os%d.pth' %
                            (dataset, output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)

            if cur_itrs >= total_itrs:
                break
if __name__ == '__main__':
    main()