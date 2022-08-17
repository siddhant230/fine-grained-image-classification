from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report


def get_performance(model, dataloader, device='cuda'):
    label_list = None
    pred_list = None

    loop = tqdm(dataloader, total=len(dataloader), leave=True)
    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)

        if label_list is None:
            label_list = labels.detach().cpu().numpy()
        else:
            label_list = np.concatenate([label_list,
                                         labels.detach().cpu().numpy()])

        if pred_list is None:
            pred_list = outputs.detach().cpu().numpy().argmax(axis=1)
        else:
            pred_list = np.concatenate([pred_list,
                                        outputs.detach().cpu().numpy().argmax(axis=1)])
    print("\n\n")
    print(classification_report(label_list, pred_list))
