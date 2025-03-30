import random
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pprint as pprint
from sklearn.metrics import confusion_matrix
import numpy as np
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
_utils_pp = pprint.PrettyPrinter()

def text_read(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.strip('\n')
    return lines


def pprint(x):
    _utils_pp.pprint(x)


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    plt.show()

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint



def load_pretrained_weights(model, weight_path):
    checkpoint = load_checkpoint(weight_path)
    state_dict=checkpoint ['params']
    # if "state_dict" in checkpoint:
    #     state_dict = checkpoint["state_dict"]
    # else:
    #     state_dict = checkpoint

    model_dict = model.state_dict()
    if "prompt_learner.token_prefix" in state_dict:
        del state_dict["prompt_learner.token_prefix"]

    if "prompt_learner.token_suffix" in state_dict:
        del state_dict["prompt_learner.token_suffix"]
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.
        # if k.startswith("clip_model."):
        #     k = k[11:]  # discard "clip."

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f"Cannot load {weight_path} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )


def calculate_class_accuracy(preds, labels, class_names):
    class_stats = {cls: {"correct": 0, "total": 0} for cls in class_names}
    for pred, label in zip(preds, labels):
        class_name = class_names[label.item()]  # 使用 label 作为索引直接获取类名
        class_stats[class_name]["total"] += 1
        if pred == label:
            class_stats[class_name]["correct"] += 1

    class_accuracies = {
        cls: stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0
        for cls, stat in class_stats.items()
    }
    return class_accuracies


def calculate_overall_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def write_accuracy_changes(before_acc, after_acc, class_names, overall_before, overall_after, output_path):
    accuracy_changes = []
    for cls in class_names:
        change = before_acc[cls] - after_acc[cls]
        accuracy_changes.append({
            "class": cls,
            "before": before_acc[cls],
            "after": after_acc[cls],
            "decrease": change
        })
    
    accuracy_changes.append({
        "class": "Overall",
        "before": overall_before,
        "after": overall_after,
        "decrease": overall_before - overall_after
    })

    accuracy_changes = sorted(accuracy_changes, key=lambda x: x["decrease"], reverse=True)

    with open(output_path, "w") as f:
        json.dump(accuracy_changes, f, indent=4)
    print(f"Results saved to {output_path}")

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None):
            with torch.no_grad():
                pretrained_state_dict = pretrained_checkpoint
                finetuned_state_dict = finetuned_checkpoint
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_checkpoint
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] -scaling_coef * self.vector[key]
        return  new_state_dict



