import pickle
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

def plot_pr_curves(px_list, py_list, ap_list, model_names, save_dir=Path('pr_curves.png')):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for i in range(len(px_list)):
        px = px_list[i]
        py = np.stack(py_list[i], axis=1)
        ap = ap_list[i]

        ax.plot(px, py[:,0], linewidth=2, label='%s  all classes %.3f mAP@0.5' % (model_names[i], ap[0,0]))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')

    ax.set_title('Precision-Recall Curves')

    fig.savefig(save_dir, dpi=250)
    plt.close(fig)

def deserialize_plot_pr_curves(pr_files, model_names):
    all_px_list = []
    all_py_list = []
    all_ap_list = []
    for pr_file in pr_files:
        with open(pr_file, 'rb') as f:
            data = pickle.load(f)
        px, py, ap, names = data
        all_px_list.append(px)
        all_py_list.append(py)
        all_ap_list.append(ap)

    plot_pr_curves(px_list=all_px_list, py_list=all_py_list, ap_list=all_ap_list, model_names=model_names)

# Example usage:
model_names = ['Model-A', 'Model-B']
pr_files = ['pr-1.pkl', 'pr-2.pkl']
deserialize_plot_pr_curves(pr_files, model_names)
