import pickle
import pickle
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_mc_curves(px_list, py_list,model_names, save_dir=Path('mc_curve.png'),xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for i in range(len(px_list)):
    #绘制all classes曲线
        px = px_list[i]
        py = py_list[i]
        y = smooth(py.mean(0), 0.05)
        #绘制all classes曲线并加上model名字
        ax.plot(px, y, linewidth=1, label= f'{model_names[i]}  all classes {y.max():.2f} at {px[y.argmax()]:.3f}')


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    save_dir=Path(save_dir)
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)



def deserialize_plot_mc_curves(mc_files,model_names_1):
    all_px_list = []
    all_p_list = []
    all_r_list = []
    all_f1_list = []
    for mc_file in mc_files:
        with open(mc_file, 'rb') as f:
            data = pickle.load(f)
        px,py,p, r, f1, ap = data
        all_px_list.append(px)
        all_p_list.append(p)
        all_r_list.append(r)
        all_f1_list.append(f1)

    plot_mc_curves(all_px_list, all_f1_list, model_names, ylabel='F1',save_dir=Path('F1_curves.png'))
    plot_mc_curves(all_px_list, all_p_list,model_names, ylabel='Precision',save_dir=Path('P_curves.png'))
    plot_mc_curves(all_px_list, all_r_list, model_names, ylabel='Recall',save_dir=Path('R_curves.png'))

# Example usage:
model_names = ['swin','RDD-gelu', 'yolov5']
mc_files = ['swin_metric.pkl','gelu_metric.pkl', 'yolov5s_metric.pkl']
deserialize_plot_mc_curves(mc_files,model_names)









