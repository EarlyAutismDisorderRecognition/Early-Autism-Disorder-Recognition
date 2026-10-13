import numpy as np
import warnings
from matplotlib import pylab
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pylab
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['Terms', 'GTO-MGA2SBR', 'BCO-MGA2SBR', 'BOA-MGA2SBR', 'OOA-MGA2SBR', 'PRHOA-MGA2SBR']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((Fitness.shape[1], 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report ',
          ' --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence.png" )
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence Curve')
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def plot_conv_Feat():
    Fitness = np.load('Fitness_Feat.npy', allow_pickle=True)
    Algorithm = ['Terms', 'GTO-A2RP', 'BCO-A2RP', 'BOA-A2RP', 'OOA-A2RP', 'PRHOA-A2RP']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((Fitness.shape[1], 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])
    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Feature Statistical Report ',
          ' --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
    plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
    plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
    plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Convergence_Feature.png" )
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Feature Convergence Curve')
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def ROC_curve():
    lw = 2
    cls = ['Deep GCN', 'GCN-dual transformer', 'CNN-BiLSTM-AN', 'SigGate Bi-RNN', 'PRHOA-MGA2SBR']
    Actual = np.load('Abide_Targets.npy', allow_pickle=True).astype('int')
    colors = cycle(["#fe2f4a", "#0165fc", "#00ffff", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('ROC curve')
    plt.savefig(path)
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    cm = confusion_matrix(np.asarray(Actual), np.asarray(Predict))
    Classes = ['Normal', 'Abnormal']
    rot = 0
    fontsizes = 12
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues', values_format='d', text_kw={'fontsize': 12})
    ax.set_xlabel('Predicted labels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual labels', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
    ax.set_yticklabels(Classes, fontsize=fontsizes)
    plt.tight_layout()
    path = "./Results/Confusion_matrix.png"
    plt.savefig(path)
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def Plot_Activation():
    eval = np.load('Eval_ALL_ACT.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    Graph_Terms = [0, 3, 6, 7, 8, 9, 10, 12]
    Activation = ["TanH", "Sigmoid", "Softmax"]
    Classifier = ['SigGate Bi-RNN', 'PRHOA-MGA2SBR']
    colors = ["#c2e699", "#78c679", "#238443"]

    for k in range(len(Graph_Terms)):
        Graph = eval[:, :, Graph_Terms[k]+4]
        Graph = Graph[:, :]
        value = Graph[2:, -2:]
        values = value.squeeze()
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#eaeaea')
        ax.set_facecolor('#eaeaea')
        bar_height = 0.2
        group_height = 1.0
        y_ticks = []
        y_labels = []

        for i, classifier in enumerate(Classifier):
            for j, lp in enumerate(Activation):
                bar_val = values[j, i]
                y_pos = i * group_height + j * bar_height
                ax.barh(y=y_pos, width=bar_val, height=bar_height, color=colors[j])  # , edgecolor='w')

                offset = bar_val * 0.01  # 1% from the right edge
                text_val = f"{bar_val:.3f}"
                ax.text(bar_val - offset, y_pos, text_val, va='center', ha='right', fontsize=12, fontweight='bold',
                        color='white')
                y_ticks.append(i * group_height + bar_height)
            y_labels.append(classifier)

        ax.set_yticks([t * group_height + bar_height for t in range(len(Classifier))])
        ax.set_yticklabels(y_labels, fontsize=11)
        ax.set_xlabel(Terms[Graph_Terms[k]], fontsize=12, fontweight='bold')

        legend_elements = [
            Patch(facecolor=colors[0], label=Activation[0]),
            Patch(facecolor=colors[1], label=Activation[1]),
            Patch(facecolor=colors[2], label=Activation[2]),
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, markerscale=5,
                   handleheight=1, handlelength=1, fontsize=12, frameon=False)

        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        path = f"./Results/Activation {Terms[Graph_Terms[k]]} Method.png"
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title(f"Activation {Terms[Graph_Terms[k]]}")
        plt.tight_layout()
        plt.savefig(path)
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def darken_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return tuple(max(min(x * amount, 1), 0) for x in c)


class HandlerTriangle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        verts = [
            [width / 2, height],  # top
            [0, 0],  # bottom left
            [width, 0]  # bottom right
        ]

        tri = Polygon(
            verts,
            facecolor=orig_handle.get_facecolor(),
            edgecolor='black',
            transform=trans
        )
        return [tri]


def Plot_Kfold():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    Graph_Terms = [0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

    Algorithm = np.asarray(['GTO-MGA2SBR', 'BCO-MGA2SBR', 'BOA-MGA2SBR', 'OOA-MGA2SBR', 'PRHOA-MGA2SBR'])
    Classifier = np.asarray(['Deep GCN', 'GCN-dual transformer', 'CNN-BiLSTM-AN', 'SigGate Bi-RNN', 'PRHOA-MGA2SBR'])
    Kfold = ['1', '2', '3', '4', '5']
    colors_alg = ['#0014CC', '#EA3680', '#86D176', '#FF3727', '#80DEEA']
    colors_cls = ['#B39DDB', '#EF9A9A', '#FFF59D', '#FFCC80', '#80DEEA']
    Kfold = np.asarray(Kfold)
    for j in range(len(Graph_Terms)):
        Graph = eval[:, :, Graph_Terms[j] + 4]

        Alg_Values = Graph[:, :5].transpose()
        tri_height = 0.05
        bars_per_group = 5
        width = 0.25
        gap = 0.20
        num_groups = len(Kfold)
        x = np.arange(num_groups) * (bars_per_group * width + gap)
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_containers = []
        for i in range(bars_per_group):
            heights = Alg_Values[i, :]
            bar_color = colors_alg[i]
            bars = ax.bar(
                x + i * width,
                heights,
                width,
                color=bar_color,
                label=Algorithm[i]
            )
            bar_containers.append(bars[0])
            dark_color = darken_color(bar_color, amount=0.5)
            for bar in bars:
                height = bar.get_height()
                cx = bar.get_x() + bar.get_width() / 2
                tri = Polygon([
                    [cx, height + tri_height],
                    [cx - width / 2, height],
                    [cx + width / 2, height]
                ], closed=True, color=dark_color)

                ax.add_patch(tri)
                num_dots = 12
                dot_y_positions = np.linspace(0, height, num_dots)
                for y in dot_y_positions:
                    ax.scatter(cx, y, color='white', s=40, zorder=5, edgecolors='black')
        ax.set_xlabel('K Fold', fontsize=15, fontweight='bold')
        ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=15, fontweight='bold')
        ax.set_xticks(x + (bars_per_group * width) / 2.5)
        ax.set_xticklabels(Kfold, fontsize=12, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(bar_containers, Algorithm, loc='upper center',
                  bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=12, frameon=False, handler_map={Polygon: HandlerTriangle()})
        path = "./Results/Kfold %s Alg.png" % (Terms[Graph_Terms[j]])
        plt.tight_layout()
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        # Method
        Mtd_Values = Graph[:, 5:].transpose()
        num_groups = len(Kfold)
        x = np.arange(num_groups) * (bars_per_group * width + gap)
        fig, ax = plt.subplots(figsize=(12, 6))
        bar_containers = []
        for i in range(bars_per_group):
            heights = Mtd_Values[i, :]
            bar_color = colors_cls[i]
            bars = ax.bar(
                x + i * width,
                heights,
                width,
                color=bar_color,
                label=Classifier[i]
            )
            bar_containers.append(bars[0])
            dark_color = darken_color(bar_color, amount=0.5)
            for bar in bars:
                height = bar.get_height()
                cx = bar.get_x() + bar.get_width() / 2
                tri = Polygon([
                    [cx, height + tri_height],
                    [cx - width / 2, height],
                    [cx + width / 2, height]
                ], closed=True, color=dark_color)

                ax.add_patch(tri)
                num_dots = 12
                dot_y_positions = np.linspace(0, height, num_dots)
                for y in dot_y_positions:
                    ax.scatter(cx, y, color='white', s=40, zorder=5, edgecolors='black')
        ax.set_xlabel('K Fold', fontsize=15, fontweight='bold')
        ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=15, fontweight='bold')
        ax.set_xticks(x + (bars_per_group * width) / 2.5)
        ax.set_xticklabels(Kfold, fontsize=12, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(bar_containers, Classifier, loc='upper center',
                  bbox_to_anchor=(0.5, 1.15), fontsize=12, ncol=3, frameon=False, handler_map={Polygon: HandlerTriangle()})
        path = "./Results/Kfold %s Mtd.png" % (Terms[Graph_Terms[j]])
        plt.tight_layout()
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('K Fold vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        # plt.show()
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def Plot_Step_per_epoch():
    eval = np.load('Eval_ALL_SPE.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    Table_Term = [0, 2, 4, 5, 6, 7, 8, 9, 10, 12]
    SPE = ['50', '100', '150', '200', '250']
    Algorithm = ['Step Per Epoch', 'GTO-MGA2SBR', 'BCO-MGA2SBR', 'BOA-MGA2SBR', 'OOA-MGA2SBR', 'PRHOA-MGA2SBR']
    Classifier = ['Step Per Epoch', 'Deep GCN', 'GCN-dual transformer', 'CNN-BiLSTM-AN', 'SigGate Bi-RNN', 'PRHOA-MGA2SBR']
    for k in range(len(Table_Term)):
        value = eval[:5, :, Table_Term[k] + 4]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], SPE)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[:, j])
        print('--------------------------------------------------', str(Terms[Table_Term[k]]),
              ' Step Per Epoch vs Algorithm Comparison --------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], SPE)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1])
        print('-------------------------------------------------- ', str(Terms[Table_Term[k]]),
              'Step Per Epoch vs Classifier Comparison  --------------------------------------------------')
        print(Table)


if __name__ == '__main__':
    plot_conv()
    plot_conv_Feat()
    ROC_curve()
    Plot_Confusion()
    Plot_Activation()
    Plot_Kfold()
    Plot_Step_per_epoch()
