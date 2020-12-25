import matplotlib.pyplot as plt

import matplotlib.ticker as ticker


def draw_pr_curve(objects, GT_num):
    TP = 0
    FP = 0
    precision_acc = []
    recall_acc = []
    objects.sort(key=lambda x: 1000*x['confidence'] + x['valid'], reverse=True)
    # print(objects)
    for obj in objects:
        if obj['valid']:
            TP += 1
        else:
            FP += 1
        precision_acc.append(TP/(TP+FP))
        recall_acc.append(TP/GT_num)
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(recall_acc, precision_acc, 'o-',
             color='g', label="Precision-Recall")
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    predict_results = [(0.91, False), (0.87, False),
                       (0.91, True), (1., True), (0.85, True), (0.9, True)]
    GT_num = 5
    draw_pr_curve([{'confidence': a, 'valid': b}
                   for a, b in predict_results], GT_num)
