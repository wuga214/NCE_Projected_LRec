import matplotlib.pyplot as plt
import seaborn as sns


def curve_weighting(lrec_result, wlrec_results, weights, metric, name):
    plt.figure(figsize=(6, 4))
    plt.hlines(lrec_result[metric], weights[0], weights[-1],
               color=sns.xkcd_rgb["pale red"], linestyles="dashed", label='Projected LRec')

    scores = []
    for i in weights:
        scores.append(wlrec_results[i][metric])

    plt.plot(weights, scores, sns.xkcd_rgb["denim blue"], label='Weighted Projected LRec')
    plt.legend()
    plt.xscale('log')

    plt.savefig('figures/{0}.pdf'.format(name), format='pdf')
    plt.show()