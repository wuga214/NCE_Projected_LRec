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
    plt.xlabel('Alpha')
    plt.ylabel(metric)

    plt.savefig('figures/{0}.pdf'.format(name), format='pdf')
    plt.show()


def scatter_plot(data, weight, name, folder='figures/latent', save=False):
    fig, ax = plt.subplots(figsize=(4, 3))
    sc = ax.scatter(data[:, 0], data[:, 1], s=10, c=weight)
    plt.colorbar(sc, orientation='horizontal', pad=0.2)
    plt.xlabel("1st-axis")
    plt.ylabel("2ed-axis")
    plt.tight_layout()
    if save:
        plt.savefig("{0}/latent_{1}.pdf".format(folder, name), format="pdf")
    plt.show()