import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.kde import gaussian_kde
import numpy as np


def curve_weighting(lrec_result, wlrec_results, weights, metric, name):
    plt.figure(figsize=(4, 2.5))
    plt.hlines(lrec_result[metric][0], weights[0], weights[-1],
               color=sns.xkcd_rgb["pale red"], linestyle='--', label='Projected LRec')

    scores = []
    errs = []
    for i in weights:
        scores.append(wlrec_results[i][metric][0])
        errs.append(wlrec_results[i][metric][1])

    plt.plot(weights, scores, sns.xkcd_rgb["denim blue"], label='Weighted Projected LRec')
    plt.legend(loc=4)
    #plt.xscale('log')
    plt.xlabel('Alpha')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig('figures/{0}.pdf'.format(name), format='pdf')
    plt.show()


def curve_sensitivity(lrec_result, wlrec_results, weights, metric, name):
    plt.figure(figsize=(4, 2.5))
    plt.hlines(lrec_result[metric][0], weights[0], weights[-1],
               color=sns.xkcd_rgb["pale red"], linestyle='--', label='Default PMI')

    scores = []
    errs = []
    for i in weights:
        scores.append(wlrec_results[i][metric][0])
        errs.append(wlrec_results[i][metric][1])

    plt.plot(weights, scores, sns.xkcd_rgb["denim blue"], label='Modified PMI')
    plt.legend(loc=4)
    plt.xlabel('Beta')
    plt.ylabel(metric)
    plt.tight_layout()
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


def pandas_scatter_plot(df, model1, model2, metric, pos_percentage, neg_percentage, max,
                        folder='figures/pairwise', save=True):
    fig, ax = plt.subplots(figsize=(4, 4))

    # k = gaussian_kde(np.vstack([df.x, df.y]))
    # xi, yi = np.mgrid[df.x.min():df.x.max():df.x.size ** 0.5 * 1j, df.y.min():df.y.max():df.y.size ** 0.5 * 1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    #
    # ax.contourf(xi, yi, np.log2(zi.reshape(xi.shape)), 10, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
    sns.kdeplot(df.x, df.y, ax=ax, cmap="Blues", shade=True, shade_lowest=False)
    ax.plot([0, 0.6+0.01], [0, 0.6+0.01], ls="--")
    ax.set_xlim([0, 0.6+0.01])
    ax.set_ylim([0, 0.6+0.01])
    plt.legend(loc='upper right')
    plt.xlabel('{0} (Counts:{1})'.format(model1, pos_percentage))
    plt.ylabel('{0} (Counts:{1})'.format(model2, neg_percentage))
    plt.tight_layout()
    if save:
        plt.savefig("{0}/{1}_{2}_{3}_scatter.pdf".format(folder, model1, model2, metric), format="pdf")
    else:
        plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.hist(df['diff'].values, bins=21, log=True, align='mid')
    plt.axvline(x=[0.0], color='red', ls='--')
    ax.set_xlim([-max - 0.01, max + 0.01])
    plt.xlabel('{0}        {1}'.format(model2, model1))
    plt.tight_layout()
    if save:
        plt.savefig("{0}/{1}_{2}_{3}_hist.pdf".format(folder, model1, model2, metric), format="pdf")
    else:
        plt.show()
    plt.close()


def pandas_bar_plot(df, x, y, hue, x_name, y_name, folder='figures/user_rating', save=True):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(ax=ax, x=x, y=y, hue=hue, data=df, errwidth=1)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xticks(rotation=15)
    if y != 'Recall@50':
        ax.legend_.remove()
    plt.tight_layout()
    if save:
        plt.savefig("{0}/{1}_bar.png".format(folder, y_name), format="png")
    else:
        plt.show()
    plt.close()


def pandas_group_hist_plot(df, var, group, x_name, y_name, folder='figures', save=True):
    plt.figure(figsize=(30, 10))
    g = sns.FacetGrid(df, hue=group, size=3)
    g.map(sns.distplot, var)
    g.add_legend()
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.tight_layout()
    if save:
        plt.savefig("{0}/{1}_hist.png".format(folder, y_name), format="png")
    else:
        plt.show()
    plt.close()



