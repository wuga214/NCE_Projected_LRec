import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from utils.io import load_yaml
from ast import literal_eval
sns.axes_style("white")


def show_training_progress(df, hue='model', metric='NDCG', name="epoch_vs_ndcg", save=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    #plt.axhline(y=0.165, color='r', linestyle='-')
    ax = sns.lineplot(x='epoch', y=metric, hue=hue, style=hue, data=df)
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        fig.savefig('{0}/train/progress/{1}.png'.format(fig_path, name),
                    bbox_inches="tight", pad_inches=0, format='png')
    else:
        plt.show()


def show_uncertainty(df, x, y, hue='model', folder='unknown', name='uncertainty_analysis', save=True):
    fig, ax = plt.subplots(figsize=(6, 4))
    models = df[hue].unique()
    for model in models:
        sns.regplot(x=x, y=y, data=df[df[hue] == model], lowess=True, scatter=False, ax=ax, label=model)
    plt.ylabel("Model Uncertainty \n (Standard Derivation)")
    plt.xlabel("Number of Ratings")
    plt.legend(loc='center right')
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        fig.savefig('{0}/analysis/{1}/{2}.pdf'.format(fig_path, folder, name),
                    bbox_inches="tight", pad_inches=0, format='pdf')
        fig.savefig('{0}/analysis/{1}/{2}.png'.format(fig_path, folder, name),
                    bbox_inches="tight", pad_inches=0, format='png')
    else:
        plt.show()


def pandas_ridge_plot(df, model, pop, k, folder='figures', name='personalization', save=True):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    num_models = len(df.model.unique())


    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(num_models, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row=model, hue=model, aspect=10, height=1, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, pop, clip_on=False, shade=True, alpha=1, lw=1.5, bw=50)
    g.map(sns.kdeplot, pop, clip_on=False, color="w", lw=1.5, bw=50)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.1, .1, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, pop)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.8)

    # Remove axes details that don't play well with overlap

    g.set_xlabels("Popularity Distribution of The Top-{0} Recommended Items".format(k))
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    #plt.tight_layout()
    if save:
        pass
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        plt.savefig("{2}/{0}/{1}.pdf".format(folder, name, fig_path), format="pdf")
        plt.savefig("{2}/{0}/{1}.png".format(folder, name, fig_path), format="png")
    else:
        plt.show()
    plt.close()


def pandas_bar_plot(df, x, y, hue, x_name, y_name, folder='figures', name='unknown', save=True):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(ax=ax, x=x, y=y, hue=hue, data=df, errwidth=1, edgecolor='black', facecolor=(1, 1, 1, 0)) #, errwidth=0.5

    num_category = len(df[x].unique())
    hatch = None
    hatches = itertools.cycle(['//', '**', '////', '----', 'xxxx', '\\\\\\\\', ' ', '\\', '...', 'OOO', "++++++++"])
    for i, bar in enumerate(ax.patches):
        if i % num_category == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    #plt.xticks(rotation=15)
    plt.legend(loc='upper left', ncol=5)
    # if 'Precision' not in y:
    ax.legend_.remove()
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        plt.savefig("{2}/{0}/{1}_bar.pdf".format(folder, name, fig_path), format="pdf")
        plt.savefig("{2}/{0}/{1}_bar.png".format(folder, name, fig_path), format="png")

        fig_leg = plt.figure(figsize=(12, 0.7))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=10)
        ax_leg.axis('off')
        fig_leg.savefig('figs/bar_legend.pdf', format='pdf')

    else:
        plt.show()
    plt.close()


def precision_recall_curve(df,
                           k,
                           folder='figures', name='precison_recall', save=True, reloaded=False):
    fig, ax = plt.subplots(figsize=(4, 3))
    precisions = ["Precision@"+str(x) for x in k]
    recalls = ["Recall@"+str(x) for x in k]

    markers = ['o', '^', 's', '*', 'P', 'h', 'd', '3', 'X', 'v', '+']
    for i in range(len(df)):
        if reloaded:
            precision = [literal_eval(x)[0] for x in df[precisions].iloc[i].tolist()]
            recall = [literal_eval(x)[0] for x in df[recalls].iloc[i].tolist()]
        else:
            precision = [x[0] for x in df[precisions].iloc[i].tolist()]
            recall = [x[0] for x in df[recalls].iloc[i].tolist()]
        ax.plot(recall, precision, label=df['model'].iloc[i], marker=markers[i])

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        plt.savefig("{2}/{0}/{1}_curve.pdf".format(folder, name, fig_path), format="pdf")
        plt.savefig("{2}/{0}/{1}_curve.png".format(folder, name, fig_path), format="png")

        fig_leg = plt.figure(figsize=(12, 0.7))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', ncol=10)
        ax_leg.axis('off')
        fig_leg.savefig('figs/legend.pdf', format='pdf')

    else:
        plt.show()
    plt.close()