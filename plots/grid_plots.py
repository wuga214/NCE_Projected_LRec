import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patches as mpatches
import seaborn as sns
from utils.io import load_yaml
sns.axes_style("white")


def show_samples(images, row, col, image_shape, name="Unknown", save=True, shift=False):
    num_images = row*col
    if shift:
        images = (images+1.)/2.
    fig = plt.figure(figsize=(col, row))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(row, col),
                     axes_pad=0.)
    for i in xrange(num_images):
        im = images[i].reshape(image_shape)
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im)
    plt.axis('off')
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        fig.savefig('{0}/train/grid/{1}.png'.format(fig_path, name), bbox_inches="tight", pad_inches=0, format='png')
    else:
        plt.show()


def latent_distribution_ellipse(means, stds, keep_rate, lim=6, name="Unknown", save=True):
    fig, ax = plt.subplots(figsize=(4, 4))
    patches = []
    m, _ = means.shape
    colors = sns.color_palette("Blues", 10).as_hex()
    plt.axis('equal')
    handles = []
    for i in range(m):
        ellipse = mpatches.Ellipse(means[i], stds[i][0], stds[i][1],
                                   edgecolor=colors[i], lw=3, facecolor='none', label="{:10.4f}".format(keep_rate**(i+1)))
        handles.append(ellipse)
        ax.add_artist(ellipse)

    ax.set(xlim=[-lim, lim], ylim=[-lim, lim])
    #plt.axis('off')
    plt.legend(handles=handles)
    plt.tight_layout()
    if save:
        fig_path = load_yaml('config/global.yml', key='path')['figs']
        fig.savefig('{0}/train/grid/{1}.png'.format(fig_path, name), bbox_inches="tight", pad_inches=0, format='png')
    else:
        plt.show()


