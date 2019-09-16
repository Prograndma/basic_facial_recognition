import numpy as np
from matplotlib import pyplot as pyp
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


class Plotter:
    def __init__(self, stuff_to_plot):
        self.data = stuff_to_plot

    def plot_embedding(self):
        x_train = []
        y_train = []
        for i in range(len(self.data)):
            images = self.data[i].image_vectors
            for j in range(len(images)):
                x_train.append(images[j])
                y_train.append(self.data[i].name)
        x_train = np.concatenate(x_train, 0)

        feat_cols = ['pixel' + str(i) for i in range(x_train.shape[1])]
        df = pd.DataFrame(x_train, columns=feat_cols)
        df['names'] = y_train
        df['label'] = df['names'].apply(lambda x: str(x))

        tsne = TSNE(n_components=2, random_state=0)
        results = tsne.fit_transform(x_train, y_train)

        df['tsne-2d-one'] = results[:, 0]
        df['tsne-2d-two'] = results[:, 1]
        pyp.figure(figsize=(16, 10))
        ax = sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="names",
            palette=sns.color_palette("hls", len(self.data)),
            data=df,
            legend="full",
            alpha=0.3)
        pyp.show(ax)
