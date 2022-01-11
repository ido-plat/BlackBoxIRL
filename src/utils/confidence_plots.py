import numpy as np
import matplotlib.pyplot as plt


class TransformedConfidencePlot:
    def __init__(self, plot_function, data_transforms):
        self.plot_func = plot_function
        self.transforms = data_transforms

    def __call__(self, confidences, labels, **kwargs):
        if self.transforms is None:
            return self.plot_func(confidences, labels, kwargs)
        transformed_confidences = []
        for confidence in confidences:
            post_tform_confidence = [confidence]
            for transform in self.transforms:
                post_tform_confidence.append(transform(confidence))
            transformed_confidences.append(post_tform_confidence)
        return self.plot_func(transformed_confidences, kwargs)


def plot_distribution(confidences, labels, num_bins=100, label_size=20, save_path=None):
    # if _is_tformed(confidences):
    #     confidences = [item for sublist in confidences for item in sublist]
    #     new_labels = []
    #     for i, conf_array in enumerate(confidences):
    #         new_labels.append(labels[i])
    #         for k in range(1, len(conf_array)):
    #
    #             new_labels.append('T'+str(k)+' '+labels[i])
    bins = np.linspace(0, 1, num_bins)
    for i, confidence in enumerate(confidences):
        plt.hist(confidence, alpha=0.5, bins=bins, label=labels[i], density=True)
    plt.legend()
    plt.xticks(fontsize=label_size)
    plt.xlabel('Confidence', fontsize=label_size + 2)
    # plt.ylim(0, 100)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.clf()
    return confidences


def plot_bar_mean(confidences, labels, font_size=10, agent_color=None, expert_color=None, agent_id=-2, expert_id=-1,
                  save_path=None):

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.xaxis.set_label_text('Confidence', fontsize=font_size + 5)
    bars = ax.barh(labels, [round(conf.mean(), 4) for conf in confidences])
    if agent_color is not None:
        bars[agent_id].set_color(agent_color)
    if expert_color is not None:
        bars[expert_id].set_color(expert_color)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.02, i.get_y() + 0.5, str(round((i.get_width()), 2)), fontsize=font_size,
                 fontweight='bold', color='grey')

    # Add Plot Title
    ax.set_title('Mean Confidence Plot', loc='center', fontsize=font_size+10)
    if save_path:
        plt.savefig(save_path)
        plt.clf()

    else:
        plt.show()
    # plt.close()

def _is_tformed(confidences):
    return isinstance(confidences[0], list)
