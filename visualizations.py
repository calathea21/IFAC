import pandas as pd
import plotnine as p9
import numpy as np
import textwrap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def visualize_performances_over_coverages(measure_of_interest="Accuracy"):

    performances = pd.read_excel("recidivism_performance_diff_coverage.xlsx")
    performances['Classifier'] = pd.Categorical(performances['Classifier'], categories=["RF", "NN", "XGB"], ordered=True)
    performances['fairness_weight'].fillna('No Weight', inplace=True)
    performances = performances[(performances['fairness_weight'] == 'No Weight') | (performances['fairness_weight'] == 1.0) | (performances['fairness_weight'] == 0.25)]

    full_coverage_data = performances[performances['method'] == 'No Reject']
    usc_data = performances[performances['method'] == 'PR']
    fsc_data = performances[performances['method'] == 'RR']


    # Set line_style and context
    sns.set_style("whitegrid")

    # Plot using Seaborn
    g = sns.FacetGrid(performances, col="Classifier", hue="method", col_wrap=3)
    g.map_dataframe(sns.lineplot, x="coverage", y=measure_of_interest, style='fairness_weight', markers=True, color='white')

    # Overlay horizontal lines for baseline accuracy
    for ax in g.axes.flat:
        classifier = ax.get_title().split('=')[1].strip()
        baseline_accuracy = full_coverage_data[full_coverage_data['Classifier'] == classifier][measure_of_interest].iloc[0]
        usc_accuracies = usc_data[usc_data['Classifier'] == classifier][measure_of_interest]
        fsc_accuracies_1 = fsc_data[(fsc_data['Classifier'] == classifier) & (fsc_data['fairness_weight'] == 1.0)][measure_of_interest]
        fsc_accuracies_25 = fsc_data[(fsc_data['Classifier'] == classifier) & (fsc_data['fairness_weight'] == 0.25)][
            measure_of_interest]
        coverages = usc_data[usc_data['Classifier'] == classifier]['coverage']
        ax.plot(coverages, usc_accuracies, linestyle='-', color='orange', marker='o', linewidth=2)
        ax.plot(coverages, fsc_accuracies_1, linestyle='-', color='green', marker='o', linewidth=2)
        ax.plot(coverages, fsc_accuracies_25, linestyle='--', color='green', marker='o', linewidth=2)
        ax.axhline(y=baseline_accuracy, linestyle='-', color='grey', label='No Reject', linewidth=2)

    legend_elements = [
        mlines.Line2D([], [], color='grey', linestyle='-', label='Full Coverage'),
        mlines.Line2D([], [], color='orange', linestyle='-', label='USC'),
        mlines.Line2D([], [], color='green', linestyle='-', label='FSC, urw=1.00'),
        mlines.Line2D([], [], color='green', linestyle='--', label='FSC, urw=0.25'),
    ]

    # Add custom legend
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(-0.5, -0.1), ncol=6)
    plt.show()


def wraping_func(text):
    return [textwrap.fill(wraped_text, 15) for wraped_text in text]


def visualize_performance_across_groups_different_classification_types(data, performance_measure_of_interest, title, show_plot = False):
    plot = p9.ggplot(data = data) + \
           p9.geom_col(mapping=p9.aes(x="Classification Type", y=performance_measure_of_interest, fill="Group"), position='dodge') + \
           p9.scale_y_continuous(limits=(0, 1)) + \
           p9.ggtitle(title)

    if show_plot:
        print(plot)

    return plot

def visualize_performance_across_groups_different_classification_types_with_conf_intervals(data, performance_measure_of_interest, title, y_axis_limits, show_plot=False):
    plot = p9.ggplot(data=data, mapping=p9.aes(x="Classification Type", y=performance_measure_of_interest + ' mean', fill="Group")) + \
           p9.geom_col(stat='identity', position='dodge', width=0.86) + \
           p9.geom_errorbar(p9.aes(x='Classification Type', ymin=performance_measure_of_interest + ' ci_low', ymax= performance_measure_of_interest + ' ci_high', fill='Group'),
                         position=p9.position_dodge(0.86), width=0.4) + \
           p9.scale_y_continuous(limits=y_axis_limits) + \
           p9.scale_x_discrete(breaks=data['Classification Type'].unique().tolist(), labels=wraping_func) + \
           p9.theme(panel_grid_minor=p9.element_blank()) + \
           p9.ggtitle(title) + \
           p9.scale_fill_manual(values=["#F8766D", "#B79F00", "#00BA38", "#00BFC4", "#2176FF", "#F564E3"])


    if show_plot:
        print(plot)

    return plot


def visualize_rejection_rate_across_pd_itemsets_with_conf_intervals(data, measure_of_interest, title, y_axis_limits, show_plot=False):
    plot = p9.ggplot(data=data, mapping=p9.aes(x="Classification Type", y=measure_of_interest + ' mean', fill="Group")) + \
           p9.geom_col(stat='identity', position='dodge', width=0.5) + \
           p9.geom_errorbar(p9.aes(x='Classification Type', ymin=measure_of_interest + ' ci_low',
                                   ymax=measure_of_interest + ' ci_high', fill='Group'), position=p9.position_dodge(0.5), width=0.3) + \
           p9.scale_y_continuous(limits=y_axis_limits) + \
           p9.scale_x_discrete(breaks=data['Classification Type'].unique().tolist(), labels=wraping_func) + \
           p9.ggtitle(title)

    if show_plot:
        print(plot)

    return plot


def visualize_fairness_across_different_classification_types_with_conf_intervals(dataframe,fairness_measure_of_interest,title, show_plot=False):
    plot = p9.ggplot(data=dataframe, mapping=p9.aes(x="Classification Type", y=fairness_measure_of_interest + ' mean')) + \
           p9.geom_col(stat='identity', position='dodge', width=0.5) + \
           p9.geom_errorbar(p9.aes(x='Classification Type', ymin=fairness_measure_of_interest + ' ci_low',
                                   ymax=fairness_measure_of_interest + ' ci_high'),
                            position=p9.position_dodge(0.5), width=0.3) + \
           p9.scale_y_continuous(limits=(0, 1)) + \
           p9.scale_x_discrete(breaks=dataframe['Classification Type'].unique().tolist(), labels=wraping_func) + \
           p9.ggtitle(title)

    if show_plot:
        print(plot)

    return plot

#sens_att_combinations come in the form of: [['sex'], ['race'], ['sex', 'race']]
def visualize_performance_measure_for_single_and_intersectional_axis(performance_dataframe, performance_measure_of_interest, path_to_save_figure, y_axis_limits = (0, 1)):
    dataframes_split_by_sens_features = dict(tuple(performance_dataframe.groupby("Sensitive Features")))
    for key, dataframe in dataframes_split_by_sens_features.items():
        plot_title = performance_measure_of_interest + " " + key
        plot = visualize_performance_across_groups_different_classification_types(dataframe, performance_measure_of_interest, title=plot_title)
        plot_path = path_to_save_figure + "\\" + plot_title + ".png"
        p9.ggsave(plot, filename=plot_path)
    return

def visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, performance_measure_of_interest, path_to_save_figure, y_axis_limits=(0, 1)):
    dataframes_split_by_sens_features = dict(tuple(averaged_performances.groupby("Sensitive Features")))
    for key, dataframe in dataframes_split_by_sens_features.items():
        plot_title = performance_measure_of_interest + " " + key
        plot = visualize_performance_across_groups_different_classification_types_with_conf_intervals(dataframe, performance_measure_of_interest, title=plot_title, y_axis_limits=y_axis_limits)
        plot_path = path_to_save_figure + "\\" + plot_title + ".svg"
        p9.ggsave(plot, filename=plot_path)
    return





