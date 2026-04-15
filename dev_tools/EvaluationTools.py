import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(truth_vector, pred_vector, dimension=3):

    if dimension==3:
        matrix=[[0,0,0],
                [0,0,0],
                [0,0,0]]
    elif dimension==2:
        matrix=[[0,0],
                [0,0]]

    for truth, pred in zip(truth_vector,pred_vector):
        if type(truth)==str and truth.lower()=='o':
            truth=0
        if type(pred)==str:
            print(f"Skipping string prediction: {pred}")
            continue
        
        matrix[-1-int(float(truth))][-1-int(float(pred))]+=1

    return np.array(matrix)

def div_check_zeros(num, denom):
    if denom==0 and num==0:
        return 1
    elif denom==0 and num!=0:
        raise ValueError("Denominator is zero, but numerator is not zero.")
    else:
        return num/denom

def accuracy(conf_matrix):
    
    total_questions=sum([sum(row) for row in conf_matrix])
    good_answers=0
    for i in range(len(conf_matrix)):
        good_answers += conf_matrix[i][i]
    accuracy=div_check_zeros(good_answers, total_questions)
    # accuracy=good_answers/total_questions
    return accuracy

def sensitivity(conf_matrix,):
    ground_truth_pos=sum(conf_matrix[0])
    true_positives=conf_matrix[0][0]
    sensitivity=div_check_zeros(true_positives, ground_truth_pos)
    return sensitivity

def specificity(conf_matrix,):
    ground_truth_neg=sum(conf_matrix[1])
    true_negatives=conf_matrix[1][1]
    specificity=div_check_zeros(true_negatives, ground_truth_neg)
    return specificity

def pos_pred_power(conf_matrix, ):
    true_positives=conf_matrix[0][0]
    predicted_positives=sum([row[0] for row in conf_matrix])
    pos_pred_power=div_check_zeros(true_positives, predicted_positives)
    return pos_pred_power

def neg_pred_power(conf_matrix,):
    true_negatives=conf_matrix[1][1]
    predicted_negatives=sum([row[1] for row in conf_matrix])
    neg_pred_power=div_check_zeros(true_negatives, predicted_negatives)
    return neg_pred_power

def combine_no_and_unknown(conf):
    output=np.asarray([conf[0],conf[1]+conf[2]])
    output=np.asarray([output[:,0],output[:,1]+output[:,2]])
    output=output.transpose()
    return output

def ignore_ground_truth_unknown(conf):
    return conf[:-1]

def ignore_unknown(conf):
    return conf[:-1,:-1]


def statistical_df(confusion_matrix_dict,include_accuracy=True):
    
    confusion_matrix_dict['overall']=sum(list(confusion_matrix_dict.values()))

    out_df=pd.DataFrame.from_dict({q:accuracy(matrix) for q, matrix in confusion_matrix_dict.items()}, orient='index', columns=['accuracy'])
    
    if include_accuracy:
        out_df['accuracy']=[accuracy(matrix) for q, matrix in confusion_matrix_dict.items()]

    out_df['specificity']=[specificity(matrix) for q, matrix in confusion_matrix_dict.items()]
    out_df['sensitivity']=[sensitivity(matrix) for q, matrix in confusion_matrix_dict.items()]
    out_df['positive predictive power']=[pos_pred_power(matrix) for q, matrix in confusion_matrix_dict.items()]
    out_df['negative predictive power']=[neg_pred_power(matrix) for q, matrix in confusion_matrix_dict.items()]

    out_df.reset_index(inplace=True, names='question')
    
    return out_df

from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


def plot_conf_matrix(array, labels, cmap='Blues', filename=None, figsz=(3,3),show_pct=False):

    df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
    
    if show_pct:
        total=df_cm.to_numpy().sum()
        
        notes=[]
        for row in array:
            newrow=[]
            for count in row:
                if count==0:
                    newrow.append('0')   
                    continue
                pct=f"{(100*count/total):.2g}"
                new_string=f"{count}\n{pct}%"
                newrow.append(new_string)
            notes.append(newrow)

        annotation=pd.DataFrame(notes, range(len(array)), range(len(array)) )
        formatting=''

    else:
        annotation=True
        formatting=".2f"


    labels=['Yes','No','Unknown',][:len(array)]
    
    htmap=heatmap(df_cm, labels, cmap, annotation=annotation,
                  formatting=formatting, figsz=figsz)
    
    htmap.set_xlabel('Predicted')
    htmap.set_ylabel('Truth')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        
    return htmap
    
    
def heatmap(array, labels,  cmap='Blues', filename=None, annotation=True, formatting=".2f",figsz=(3,3)):
    
    df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
    plt.figure(figsize=figsz)
    htmap=sns.heatmap(df_cm, annot=annotation,
                    # annot_kws={"ha": 'left'},
                    fmt=formatting,
                    cbar=False, cmap=cmap,
                    linewidths=1, square=True, linecolor='black') 

    htmap.set_xticklabels(labels)
    htmap.set_yticklabels(labels)

    htmap.xaxis.set_label_position('top')
    htmap.xaxis.tick_top()
    htmap.tick_params(left=False, top=False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        
    return htmap



def plot_scores(df, score_names,yrange=(0.1,1.03), title=None,
                filename=None):
    
    colors=['Tomato', 'blue', 'green','purple','orange']

    plt.figure(figsize=(6,4))
    markers=['o','s','^', 'D', '>']
    for i, score in enumerate(score_names):
        question_name=df['question'].str.title()
        plt.plot(question_name, df[score], marker=markers[i], color=colors[i])

    plt.title(title)
    plt.ylim(yrange)
    plt.xticks(rotation=45, ha='right')
    plt.legend([score.capitalize() for score in score_names])
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

import ptitprince as pt

def raincloud_plot(stats_df, color=None, figsz=(6,4), filename=None,ylims=None, include_accuracy=True, debug=False,):
    
    if color is None:
        color='#1F77B4'

    setstuff=[]
    cols=[]
    acc=[]
    if include_accuracy:
        columns_to_use=stats_df.columns[1:]
        labels=["Accuracy","Sensitivity","Specificity","NPV",'PPV']
    else:
        columns_to_use=stats_df.columns[2:]
        labels
    for col in columns_to_use:
        setstuff+=['ccas']*len(stats_df)
        cols+=[col]*len(stats_df)
        acc+=stats_df[col].values.tolist()
    accdf=pd.DataFrame()
    accdf['set']=setstuff
    accdf['scores']=acc
    accdf['scoretype']=cols
    if debug: 
        print(accdf[accdf['scoretype']=='specificity'])


    plt.figure(figsize=figsz)
    rc=pt.RainCloud(data=accdf, y='scores',x='scoretype',hue='scoretype',
                bw=0.5, cut=0, orient='v', palette=[color]*5, width_viol=.5, width_box=.3,)
    rc.set_xticklabels(labels)

    rc.set_xlabel('')
    # auto_y_min, auto_y_max=rc.get_ylim()
    # rc.set_ylim((auto_y_min, 1))
    rc.set_ylim(ylims)
    rc.set_ylabel('Score')
    rc.grid(False)
    sns.despine()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


def barplot(data, xlabels,ylabel=None,colors=None, figsz=(4,4), filename=None, ylims=(0,1),ytick_spacing=0.05):
    
    new_df=pd.DataFrame()
    new_df['name']=xlabels
    new_df['value']=data
    if colors==None:
        colors = plt.cm.tab10.colors 
    new_df['color']=colors[:len(new_df)]
    
    plt.figure(figsize=figsz)
    bar1=sns.barplot(data=new_df, y='value',x='name',  hue='color', legend=False)
    # bar2=sns.barplot(x=labels, hue=labels,y=[0,1,0], palette=['#FFFFFF','#FFFFFF', '#FFFFFF',])
    # bar3=sns.barplot(x=labels, hue=labels,y=[accuracy(bars_conf),accuracy(ccas_conf),0], palette=['#FFFFFF',])
    bar1.set_ylabel(ylabel)

    plt.ylim(.8,1)
    # import matplotlib.patches as patches
    # present_patch = patches.Patch(color='#1F77B4', label="Present")
    # absent_patch = patches.Patch(color="#ABCBE0", label="Absent")
    # pending_patch = patches.Patch(color="#9F9F9F", label="Pending")
    bar1.set_xticklabels(xlabels)
    bar1.set_xlabel('')
    bar1.set_yticks(np.arange(ylims[0],ylims[1]+ytick_spacing,ytick_spacing))
    bar1.set_ylim(ylims)
    # bar1.legend(handles=[pending_patch], labels=['Pending'], loc='lower left',bbox_to_anchor=(1, .76), frameon=False)
    sns.despine()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

class PairSuperiorityPlot:
    def __init__(self, stat_array_1, stat_array_2, model1_name="X", model2_name="Y",title=None, out_dir=None,colormap=None,ylim=None):
        """
        Adapted/stolen from Calvin's CircuitPyper plotting functions. 
        Initializes the resampling plot object for visualizing paired delta statistics.
        This class is used to plot the paired delta between a statistic observed for 
        each resampling, and can visualize either bootstraps or permutations.
        Args:
            stat_array_1 (float): The statistical values 
            stat_array_2 (float): The R-squared value for the second region of interest (ROI).
            model1_name (str, optional): Name of the first model. Defaults to "X".
            model2_name (str, optional): Name of the second model. Defaults to "Y".
            stat (str, optional): The statistic name to be plotted. Defaults to "R²".
            out_dir (str, optional): Where to save. 
            observed_stat_array (float): The observed statistical values, where index 0 is the first observation and index 1 is the other.
            method (str): bootstrap | permtuation. If permutation, draws the distribution vertical line at the Delta's value. IF bootstrap, plot at 0.
        """
        self.stat_array_1 = stat_array_1
        self.stat_array_2 = stat_array_2 
        self.delta_array = self._stat_array_1 - self._stat_array_2
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.out_dir = out_dir
        self.title=title
        # self.BLACK = '#211D1E'
        # self.GREY = '#8E8E8E'
        # self.WHITE = "#FFFFFF"
        self.labels=["Accuracy","Sensitivity","Specificity","NPV",'PPV']
        self.markers=['s','o','d','^','*']
        if colormap is None:
            self.colormap=LinearSegmentedColormap.from_list("white_to_blue",["#ffffff", '#1F77B4'])
        else:
            self.colormap=colormap
        # self.colors={'before':colormap(.999),'slope':colormap(0.5),'after':colormap(.999)}
        # self.colors = self.compute_line_colors_abrupt()
        # self.colors=[self.BLACK,self.WHITE]
        self.ylim=ylim
        # self.prepare_out_dir()

    @property
    def stat_array_1(self):
        return self._stat_array_1

    @stat_array_1.setter
    def stat_array_1(self, value):
        value = np.array(value)  # Convert to numpy array
        if hasattr(self, '_stat_array_2') and len(value) != len(self._stat_array_2):
            raise ValueError("stat_array_1 and stat_array_2 must have the same length.")
        self._stat_array_1 = value

    @property
    def stat_array_2(self):
        return self._stat_array_2

    @stat_array_2.setter
    def stat_array_2(self, value):
        value = np.array(value)
        if hasattr(self, '_stat_array_1') and len(value) != len(self._stat_array_1):
            raise ValueError("stat_array_1 and stat_array_2 must have the same length.")
        self._stat_array_2 = value

    # def prepare_out_dir(self):
    #     if self.out_dir is not None:
    #         os.makedirs(self.out_dir, exist_ok=True)

    def plot_paired_slopes(self, ax):

        cm = sns.color_palette("tab10")
        self.plot_resample_dots(ax, self.stat_array_1, 1, cm)
        self.plot_resample_dots(ax, self.stat_array_2, 0, cm, legend=True)

        for i in range(len(self.stat_array_1)):
            ax.plot([0, 1], [self.stat_array_2[i], self.stat_array_1[i]], 
                    color=cm[i],linewidth=2, alpha=0.5)
            
        self.setup_slope_subplot(ax)

    def plot_resample_dots(self, ax, y_vals, x_coord, colormap, legend=False,size=150, edgecolor='white', alpha=0.9, linewidth=.5, zorder=3):
        
        for i, y in enumerate(y_vals):
            artist=ax.scatter([x_coord], [y], color=colormap[i],
                   edgecolors=edgecolor, linewidth=linewidth, 
                   alpha=alpha, s=size, marker=self.markers[i], 
                   zorder=zorder, label=self.labels[i])
        if legend:
            ax.legend(self.labels,loc='lower right',fontsize='medium',frameon=False)

    def setup_slope_subplot(self, ax):
        ax.set_xticks([0, 1])
        ax.set_xticklabels([self.model2_name, self.model1_name])
        ax.set_title(self.title, fontsize=20)
        # ax.set_ylabel(f"", fontsize=20)
        ax.set_xlim(-0.5, 1.5)
        # y_max = max(np.max(self.stat_array_1), np.max(self.stat_array_2))
        # y_min = min(np.max(self.stat_array_1), np.max(self.stat_array_2))
        ax.set_ylim(self.ylim)
        ax.tick_params(labelsize=16)
        sns.despine(ax=ax)

    # def annotate_paired_slopes(self, ax):
    #     t,p = ttest_rel(np.array(self.stat_array_1), np.array(self.stat_array_2))
    #     x_text = 0.05
    #     ha_text = 'left' 
    #     stat_text = f"t = {t:.4f}\np = {p:.4f}"
    #     ax.text(x_text, 0.95, stat_text, ha=ha_text, va='top', fontsize=14, color=self.BLACK, transform=ax.transAxes)

    def draw(self, verbose=True, ax_pair=None, save=True):
        
        abs_limit = np.max(np.abs(self.delta_array))

        fig, axes = plt.subplots(figsize=(4,4))

        self.plot_paired_slopes(axes)
        # Increase the width of the axis lines
        for spine in axes.spines.values():
            spine.set_linewidth(2)

        # Save the figure
        if save and self.out_dir is not None and ax_pair is None:
            # name = f'superiority_plot-{self.model1_name}-vs-{self.model2_name}.svg'
            plt.savefig(self.out_dir, format='svg', bbox_inches='tight')
        if ax_pair is None:
            plt.tight_layout()
            if verbose:
                plt.show()
