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

def accuracy(conf_matrix):
    
    total_questions=sum([sum(row) for row in conf_matrix])
    good_answers=0
    for i in range(len(conf_matrix)):
        good_answers += conf_matrix[i][i]
    accuracy=good_answers/total_questions
    return accuracy

def sensitivity(conf_matrix,):
    ground_truth_pos=sum(conf_matrix[0])
    true_positives=conf_matrix[0][0]
    return true_positives/ground_truth_pos

def specificity(conf_matrix,):
    ground_truth_neg=sum(conf_matrix[1])
    true_negatives=conf_matrix[1][1]
    return true_negatives/ground_truth_neg

def pos_pred_power(conf_matrix, ):
    true_positives=conf_matrix[0][0]
    predicted_positives=sum([row[0] for row in conf_matrix])
    return true_positives/predicted_positives

def neg_pred_power(conf_matrix,):
    true_negatives=conf_matrix[1][1]
    predicted_negatives=sum([row[1] for row in conf_matrix])
    return true_negatives/predicted_negatives

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

def plot_conf_matrix(array, cmap='Blues', filename=None, figsz=(3,3)):

    plt.figure(figsize=figsz)
    df_cm = pd.DataFrame(array, range(len(array)), range(len(array)))
    # plt.figure(figsize=(10,7))
    # sns.set(font_scale=1.4) # for label size
    map=sns.heatmap(df_cm, annot=True,
                fmt=".0f",
                #  annot_kws={"size": 16},
                cbar=False, cmap=cmap,
                linewidths=1, square=True, linecolor='black') 
    map.set_xlabel('Predicted',)

    labels=['Yes','No','Unknown',][:len(array)]
    map.set_xticklabels(labels)
    map.set_yticklabels(labels)
    map.xaxis.set_label_position('top')
    map.xaxis.tick_top()

    map.tick_params(left=False, top=False)
    map.set_ylabel('Truth')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

# bars_cmap = LinearSegmentedColormap.from_list(
#     "white_to_blue",
#     ["#ffffff", '#1F77B4']
# )


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

def raincloud_plot(stats_df, color=None, filename=None,ylims=None, debug=False):
    
    if color is None:
        color='#1F77B4'

    setstuff=[]
    cols=[]
    acc=[]
    for col in stats_df.columns[1:]:
        setstuff+=['ccas']*len(stats_df)
        cols+=[col]*len(stats_df)
        acc+=stats_df[col].values.tolist()
    accdf=pd.DataFrame()
    accdf['set']=setstuff
    accdf['scores']=acc
    accdf['scoretype']=cols
    if debug: 
        print(accdf[accdf['scoretype']=='specificity'])


    plt.figure(figsize=(6,4))
    rc=pt.RainCloud(data=accdf, y='scores',x='scoretype',hue='scoretype',
                bw=0.5, cut=0, orient='v', palette=[color]*5, width_viol=.5, width_box=.3,)
    rc.set_xticklabels(["Accuracy","Sensitivity","Specificity","NPV",'PPV'])

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