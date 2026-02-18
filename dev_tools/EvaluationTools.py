import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(truth_vector, pred_vector):

    matrix=[[0,0,0],
            [0,0,0],
            [0,0,0]]
    
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


def plot_scores(df, score_names,yrange=(0.1,1.03), title='Schmahmann CCAS evaluation by question',
                filename=f'/Users/rm026/Documents/schmahmann/plots/stats_by_question.png'):
    
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
