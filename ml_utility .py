import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# 1. Fucntion to create piechart 

def create_piechart(data, column):
    """
    Objective
    ---------- 
    Create Pichart for Categorical varaibles present in Pandas Dataframe
    
    parameters
    ----------
    data: this is pandas dataframe
    column: this is column name which is used to create plot
        
    returns
    ----------
    this will show piechart
    
    """
    labels = list(data[column].value_counts().to_dict().keys())
    sizes = list(data[column].value_counts().to_dict().values())
   
    plt.pie(sizes, 
            labels=labels, 
            autopct='%1.2f%%',
            shadow=False, 
            startangle=45)
    
    plt.axis('equal')  
    plt.title("Piechart - {}".format(column))
    plt.show()
    
# 1. Fucntion to check missing data 
    
def missing_data(df):
    """
    Objective
    ----------
    it shows the missing data in each column with 
    total missing values, percentage of missing value and
    its data type in descending order.
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
    
    returns
    ----------
    missing_data: output data frame(pandas dataframe)
    
    """
    
    total = df.isnull().sum().sort_values(ascending=False)
    
    percent = round((df.isnull().sum()/df.isnull().count()  * 100).sort_values(ascending=False),2)
    
    data_type = df.dtypes
    missing_data = pd.concat([total,percent,data_type],
                             axis=1,
                             keys=['Total','Percent','Data_Type']).sort_values("Total", 
                                                                               axis = 0,
                                                                               ascending = False)
    
    return missing_data


def drop_duplicates(df):
    """
    Objective
    ----------
    Drop duplicates rows in data frame except for the first occurrence.
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
        
    returns
    ----------
    dataframe with all unique rows
    """
        
    try:
        dr = df.duplicated().value_counts()[1]
        print("[INFO] Dropping {} duplicates records...".format(dr))
        f_df = df.drop_duplicates(keep="first")
        
        return f_df
    except KeyError:
        print("[INFO] No duplicates records found")
        return df

    
def boxplot(df,width=20,height=200):
    """
    Objective
    ----------
    Draw a box plot to show distributions, skiping all the object variables
    (adjust the width and height to get best possible result)
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
    width: int
        width for box plot
    height: int
        height for box plot
        
    returns
    ----------
    matplotlib Axes
    Returns the Axes object with the plot drawn onto it.   
    """
    sns.set_theme(style="darkgrid")
    
    cols = list(df.select_dtypes(["float64","int64"]).columns)
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(len(cols),figsize=(width,height))
    
    
    for i, col in enumerate(cols):
        sns.boxplot(df[col] , ax = axs[i])
        
        
def feature_scaling(X_train, X_test, method = "StandardScaler", return_df = False):
    """
    Objective
    ----------
    performs normalization or Standardization on input dataset 
    for feature scaling
    
    parameters
    ----------
    X_train: pandas dataframe
        all independent features in dataframe for training 
    
    X_test: pandas dataframe
        all independent features in dataframe for testing

    method : str , options "StandardScaler" or "MinMax" (dfault="StandardScaler")
        type of method to perform feature scaling
        "StandardScaler" is used for standardization
        and "MinMax" is used for Normalization
    
    return_df : bool (defualt=False)
        True will return the output in pandas Dataframe format
        False will return the output in array format

    returns 
    ----------
     X_train_scale, X_test_scale , scale  object
        return sclae data in array or dataframe format
        
    """
    if method == "StandardScaler":
        
        sc = StandardScaler()
        
        if return_df:
        
            # return data frame format
            X_train_scale = pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns)
            X_test_scale = pd.DataFrame(sc.transform(X_test),columns=X_test.columns)

            return X_train_scale , X_test_scale, sc
        else:
            
            # return array format
            X_train_scale =sc.fit_transform(X_train)
            X_test_scale =sc.transform(X_test)
            
            return X_train_scale , X_test_scale, sc
    
    elif method =="MinMax":
        
        mm_scaler = MinMaxScaler()
        
        if return_df:
        
            # return data frame format
            X_train_scale = pd.DataFrame(mm_scaler.fit_transform(X_train),columns=X_train.columns)
            X_test_scale = pd.DataFrame(mm_scaler.transform(X_test),columns=X_test.columns)

            return X_train_scale , X_test_scale, mm_scaler
        else:
            
            # return array format
            X_train_scale =mm_scaler.fit_transform(X_train)
            X_test_scale =mm_scaler.transform(X_test)
            
            return X_train_scale , X_test_scale , mm_scaler
        
        
# Helper function to plot cunfusion matrix and classification report 

def plot_confusion_metrix(y_true, y_pred,classes,
                         normalize=False,
                         title='Confusion Matrix',
                         cmap=plt.cm.Blues):
    """
    Objective
    ----------
    plot confussion matrix, classification report and accuracy score
    
    parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    
    classes : list
        List of labels to index the matrix
        
    title : title for matrix
    cmap : colormap for matrix 
    
    returns 
    ----------
   all accruacy matrix 
    """
    
    
    cm = confusion_matrix(y_true,y_pred)
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, Without Normalisation")

    
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=35)
    plt.yticks(tick_marks,classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() /2.
    
    for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[0])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    print("-----------------------------------------------------")
    print('Classification report')
    print(classification_report(y_true,y_pred))
    
    acc= accuracy_score(y_true,y_pred)
    print("Accuracy of the model: ", acc)

    
    