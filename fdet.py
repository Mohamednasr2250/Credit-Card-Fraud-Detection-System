
import os

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import (classification_report,confusion_matrix,ConfusionMatrixDisplay,precision_recall_curve,average_precision_score)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



from scipy.stats import describe,skewm,kurtosis,ks_2samp

from scipy.special import expit as scipy_sigmoid





from NNoptcore import (gradient_descent,random_mini_batches,initialize_velocity,update_parameters_with_momentum
                       ,initialize_adam,update_parameters_with_adam,update_lr)



os.makedirs("out",exist_ok=True)




DATA_PATH="data/creditcard.csv"
LAYER_DIMS=[30,64,32,16,1]
NUM_EPOCHS=150
MINI_BATCH_SIZE=256
CLIP_NORM=5.0
PRINT_EVERY=25
OPTIMIZERS=[]







def load_with_spark(csv_path=DATA_PATH):




    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col,count,when,mean,stddev
        from pyspark.ml.stat import Correlation 
        from pyspark.ml.feature import VectorAssembler


        spark=SparkSession.builder \
            .appName("FraudDetection") \
            .master("local[*]") \
            .config("spark.driver.memory","2g") \
            .getOrCreat()
        spark.sparkContext.setLogLevel("ERORR")

        df=spark.read.csv(csv_path,header=True,inferSchema=True)
        

        print()
        print()
        print()



        print()

        df.groupBy("Class").agg(count('*').alias("count")).show()

        total=df.count()
        n_fraud=df.filter(col("class")==1).count()
        fraud_rate=n_fraud/total*100
        print()

        print()
        df.select("Amount").summary("mean","stddev","min","max").show()


        pdf=df.toPandas()
        spark.stop()
        print()

        return pdf
    


    except ImportError:
        print()
        print()
        return pd.read_csv(csv_path)




    except Exception as e:
        print()
        print()
        return pd.read_csv(csv_path)
    








def preprocess_and_analyse(df):

    X=df.drop('class',axis=1).values.astype(np.float32)
    Y=df['class'].values.astype(np.float32).reshape(-1,1)


    scaler=StandardScaler()

    X[:,[0,29]]=scaler.fit_transform(X[:,[0,29]])
    
    print()

    stats=describe(X.flatten())
    print()


    fraud_idx=Y.flatten()==1
    normal_idx=Y.flatten()==0

    ks_scores=[]

    for feat in range(X.shape[1]):
        stat,p=ks_2samp(X[fraud_idx,feat],X[normal_idx,feat])
        ks_scores.append((feat,stat,p))
    ks_scores.sort(key=lambda x:-x[1])
    print()

    for feat,stat,p in ks_scores[:5]:
        col_name=df.columns[feat]
        print()
    print()

    X_tr,X_te,Y_tr,Y_te=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)

    return X_tr.T,X_te.T,Y_tr.T,Y_te.T







def load_sample_data(n=5000,seed=42):
    np.random.seed(seed)
   
    n_fraud=int(n*0.02)
    n_normal=n-n_fraud
   
    x_normal=np.random.randn(30,n_normal)*0.5
    x_fraud=np.random.randn(30,n_fraud)*0.8+1.5
   
    y_normal=np.zeros((1,n_normal))
    y_fraud=np.ones((1,n_fraud))
   
    x=np.concatenate([x_normal,x_fraud],axis=1)
    y=np.concatenate([y_normal,y_fraud],axis=1)
   
   
    perm=np.random.permutation(n)
   
    x,y=x[:,perm],y[:,perm]
    n_te=int(n*0.2)

    return x[:,n_te:],y[:,n_te:],x[:,:n_te],y[:,:n_te]




def train_sklearn_baseline(x_train,y_train,x_test,y_test):


    print()
    
    lr=LogisticRegression(    max_iter=1000,random_state=42,class_weight='balanced',C=0.1)
     
    lr.fit(x_train.T,y_train.flatten())

    y_pred=lr.predict(x_test.T)
    y_pred=lr.predict_proba(x_test.T)[:,1]
    
    print(classification_report
          (y_test.flatten(),y_pred,target_names=['normal','fraud'],zero_division=0))
    auc=roc_auc_score(y_test.flatten(),y_prob)

    print()
    print()


    return{"auc":auc,"predictions":y_pred,"probabilities":y_prob}




"""
def train_with_optimizers():





    return parameters,costs

"""

def evaluate_with_sklearn():
    
    AL,_=forward_propagation(x_test,parameters)
    preds=(AL>0.5).astype(int)

    y_true=y_test.flatten()
    y_pred=preds.flatten()
    y_prob=AL.flatten()


    y_prob_stable=scipy_sigmoid(mp.clip(y_prob*10 -5,-500,500))

    auc=roc_auc_score(y_true,y_prob)

    ap=average_precison_score(y_true,y_prob)

    fpr,tpr,_=roc_curve(y_true,y_prop)

    report=classification_report(y_true,y_pred,target_names=['normal','fraud'],output_dict=True,zero_division=0)


    recall=report['fraud']['recall']
    precision=report['fraud']['perecision']
    f1=report['fraud']['f1-score']


    return{}






#






#