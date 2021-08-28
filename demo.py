
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
#loading dataset using pandas
import pandas as pd
import numpy as np
import base64
#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import keras_preprocessing 
warnings.filterwarnings('ignore')
from matplotlib import pyplot
#EDA
from collections import Counter
# data preprocessing
from sklearn.preprocessing import StandardScaler
# data splitting
from sklearn.model_selection import train_test_split
# data modeling
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#ensembling
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
import sklearn
from sklearn import datasets, linear_model, svm, neighbors, naive_bayes
from PIL import Image

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: #737378 
    }}
    </style>
    """,
    unsafe_allow_html=True
)

"""
# Diabetes dataset classification using ensemble learning
## Panchanand Jha 
The main objective of this work is to provide better classification model and comparsion of various ensemble learning algorithms. 
In this work various machine learning classifiers such as Bagging, adaptive boosting, extreme gradient boosting, hard voting, soft voting
and keras sequntial model have been used. Finally comparison of the adopted algorithms have been made.  
"""

"""
# Introduction 
The number of features of dataset is called dimention or also known as dataset dimensionality. This Pima dataset consist of eight dimension or features
which are Number of times pregnant, Plasma glucose concentration, Diastolic blood pressure, Triceps skin fold thickness, 2-Hour serum insulin, 
Body mass index, Diabetes pedigree function and Age. This dimension or festures are also known as attributes and feature vector is called an instance. 
The features are further used to create models which is called learning or training. The learned model is known as hypothesis or learner. 
These learners are collectively known as ensemble learners. Ensemble method basically trains multiple learners to classify the same dataset. 
This method is also known as committee-based-learning or muiltiple classifier system. In this method multiple learners known as base learners 
are generated from training data by base learning algorithm. The base learning algorithm can be decision tree, knn, ann etc. If the base learners
are same then it is called homogeneous ensemble model else heterogeneous ensemble model.  

"""


def main():
    st.title("1. Bagging Classifier with decision tree as a base learner")
# importing diabetes dataset

    data = pd.read_csv("Diabeted_Ensemble.csv")
    data1 = st.checkbox('check the box to see DataFrame')

    if data1:
        """
        dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
        The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, 
        based on certain diagnostic measurements included in the dataset. 
        Several constraints were placed on the selection of these instances from a larger database. 
        In particular, all patients here are females at least 21 years old of Pima Indian heritage.
        This dataset consists of several medical variables such as glucose level, number of time pregnat, Body mass index, age, insulin etc. 
        """
        st.write('diabetes dataset', data.head())

# output is class variable and lets rename it  as Outcome
    data.rename(columns={' Class variable': 'Outcome'},
            inplace=True, errors='raise')

    lb = LabelEncoder()
    data['Outcome']=lb.fit_transform(data['Outcome'])

# lets create feature and target dataset
    X = data.drop('Outcome',axis=1)
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

# now we can normalize data using StandardScaler modlue
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# lets check output target unique values 
    labels = st.checkbox('check the box to see count of Labels')

    if labels:
        st.write('Labels', (data['Outcome'].value_counts()))

### here we apply different models 
# first model is bagging classifier with decision tree as a base learner

    """
    # Bagging Algorithm 
    The Bagging is abbreviation of  bootstrap aggregating. To know bagging, let us first understand bootstrap concept.
    The bootstrap method basically divides the dataset into n-number of cases with m-number of identical samples.
    Then separate model with target variable is built on each of samples and that yields n-number of predictions for each sample of dataset. 
    Then mean or average of the individual prediction can be caluculated and used as a final prediction . 
    Bagging adopts bootstrap distribution for creating multiple base learners. The main concept of bagging is to aggregate the output of base learner.
    This aggregation can be voting for classification and averaging for regression. 
    """

    image_bag = Image.open('bagging_algo.jpg')
    st.image(image_bag, caption='Courtsey: Ensemble Methods by Zhi-Hua Zhou. ')


    """

    The main criteria for bagging is Akaiki's information. 
    Akaiki's information criterion is a criterion for comparing the relative predictive information present in a model.
     One common formulation of the criterion is:
                    
                    AIC=ln(S^2 )+2m/N

    where m is the number of parameters in the model,
     S^2 is the sum of the squared residuals (observed-expected values), 
    and N is the number of observations.
    This criterion can be maximized over various values of m by adjusting one of the algorithm parameters. 
    This operation can help to evaluate a trade-off between the fit of the model and model simplicity (i.e., reduction in complexity).
    

    """

    model1 = 'baggingclassifier'
    bag_clf = BaggingClassifier(base_estimator =  DecisionTreeClassifier(), n_estimators = 500,
                            bootstrap = True, n_jobs = 1, random_state = 42)
    st.write('bagging estimators:', bag_clf)

    bag_clf.fit(X_train, y_train)
    bag_predicted = bag_clf.predict(X_test)
    st.write('Predicted', bag_clf.predict(X_test), 'actual', y_test)
    bag_conf_matrix = confusion_matrix(y_test, bag_predicted)
    bag_acc_score = accuracy_score(y_test, bag_predicted)
    st.write('Diabets:1 and Non-Diabets:0')
    cm = st.checkbox('check accuracy and confusion matrix for bagging classifier')
    if cm:
        """
        To evaluate the performance of the machine learning model,
        confusion matrix is used to evaluate the classification model based on predicted and actual values. 
        This matrix compares the actual target values or lables with the predicted using machine learning algorithm.
        It can be generated for binary class or multi class classification i.e. number of target variables. 
        
        These matrix has four important terms namely
        True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN). 
        In this work there are two labels 0(No-Diabetes) and 1(Diabetes). 
        Below is the description of confusion matrix terms:

            – TP: When prediction is Diabetes and actual is Diabetes.
            – TN: when prediction is No-Diabetes and Actual is No-Diabetes.
            – FP: when prediction is No-Diabetes and actual is Diabetes.
            – FN: when prediction is Diabetes and actual is No-Diabetes.

        Using the above mentioned terms, the following parameters are found out, that
        evaluate the performance of any particular machine learning technique.

        – Precision or True Positive Rate: It measures the correctness of the
        classifier. It is the ratio of correctly labeled as positive and total number
        of positive classifications as given 

                Precision = True Positive /  (True Positive + False Positive)

        -Recall: It measures the completeness of the classifier result. 
        It is the ratio of total number of positive labels with total number of truly positive
        represented by.

                Recall = True Positive / (True Positive + False Negative)

        – False positive rate: This false positive rate corresponds to the proportion
        of Diabetes cases that are incorrectly classified as No-Diabetes cases with
        respect to all Diabetes cases given.

                False Positive = False Positive / (False Positive + True Negative)
        
        – F-measure: It is the harmonic mean of recall and precision. It is necessary
        to optimize the classification towards either recall or precision, which can
        influence the final result. The range of this is in between 0 to 1 and it gives
        the precision of classifier..

                F−measure = 2 ∗ (1/(1/Precision)+(1/recall))
    
        – Accuracy: it is commonly used in classification problems. It can be determined as the ratio of correctly classified sample to total number of samples
        by

                Accuracy = True Positive + True Negative / (True Positive + True Negative + False Positive + False Negative)

        """
        st.write("confussion matrix of bagging classifier")
        fig = plt.plot()
        sns.heatmap(bag_conf_matrix, annot=True)
        st.pyplot(fig)
        # st.write(bag_conf_matrix)
        st.write(classification_report(y_test,bag_predicted))
        st.write("Accuracy of baggingclassifier:",bag_acc_score*100)

    bag_roc = st.checkbox('Bagging Receiver Operating Characterstic Curve')
    if bag_roc:
        """
        ROC curve represents the true-positive rate over the false-positive rate.
        The true positive rate is also referred to as sensitivity.



                True Positive Rate = True Positives / (True Positives + False Negatives)

            False Positive Rate = False Positives / (False Positives + True Negatives)

        The false positive rate is also referred to as the inverted specificity where specificity is the total number of true negatives 
        divided by the sum of the number of true negatives and false positives.

                Specificity = True Negatives / (True Negatives + False Positives)

        These equations are further used for each model to generate confusion matrix. 

        The total area under the ROC curve (AUC) is useful for showing graphically the relative predictive power of a model. 
        
        """
       
        bag_false_positive_rate,bag_true_positive_rate,bag_threshold = roc_curve(y_test,bag_predicted)
        sns.set_style('whitegrid')
        plt.figure(figsize=(10,5))
        plt.title('Receiver Operating Characterstic Curve')
        plt.plot(bag_false_positive_rate,bag_true_positive_rate,label='bagging classifier')
        plt.plot([0,1],ls='--')
        plt.plot([0,0],[1,0],c='.5')
        plt.plot([1,1],c='.5')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.legend()
        plt.show()
        st.pyplot()

    st.header("2. XGBoost Classifier")

    """
    # Boosting
    The basic idea behind boosting is to add an outside processing loop around a weak learner to convert into a stronger learner.
     Between boosting iterations, a parameter like a split point in a decision tree is adjusted by an amount proportional to the false-negative error rate. 
     Various forms of boosting algorithms focus on false-negative forces algorithm on resolving errors in classification models (with categorical targets) and numerical errors in regression models (with continuous targets).
    There are many types of boosting algorithms, but the most common are the following:

    • AdaBoost

    • Gradient boosting

    • Stochastic gradient boosting

    •XGBoost

    """

    image = Image.open('general_boosting.jpg')
    st.image(image, caption='Courtsey: Ensemble Methods by Zhi-Hua Zhou. ')

    """
    The major goal of the learner is to generalize the knowledge from training data for future unknwon instances. 
    This unknown instances is also known as testing data. Therefore generalization is nothing but prediction result of the testing dataset.
    But before going through testing data, model need to be validated with ground truth by adjusting the tuning parameters. 
    This process is called validation. 
    ### here is process to find generalization error:
    """
    image2 = Image.open('generalization_error.jpg')
    st.image(image2, caption='Courtsey: Ensemble Methods by Zhi-Hua Zhou. ')
    # model 3 xtreme gradient boosting
    model3 = 'Extreme Gradient Boost'
    xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
                    reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
    st.write(xgb)
    xgb.fit(X_train, y_train)
    xgb_predicted = xgb.predict(X_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
    xgb_acc_score = accuracy_score(y_test, xgb_predicted)

    XGboost_cm = st.checkbox(' Accuracy and Confusion matrix of XGBoost')
    if XGboost_cm:
        st.write("confussion matrix of XGBoost")
        plt.figure()
        sns.heatmap(xgb_conf_matrix, annot=True)
        st.pyplot()
        st.write("Accuracy of XGBoost:",xgb_acc_score*100,'\n')


    st.header("3. Hard Voting Classifier")
    st.write(' In hard voting we have taken three learners KNN, learning perceptron and SVM ')
     # lets use voting classifier start with hard voting
    # Instantiate the learners (classifiers)
    learner_1 = neighbors.KNeighborsClassifier(n_neighbors=5)
    learner_2 = linear_model.Perceptron(tol=1e-2, random_state=0)
    learner_3 = svm.SVC(gamma=0.001)
    # Instantiate the voting classifier
    voting = VotingClassifier([('KNN', learner_1),
                           ('PRC', learner_2),
                           ('SVM', learner_3)])

# Fit classifier with the training data
    voting.fit(X_train, y_train)

# Predict the most voted class
    hard_predictions = voting.predict(X_test)
    hard_conf_matrix = confusion_matrix(y_test, hard_predictions)
    hard_accuracy = accuracy_score(y_test, hard_predictions)
# Accuracy of hard voting
    hard_cm = st.checkbox('Accuracy and Confusion matrix for Hard Voting')
    if hard_cm:
        st.write("confussion matrix of hard voting")
        plt.figure()
        sns.heatmap(hard_conf_matrix, annot=True)
        st.pyplot()
        st.write('Accuracy of Hard Voting:', accuracy_score(y_test, hard_predictions))
        st.write('classification report of XGBoost: ', classification_report(y_test,hard_predictions)) 
#################


    st.header("4. Soft Voting Classifier")

# Soft Voting # 
# Instantiate the learners (classifiers)
    learner_4 = neighbors.KNeighborsClassifier(n_neighbors = 5)
    learner_5 = naive_bayes.GaussianNB()
    learner_6 = svm.SVC(gamma = 0.001, probability = True)

# Instantiate the voting classifier
    voting = VotingClassifier([('KNN', learner_4),
                           ('NB', learner_5),
                           ('SVM', learner_6)],
                            voting = 'soft')

# Fit classifier with the training data
    voting.fit(X_train, y_train)
    learner_4.fit(X_train, y_train)
    learner_5.fit(X_train, y_train)
    learner_6.fit(X_train, y_train)

# Predict the most probable class
    soft_predictions = voting.predict(X_test)
    soft_conf_matrix = confusion_matrix(y_test, soft_predictions)

# Get the base learner predictions
    predictions_4 = learner_4.predict(X_test)
    predictions_5 = learner_5.predict(X_test)
    predictions_6 = learner_6.predict(X_test)
    soft_accuracy = accuracy_score(y_test, soft_predictions)
# Accuracies of base learners
    soft_cm = st.checkbox('Accuracy and Confusion matrix for soft Voting')
    if soft_cm:
        st.write("confussion matrix of soft voting")
        plt.figure()
        sns.heatmap(soft_conf_matrix, annot=True)
        st.pyplot()
        st.write('Accuracy of soft Voting:', accuracy_score(y_test, soft_predictions))
        st.write('classification report of soft voting: ', classification_report(y_test,soft_predictions)) 
#################
        st.write('accuracy of KNN:', accuracy_score(y_test, predictions_4))
        st.write('Accuracy of NB:', accuracy_score(y_test, predictions_5))
        st.write('Accuracy of SVM:', accuracy_score(y_test, predictions_6))
        # Accuracy of Soft voting
        st.write('Accuracy of Soft Voting:', accuracy_score(y_test, soft_predictions))

    st.write('Important features using {xgb.feature_importance_}')
    imp_feature = pd.DataFrame({'Feature': [' Number of times pregnant', ' Plasma glucose concentration',
       ' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin', ' Body mass index',
       ' Diabetes pedigree function', ' Age (years)'], 'Importance': xgb.feature_importances_})
    colors = np.array(["blue","magenta","orange","purple","beige","brown","cyan","yellow"])

    plt.figure(figsize=(10,4))
    plt.title("barplot Represent feature importance ")
    plt.xlabel("importance ")
    plt.ylabel("features")
    plt.barh(imp_feature['Feature'],imp_feature['Importance'],color = colors)
    plt.show()
    st.write(imp_feature)
    st.pyplot()


    boost_roc = st.checkbox('Boosting Receiver Operating Characterstic Curve')
    if boost_roc:
        xgb_false_positive_rate,xgb_true_positive_rate,xgb_threshold = roc_curve(y_test,xgb_predicted)
        bag_false_positive_rate,bag_true_positive_rate,bag_threshold = roc_curve(y_test,bag_predicted)

        sns.set_style('whitegrid')
        plt.figure(figsize=(10,5))
        plt.title('Reciver Operating Characterstic Curve')
        plt.plot(xgb_false_positive_rate,xgb_true_positive_rate,label='Extreme Gradient Boost')
        plt.plot(bag_false_positive_rate,bag_true_positive_rate,label='bagging classifier')


        plt.plot([0,1],ls='--')
        plt.plot([0,0],[1,0],c='.5')
        plt.plot([1,1],c='.5')
        plt.ylabel('True positive rate')
        plt.xlabel('False positive rate')
        plt.legend()
        plt.show()
        st.pyplot()



    st.header('5. Keras Sequential Model')
    """
    Here We have used Keras Sequentila model with four layers and the activation function for dense layer is ReLU and sigmoid.
        
        model.compile(loss='binary_crossentropy',   # since we are predicting 0/1
        optimizer='adam',
        metrics=['accuracy'])
    
    """


    df_d = pd.read_csv("Diabeted_Ensemble.csv")
    # output is class variable and lets rename it  as Outcome
    df_d.rename(columns={' Class variable': 'Outcome'},
            inplace=True, errors='raise')

    lb = LabelEncoder()
    df_d['Outcome']=lb.fit_transform(df_d['Outcome'])
    features = list(df_d.columns.values)
    features.remove('Outcome')
    st.write('features of the dataset', features)
    Xx = df_d[features]
    yy = df_d['Outcome']
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(Xx, yy, test_size=0.25, random_state=0)

   

    X_train_1 = X_train_1.values
    y_train_1 = y_train_1.values
    X_test_1  = X_test_1.values
    y_test_1  = y_test_1.values
    

    NB_EPOCHS = 500  # num of epochs to test for
    BATCH_SIZE = 10
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X_train_11)

    model = Sequential()

# 1st layer: input_dim=8, 12 nodes, RELU
    model.add(Dense(32, input_dim=8,  activation='relu'))
# 2nd layer: 8 nodes, RELU
    model.add(Dense(8,  activation='relu'))
    model.add(Dense(12, activation='relu'))
# output layer: dim=1, activation sigmoid
    model.add(Dense(1,  activation='sigmoid' ))

# Compile the model
    model.compile(loss='binary_crossentropy',   # since we are predicting 0/1
             optimizer='adam',
             metrics=['accuracy'])
    #model.summary()
    
# checkpoint: store the best model
    ckpt_model = 'pima-weights.best.hdf5'
    checkpoint = ModelCheckpoint(ckpt_model, 
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
    callbacks_list = [checkpoint]

    st.write('Starting training...')
# train the model, store the results for plotting
    history = model.fit(X_train_1,
                    y_train_1,
                    epochs=100,
                    validation_data=(X_test_1, y_test_1),
                    batch_size=8,
                    callbacks=callbacks_list,
                    verbose=0)  
# Model accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()
    st.pyplot()


    # Model Losss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()
    st.pyplot()

    # print final accuracy
    scores = model.evaluate(X_test, y_test, verbose=0)
    st.write("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    seq_accuracy = scores[1]
    st.write(seq_accuracy)


    """
    ## Comparison of algorithms 
    """
    comp = pd.DataFrame()
    comp["model"] = ["Bagging", "Boosting", "hard_voting", "soft_voting", "sequantial"]
    comp["accuracy"] = [bag_acc_score, xgb_acc_score, hard_accuracy, soft_accuracy, seq_accuracy ]
    st.dataframe(comp)

    plt.barh(comp['model'],comp['accuracy'])
    plt.show()
    st.pyplot()

    """
    This work adopted multiple classifiers for Pima diabetes dataset as discussed. 
    The best result obatained is 81.82 % accuracy by Bagging classifier with decision tree as a base learner.
    On the other hand Hard voting with KNN, percepton and SVM as a learner yields 80.52% accuracy. Sequential model with four layers 
    yields least accuracy but this can be imporved with deep learning models. 
    To improve the accuracy of model, cross validation with multiple folds is suggested. 

    To improve the accuracy: 

    * data preprocessing and feature engineering can be applied. 
    * finding null or missing values and imputations.
    * Normalization or standardization of dataset.
    * Treatment of outliers ( except the model which searches anomalies instead of common patterns)
    * SVM can handle large variables, In case of hard voting SVM is one of a learner. 
    
    These methods can be adopted to increase the efficiency and accuracy of models. 

    ### References 

    Zhi-Hua Zhou. 2012. Ensemble Methods: Foundations and Algorithms. Chapman and Hall/CRC Press, New York.



  
    """

if __name__ == '__main__':
    main()


   
