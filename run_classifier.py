"""
Script to run classifier for landslide prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_recall_curve,
                             auc, roc_auc_score,roc_curve)
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance
import os
import json
from datetime import date, timedelta
from sklearn.metrics import f1_score
import wandb
import argparse
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import pickle
import random
import time


def get_args():
    parser = argparse.ArgumentParser(description='Running ML Pipeline for Landslide Prediction')
    parser.add_argument('--root_dir', help='Root directory of data')
    parser.add_argument('--results_dir', help='Results directory for runs')
    parser.add_argument('--model', help='ML Model. Currently supports rf, gb, and xgb')
    parser.add_argument('--test_year', help='Test year for study. Supports 2016-2023')
    parser.add_argument('--forecast_model', help='Precipitation Forecast Model Used')
    parser.add_argument('--ensemble_num', help='Ensemble member id used from precipitation forecast model')
    parser.add_argument('--hindcast_model', help='Hindcast precipitation model used')
    parser.add_argument('--experiment_type', help='Type of experiment. no_hindcast, no_forecast refers'
                                                  'to removing those features respectively, full is standard')
    parser.add_argument('--wandb_setting', help='Wandb experiment setting, offline or online')
    parser.add_argument('--test_forecast', help='Model to test forecast for 2023 based on ukmo training')
    parser.add_argument('--parameter_tuning', help='Specify if tuning model hyperparameters', default=None)

    return parser.parse_args()



def daterange(date1, date2):
    date_list = []
    for n in range(int((date2 - date1).days) + 1):
        dt = date1 + timedelta(n)
        date_list.append(dt.strftime("%Y-%m-%d"))
    return date_list


def calc_importance(model, X, save_dir):
    importances = model.feature_importances_
    feature_names = X.columns
    df = pd.DataFrame(np.array([importances]), columns=feature_names, index=['importance'])
    df = df.sort_values(by='importance', axis=1, ascending=False)
    df.to_csv('{}/FI.csv'.format(save_dir))
    return df


def roc_auc(y_test, probabilities, results_directory):
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    rfc_auc = roc_auc_score(y_test, probabilities[:, 1])
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('RFC: ROC AUC=%.3f' % rfc_auc)
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    rfc_fpr, rfc_tpr, _ = roc_curve(y_test, probabilities[:, 1])
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(rfc_fpr, rfc_tpr, marker='.', label='RFC')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.savefig('{}/roc_auc_curve.png'.format(results_directory))


def calc_perm_importance(model, X, y, save_dir):
    permimp = permutation_importance(model, X, y, random_state=0)
    feature_names = X.columns
    # only want summaries, remove importances array
    del permimp['importances']
    # convert to df and sort by importance val
    df = pd.DataFrame(permimp.values(), columns=feature_names, index=['importance', 'stddev'])
    df = df.sort_values(by='importance', axis=1, ascending=False)
    #df.to_csv('{}/PermI.csv'.format(save_dir))
    return df


def plot_importances(imp, perm, save_dir):
    '''
    :param imp: feature importances
    :param perm: permutation importances
    :param dir: save directory
    :return: plot of FI with permutation
    '''
    # Normalize first
    imp = imp / imp.max(axis=1).importance
    perm = perm / perm.max(axis=1).importance
    #import ipdb
    #ipdb.set_trace()

    X_axis = np.arange(len(imp.columns))
    plt.figure()
    plt.rc('font', family='serif')
    plt.bar(X_axis - 0.1, imp.loc['importance'].values, label='Importance')
    plt.bar(X_axis + 0.1, perm.loc['importance'].values, label='Permutation Importance')
    plt.xticks(X_axis, imp, rotation=90)
    plt.title('Feature Importances', fontname='serif')
    plt.xlabel('Feature Name', fontname='serif')
    plt.ylabel('Importance', fontname='serif')
    plt.legend()
    #plt.show()
    plt.savefig('{}/importances.png'.format(save_dir))
    plt.clf()


def get_threshold_precision_and_recall(probabilities, labels, results_dir):
    '''
    Precision and Recall curve for varying threshold
    '''

    # calculate pr curves
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)

    # Save precision, recall, thresholds to csv to plot on same curve/investigate later
    pr_df = pd.DataFrame()
    thresholds = np.append(thresholds, np.nan)
    pr_df['thresholds'] = thresholds
    pr_df['precision'] = precision
    pr_df['recall'] = recall
    pr_df.to_csv('{}/precision_recall.csv'.format(results_dir))

    # calculate fscore
    fscore = (2 * precision * recall) / (precision + recall)
    # Check if there are nans (this is from divide by zero issue)
    nan_ix = np.argwhere(np.isnan(fscore))
    if len(nan_ix) == 0:
        print('No nan values in fscore')
    else:
        fscore = np.delete(fscore, nan_ix)
        precision = np.delete(precision, nan_ix)
        recall = np.delete(recall, nan_ix)

    # Get count of pos, neg events in labels
    print('The number of non-landslide events is: {}'.format(labels['label'].tolist().count(0)))
    print('The number of landslide events is: {}'.format(labels['label'].tolist().count(1)))

    baseline = labels['label'].tolist().count(1)/len(labels)

    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    plt.figure()
    plt.plot(recall, precision, marker='.', label='RFC')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best Threshold: {}'.format(thresholds[ix]))
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.text(0.02, 0.99, 'AUC: {0:.2f}'.format(auc(recall, precision)), ha='left', va='top')
    plt.text(0.02, 0.95, 'AP: {0:.2f}'.format(np.mean(precision)), ha='left', va='top')
    plt.axhline(y=baseline, color='r', linestyle='dashed', label='No skill')
    plt.legend()
    #plt.show()
    plt.savefig('{}/Precision-Recall_Curve.png'.format(results_dir), bbox_inches='tight')
    plt.clf()

    return thresholds[ix]


def calc_confusion_matrix(y_test, y_pred, save_directory):
    """
    Calculates confusion matrix
    """
    CM = confusion_matrix(y_test, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    print('True Positive is {}'.format(TP))
    print('True Negative is {}'.format(TN))
    print('False Positive is {}'.format(FP))
    print('False Negative is {}'.format(FN))

    FP_Rate = FP / (FP + TN)
    TP_Rate = TP / (TP + FN)
    FN_Rate = FN / (FN + TP)
    TN_Rate = TN / (TN + FP)

    with open('{}/confusion_matrix.txt'.format(save_directory), 'w') as f:
        f.write('False positive rate at best threshold is {}'.format(FP_Rate))
        f.write('True positive rate at best threshold is is {}'.format(TP_Rate))
        f.write('False negative rate at best threshold is is {}'.format(FN_Rate))
        f.write('True negative rate at best threshold is is {}'.format(TN_Rate))

    print('False positive rate at best threshold is {}'.format(FP_Rate))
    print('True positive rate at best threshold is is {}'.format(TP_Rate))
    print('False negative rate at best threshold is is {}'.format(FN_Rate))
    print('True negative rate at best threshold is is {}'.format(TN_Rate))


def calc_model_performance(model, labels, probs, xtest, results_dir, wandb_id, model_type):
    '''
    Calculate metrics for model
    Inputs model, labels, preds, test samples and window_size
    :return:
    '''

    # Generate precision-recall curve
    best_thresh = get_threshold_precision_and_recall(probs[:,1], labels, results_dir)

    # Set probs to best_thresh and calculate metrics from there
    # precision_recall_curve takes probability estimates of the positive class as input so filter that
    predictions = (probs[:, 1] >= best_thresh)

    df_pred = pd.DataFrame()
    preds_1_and_0 = predictions*1
    df_pred['prediction'] = preds_1_and_0

    # Test set accuracy
    print('Test set accuracy at threshold {}: {}'.format(best_thresh, accuracy_score(labels, predictions)))
    print('Test set F1 at threshold {}: {}'.format(best_thresh, f1_score(labels, preds_1_and_0)))

    with open('{}/model_testing_results.txt'.format(results_dir), 'w') as f:
        f.write('Test set accuracy at threshold {}: {}'.format(best_thresh, accuracy_score(labels, predictions)))
        f.write('Test set F1 at threshold {}: {}'.format(best_thresh, f1_score(labels, preds_1_and_0)))

    wandb_id.log({
        'Test set Accuracy': accuracy_score(labels, predictions),
        'Test set F1': f1_score(labels, preds_1_and_0)
    })

    # Calculate confusion matrix
    calc_confusion_matrix(labels, predictions, results_dir)

    # Calculate importances
    if model_type == 'rf':
        importances = calc_importance(model, xtest, results_dir)
        perm_importances = calc_perm_importance(model, xtest, labels, results_dir)
        plot_importances(importances, perm_importances, results_dir)

    # Calc precision recall
    print('--------------------')
    report = classification_report(labels, predictions, output_dict=True)
    print(report)

    # Save Classification Report
    pd.DataFrame(report).transpose().to_csv('{}/Test_classification_report.csv'.format(results_dir))


def get_class_balance(train, test, results_dir):
    '''
    Calculate the class balance for the split of the various
    '''
    class_balance_dict = {}

    class_balance_dict['train_samples'] = len(train)
    class_balance_dict['train_samples_positive'] = sum(train)
    class_balance_dict['train_samples_negative'] = len(train) - sum(train)

    class_balance_dict['test_samples'] = len(test)
    class_balance_dict['test_samples_positive'] = sum(test)
    class_balance_dict['test_samples_negative'] = len(test) - sum(test)

    with open('{}/class_balance.csv'.format(results_dir), "w") as fp:
        json.dump(class_balance_dict, fp)  # encode dict into JSON


def get_monsoon_season(year):
    """
    Get the monsoon season dates (April to end of Sept)
    """
    start = date(year, 4, 1)
    end = date(year, 10, 31)
    monsoon_dates = daterange(start, end)
    return monsoon_dates


def load_data(test_year, data_dir, experiment_type):
    """
    Function to load in train, test, validation (for param tuning) datasets
    Specify test year, all other years used for training
    """
    year_split_dict = {
        '2017': ['2016'],
        '2018': ['2016', '2017'],
        '2019': ['2016', '2017', '2018'],
        '2020': ['2016', '2017', '2018', '2019'],
        '2021': ['2016', '2017', '2018', '2019', '2020'],
        '2022': ['2016', '2017', '2018', '2019', '2020', '2021'],
        '2023': ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
    }

    if test_year == '2024':
        # Training the model on all the data to save
        train_years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    else:
        train_years = year_split_dict[test_year]

    df_train_list = []
    monsoon_train_list = []
    for y in train_years:
        df = pd.read_csv('{}/{}_windowsize14_district.csv'.format(data_dir, y))
        df_train_list.append(df)
        monsoon_train = daterange(date(int(y), 4, 1), date(int(y), 10, 31))
        monsoon_train_list.append(monsoon_train[:])

    # Split into monsoon season
    monsoon_train_list = [x for xs in monsoon_train_list for x in xs]
    df_train = pd.concat(df_train_list)
    monsoon_train = df_train[df_train['date'].isin(monsoon_train_list)]

    if test_year == '2024':
        y_train = monsoon_train['label']
        X_train = monsoon_train.drop(columns=['label', 'Unnamed: 0'])
        y_test = None
        X_test = None
        if 'Unnamed: 0.1' in X_train.columns:
            X_train = X_train.drop(columns=['Unnamed: 0.1'])
        return X_train, y_train, X_test, y_test


    # Split into monsoon season test set
    df_test = pd.read_csv('{}/{}_windowsize14_district.csv'.format(data_dir, test_year))
    monsoon_test_list = daterange(date(int(test_year), 4, 1), date(int(test_year), 10, 31))
    monsoon_test = df_test[df_test['date'].isin(monsoon_test_list)]

    df_test = shuffle(monsoon_test)
    df_test = df_test.dropna()
    y_test = df_test['label']

    unwanted_cols = ['label', 'Unnamed: 0', 'Unnamed: 0.1']
    X_test = df_test.drop(columns=unwanted_cols, errors='ignore')

    if os.path.exists('{}/train_data_2016-2022.csv'.format(data_dir)):
        train = pd.read_csv('{}/train_data_2016-2022.csv'.format(data_dir))
        val = pd.read_csv('{}/val_data_2016-2022.csv'.format(data_dir))
        X_train = train.drop(columns=['label'])
        y_train = train[['label']]
        X_val = val.drop(columns=['label'])
        y_val = val[['label']]
        return X_train, y_train, X_test, y_test, X_val, y_val
    else:
        # Shuffle
        df_train = shuffle(monsoon_train)

        # Drop Nans
        df_train = df_train.dropna().reset_index()
        df_train = df_train.drop(columns=['index'])

        # Split into label and features, preserve date and location
        ytrain = df_train['label']

        # Drop any columns we don't want
        Xtrain = df_train.drop(columns=unwanted_cols, errors='ignore')

        # Split into train and validation for parameter tuning, use 15% for validation
        X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain, test_size=0.15,
                                                          random_state=random.randint(1,1000), stratify=ytrain)

        train_data = pd.concat([X_train, y_train])
        train_data = train_data.rename(columns={0:'label'})
        val_data = pd.concat([X_val, y_val])
        val_data = val_data.rename(columns={0: 'label'})

        train_data.to_csv('{}/train_data_2016-2022.csv'.format(data_dir), index=False)
        val_data.to_csv('{}/val_data_2016-2022.csv'.format(data_dir), index=False)

    if experiment_type == 'no_hindcast':
        X_train = X_train.drop(X_train.filter(regex='tminus').columns, axis=1)
        X_train = X_train.drop(columns=['mean_precip_rate', 'max_precip_rate', 'min_precip_rate',
                                        'total_cumulative_precipitation'])
        X_test = X_test.drop(columns=['mean_precip_rate', 'max_precip_rate', 'min_precip_rate',
                                        'total_cumulative_precipitation'])
        X_test = X_test.drop(X_test.filter(regex='tminus').columns, axis=1)
    if experiment_type == 'no_forecast':
        X_train = X_train.drop(X_train.filter(regex='ens').columns, axis=1)
        X_test = X_test.drop(X_test.filter(regex='ens').columns, axis=1)

    return X_train, y_train, X_test, y_test, X_val, y_val


def run_rf(data_dir, Xtrain, ytrain, Xtest, ytest, Xval, yval, results_dir, wandb_exp, model_type, test_year, tuning):
    """
    Experiment for train and test set from all years, all regions
    """

    # Preserve date and location information for output
    info_cols = ['date', 'district']
    train_info = Xtrain[info_cols]
    X_train = Xtrain.drop(columns=info_cols)

    X_val = Xval.drop(columns=info_cols)

    if test_year == '2024':
        forest = None   # putting this here for now but will update
        X_train['label'] = ytrain
        X_train = X_train.dropna()
        X_train = X_train.reset_index()
        X_train = shuffle(X_train)
        ytrain=X_train['label']
        X_train = X_train.drop(columns=['label', 'index'])
        print('Fitting model on all data including 2023 and returning')
        # Fit model and save
        forest.fit(X_train, ytrain)
        # Save the model to file
        joblib.dump(forest, '{}/rf_model.joblib'.format(results_dir))

        # saving as pickle too
        with open("{}/rf_model.pkl".format(results_dir), "wb") as file:
            pickle.dump(forest, file)
        return

    test_info = Xtest[info_cols]
    Xtest = Xtest.drop(columns=info_cols)

    if not tuning:
        if not os.path.exists('{}/best_model.pkl'.format(data_dir)):
            # Create an instance of Random Forest
            forest = RandomForestClassifier(criterion='gini',
                                            random_state=random.randint(1, 1000),
                                            n_estimators=200,
                                            n_jobs=-1,
                                            class_weight='balanced')
        else:
            with open('{}/best_model.pkl'.format(data_dir), 'rb') as f:
                forest = pickle.load(f)
        print('No model hyperparameter tuning')
        # Concat the train and validation sets together to train on the entire available dataset
        X_train = pd.concat([X_train, X_val])
        ytrain = pd.concat([ytrain, y_val])

        # Fit the model
        print('Fitting model...')
        forest.fit(X_train, ytrain)
    else:
        # Tune the model
        model_state = 'tuned_best_params'
        print('Tuning the model')
        param_grid = {
            'bootstrap': [True],
            'max_depth': [3, 5, 7],
            'min_samples_split': [4, 6, 8],
            'n_estimators': [100, 200, 300]
        }

        tune_list = []
        for i in range(3):
            print('Tuning for iteration: {}'.format(i))
            # Create an instance of Random Forest
            forest = RandomForestClassifier(criterion='gini',
                                            random_state=random.randint(1, 1000),
                                            n_estimators=200,
                                            n_jobs=-1,
                                            class_weight='balanced')
            # grid search cv
            rf_cv = GridSearchCV(estimator=forest,
                                 param_grid=param_grid,
                                 scoring=['f1'],
                                 refit='f1',
                                 cv=3,
                                 n_jobs=-1)

            # Fit the grid search to the data
            print('Running grid search cv on training set...')
            start_time = time.time()
            rf_cv.fit(X_train, y_train)
            print("--- %s seconds to hyperparameter tune ---" % (time.time() - start_time))

            # Set model to the best estimator from grid search
            best_forest = rf_cv.best_estimator_

            # Prediction on validation set with best forest
            tuned_probs = best_forest.predict_proba(X_val)
            accuracy = best_forest.score(X_val, y_val)
            f1 = f1_score(y_val, (tuned_probs[:,1] >= 0.5)*1)

            forest_dict = rf_cv.best_params_
            forest_dict['accuracy'] = accuracy
            forest_dict['f1'] = f1
            forest_dict['best_model'] = best_forest
            tune_list.append(forest_dict)

        # Get max F1 and use this as our best model
        max_dict = max(tune_list, key=lambda x: x['f1'])
        print('Max model stats after parameter tuning is: {}'.format(max_dict))
        # Save the model to file
        joblib.dump(max_dict['best_model'], '{}/rf_model.joblib'.format(results))
        # saving as pickle too
        with open("{}/rf_model.pkl".format(results), "wb") as file:
            pickle.dump(max_dict['best_model'], file)
        forest = max_dict['best_model']

    '''
    scoring = ['accuracy', 'f1']
    cv_scoring = cross_validate(forest, Xtrain, ytrain, scoring=scoring, cv=5)
    with open('{}/model_training_results.txt'.format(results_dir), 'w') as f:
        f.write('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
        f.write('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
        f.write('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
        f.write('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))
    '''

    # Measure model performance on test set
    print('Evaluating model...')
    probs = forest.predict_proba(Xtest)

    acc_test = forest.score(Xtest, ytest)
    f1_test = f1_score(ytest, (probs[:, 1] >= 0.5) * 1)

    # Re-indexing so we can put it all in a df
    test_info = test_info.reset_index().drop(columns=['index'])
    ytest = ytest.reset_index().drop(columns=['index'])

    accuracy = forest.score(Xtest, ytest)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    df_probs = pd.DataFrame()
    df_probs['model soft predictions'] = probs[:,1]
    df_probs['groundtruth'] = ytest
    df_probs['date'] = test_info['date']
    df_probs['district'] = test_info['district']
    df_probs.to_csv('{}/predictions_and_groundtruth.csv'.format(results))

    # Plot roc_auc curve
    roc_auc(ytest, probs, results)

    # Calculate Model Performance
    calc_model_performance(forest, ytest, probs, Xtest, results, wandb_exp, model_type)

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'accuracy': acc_test,
        'F1 at 0.5': f1_test,
        'roc': wandb.plot.roc_curve(ytest, probs, classes_to_plot=[1]),
        'pr': wandb.plot.pr_curve(ytest, probs, classes_to_plot=[1]),
        'Model State': model_state
    })


def run_gb(data_dir, Xtrain, ytrain, Xtest, ytest, Xval, yval, results_dir, wandb_exp, model_type, test_year, tuning):
    """
    Experiment for train and test set from all years, all regions
    """
    # Preserve date and location information for output
    info_cols = ['date', 'district']
    train_info = Xtrain[info_cols]
    X_train = Xtrain.drop(columns=info_cols)

    X_val = Xval.drop(columns=info_cols)

    if test_year == '2024':
        clf = None  # putting this here for now but will update
        X_train['label'] = ytrain
        X_train = X_train.dropna()
        X_train = X_train.reset_index()
        X_train = shuffle(X_train)
        ytrain = X_train['label']
        X_train = X_train.drop(columns=['label', 'index'])
        print('Fitting model on all data including 2023 and returning')
        # Fit model and save
        clf.fit(X_train, ytrain)
        # Save the model to file
        joblib.dump(clf, '{}/gb_model.joblib'.format(results_dir))

        # saving as pickle too
        with open("{}/gb_model.pkl".format(results_dir), "wb") as file:
            pickle.dump(clf, file)
        return

    test_info = Xtest[info_cols]
    Xtest = Xtest.drop(columns=info_cols)

    if not tuning:
        if not os.path.exists('{}/best_model.pkl'.format(data_dir)):
            # Create an instance of XGB
            clf = GradientBoostingClassifier(random_state=random.randint(1, 1000))
        else:
            with open('{}/best_model.pkl'.format(data_dir), 'rb') as f:
                clf = pickle.load(f)
        print('No model hyperparameter tuning')
        # Concat the train and validation sets together to train on the entire available dataset
        X_train = pd.concat([X_train, X_val])
        ytrain = pd.concat([ytrain, y_val])
        # Fit the model
        print('Fitting model...')
        clf.fit(X_train, ytrain)
    else:
        # Tune the model
        model_state = 'tuned_best_params'
        print('Tuning the model')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 3, 5],
        }

        tune_list = []
        for i in range(3):
            print('Tuning for iteration: {}'.format(i))
            # Create an instance of  clf
            clf = GradientBoostingClassifier(random_state=random.randint(1, 1000))
            # grid search cv
            clf_cv = GridSearchCV(estimator=clf,
                                  param_grid=param_grid,
                                  scoring=['f1'],
                                  refit='f1',
                                  cv=3,
                                  n_jobs=-1)

            # Fit the grid search to the data
            print('Running grid search cv on training set...')
            start_time = time.time()
            clf_cv.fit(X_train, y_train)
            print("--- %s seconds to hyperparameter tune ---" % (time.time() - start_time))

            # Set model to the best estimator from grid search
            best_clf = clf_cv.best_estimator_

            # Prediction on validation set with best XGB
            tuned_probs = best_clf.predict_proba(X_val)
            accuracy = best_clf.score(X_val, y_val)
            f1 = f1_score(y_val, (tuned_probs[:, 1] >= 0.5) * 1)

            clf_dict = clf_cv.best_params_
            clf_dict['accuracy'] = accuracy
            clf_dict['f1'] = f1
            clf_dict['best_model'] = best_clf
            tune_list.append(clf_dict)

        # Get max F1 and use this as our best model
        max_dict = max(tune_list, key=lambda x: x['f1'])
        print('Max model stats after parameter tuning is: {}'.format(max_dict))
        # Save the model to file
        joblib.dump(max_dict['best_model'], '{}/gb_model.joblib'.format(results))
        # saving as pickle too
        with open("{}/gb_model.pkl".format(results), "wb") as file:
            pickle.dump(max_dict['best_model'], file)
        clf = max_dict['best_model']

    '''
    scoring = ['accuracy', 'f1']
    cv_scoring = cross_validate(forest, Xtrain, ytrain, scoring=scoring, cv=5)
    with open('{}/model_training_results.txt'.format(results_dir), 'w') as f:
        f.write('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
        f.write('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
        f.write('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
        f.write('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))
    '''

    # Measure model performance on test set
    print('Evaluating model...')
    probs = clf.predict_proba(Xtest)

    acc_test = clf.score(Xtest, ytest)
    f1_test = f1_score(ytest, (probs[:, 1] >= 0.5) * 1)

    # Re-indexing so we can put it all in a df
    test_info = test_info.reset_index().drop(columns=['index'])
    ytest = ytest.reset_index().drop(columns=['index'])

    accuracy = clf.score(Xtest, ytest)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    df_probs = pd.DataFrame()
    df_probs['model soft predictions'] = probs[:, 1]
    df_probs['groundtruth'] = ytest
    df_probs['date'] = test_info['date']
    df_probs['district'] = test_info['district']
    df_probs.to_csv('{}/predictions_and_groundtruth.csv'.format(results))

    # Plot roc_auc curve
    roc_auc(ytest, probs, results)

    # Calculate Model Performance
    calc_model_performance(clf, ytest, probs, Xtest, results, wandb_exp, model_type)

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'accuracy': acc_test,
        'F1 at 0.5': f1_test,
        'roc': wandb.plot.roc_curve(ytest, probs, classes_to_plot=[1]),
        'pr': wandb.plot.pr_curve(ytest, probs, classes_to_plot=[1]),
        'Model State': model_state
    })


def run_xgb(data_dir, Xtrain, ytrain, Xtest, ytest, Xval, yval, results_dir, wandb_exp, model_type, test_year, tuning):
    """
    Experiment for train and test set from all years, all regions
    """

    # Preserve date and location information for output
    info_cols = ['date', 'district']
    train_info = Xtrain[info_cols]
    X_train = Xtrain.drop(columns=info_cols)

    X_val = Xval.drop(columns=info_cols)

    if test_year == '2024':
        clf = None  # putting this here for now but will update
        X_train['label'] = ytrain
        X_train = X_train.dropna()
        X_train = X_train.reset_index()
        X_train = shuffle(X_train)
        ytrain = X_train['label']
        X_train = X_train.drop(columns=['label', 'index'])
        print('Fitting model on all data including 2023 and returning')
        # Fit model and save
        clf.fit(X_train, ytrain)
        # Save the model to file
        joblib.dump(clf, '{}/xgb_model.joblib'.format(results_dir))

        # saving as pickle too
        with open("{}/xgb_model.pkl".format(results_dir), "wb") as file:
            pickle.dump(clf, file)
        return

    test_info = Xtest[info_cols]
    Xtest = Xtest.drop(columns=info_cols)

    if not tuning:
        if not os.path.exists('{}/best_model.pkl'.format(data_dir)):
            # Create an instance of XGB
            clf = XGBClassifier(random_state=random.randint(1, 1000), eta=0.05, max_depth=3, min_child_weight=0,
                                     max_delta_step=2, subsample=1, tree_method='approx')
        else:
            with open('{}/best_model.pkl'.format(data_dir), 'rb') as f:
                clf = pickle.load(f)
        print('No model hyperparameter tuning')
        # Concat the train and validation sets together to train on the entire available dataset
        X_train = pd.concat([X_train, X_val])
        ytrain = pd.concat([ytrain, y_val])
        # Fit the model
        print('Fitting model...')
        clf.fit(X_train, ytrain)
    else:
        # Tune the model
        model_state = 'tuned_best_params'
        print('Tuning the model')
        param_grid = {
            'eta': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [0, 1, 2],
        }

        tune_list = []
        for i in range(3):
            print('Tuning for iteration: {}'.format(i))
            # Create an instance of  clf
            clf = XGBClassifier(random_state=random.randint(1, 1000), eta=0.05, max_depth=3, min_child_weight=0,
                                     max_delta_step=2, subsample=1, tree_method='approx')
            # grid search cv
            clf_cv = GridSearchCV(estimator=clf,
                                 param_grid=param_grid,
                                 scoring=['f1'],
                                 refit='f1',
                                 cv=3,
                                 n_jobs=-1)

            # Fit the grid search to the data
            print('Running grid search cv on training set...')
            start_time = time.time()
            clf_cv.fit(X_train, y_train)
            print("--- %s seconds to hyperparameter tune ---" % (time.time() - start_time))

            # Set model to the best estimator from grid search
            best_clf = clf_cv.best_estimator_

            # Prediction on validation set with best XGB
            tuned_probs = best_clf.predict_proba(X_val)
            accuracy = best_clf.score(X_val, y_val)
            f1 = f1_score(y_val, (tuned_probs[:, 1] >= 0.5) * 1)

            clf_dict = clf_cv.best_params_
            clf_dict['accuracy'] = accuracy
            clf_dict['f1'] = f1
            clf_dict['best_model'] = best_clf
            tune_list.append(clf_dict)

        # Get max F1 and use this as our best model
        max_dict = max(tune_list, key=lambda x: x['f1'])
        print('Max model stats after parameter tuning is: {}'.format(max_dict))
        # Save the model to file
        joblib.dump(max_dict['best_model'], '{}/xgb_model.joblib'.format(results))
        # saving as pickle too
        with open("{}/xgb_model.pkl".format(results), "wb") as file:
            pickle.dump(max_dict['best_model'], file)
        clf = max_dict['best_model']

    '''
    scoring = ['accuracy', 'f1']
    cv_scoring = cross_validate(forest, Xtrain, ytrain, scoring=scoring, cv=5)
    with open('{}/model_training_results.txt'.format(results_dir), 'w') as f:
        f.write('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
        f.write('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
        f.write('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
        f.write('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))
    '''

    # Measure model performance on test set
    print('Evaluating model...')
    probs = clf.predict_proba(Xtest)

    acc_test = clf.score(Xtest, ytest)
    f1_test = f1_score(ytest, (probs[:, 1] >= 0.5) * 1)

    # Re-indexing so we can put it all in a df
    test_info = test_info.reset_index().drop(columns=['index'])
    ytest = ytest.reset_index().drop(columns=['index'])

    accuracy = clf.score(Xtest, ytest)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    df_probs = pd.DataFrame()
    df_probs['model soft predictions'] = probs[:, 1]
    df_probs['groundtruth'] = ytest
    df_probs['date'] = test_info['date']
    df_probs['district'] = test_info['district']
    df_probs.to_csv('{}/predictions_and_groundtruth.csv'.format(results))

    # Plot roc_auc curve
    roc_auc(ytest, probs, results)

    # Calculate Model Performance
    calc_model_performance(clf, ytest, probs, Xtest, results, wandb_exp, model_type)

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'accuracy': acc_test,
        'F1 at 0.5': f1_test,
        'roc': wandb.plot.roc_curve(ytest, probs, classes_to_plot=[1]),
        'pr': wandb.plot.pr_curve(ytest, probs, classes_to_plot=[1]),
        'Model State': model_state
    })


def run_mlp(data_dir, Xtrain, ytrain, Xtest, ytest, Xval, yval, results_dir, wandb_exp, model_type, test_year, tuning):
    """
    Experiment for train and test set from all years, all regions
    """

    # Preserve date and location information for output
    info_cols = ['date', 'district']
    train_info = Xtrain[info_cols]
    X_train = Xtrain.drop(columns=info_cols)

    X_val = Xval.drop(columns=info_cols)

    if test_year == '2024':
        clf = None  # putting this here for now but will update
        X_train['label'] = ytrain
        X_train = X_train.dropna()
        X_train = X_train.reset_index()
        X_train = shuffle(X_train)
        ytrain = X_train['label']
        X_train = X_train.drop(columns=['label', 'index'])
        print('Fitting model on all data including 2023 and returning')
        # Fit model and save
        clf.fit(X_train, ytrain)
        # Save the model to file
        joblib.dump(clf, '{}/mlp_model.joblib'.format(results_dir))

        # saving as pickle too
        with open("{}/mlp__model.pkl".format(results_dir), "wb") as file:
            pickle.dump(clf, file)
        return

    test_info = Xtest[info_cols]
    Xtest = Xtest.drop(columns=info_cols)

    if not tuning:
        if not os.path.exists('{}/best_model.pkl'.format(data_dir)):
            # Create an instance of XGB
            clf = MLPClassifier(random_state=random.randint(1, 1000))
        else:
            with open('{}/best_model.pkl'.format(data_dir), 'rb') as f:
                clf = pickle.load(f)
        print('No model hyperparameter tuning')
        # Fit the model
        print('Fitting model...')
        clf.fit(X_train, ytrain)
    else:
        # Tune the model
        model_state = 'tuned_best_params'
        print('Tuning the model')
        param_grid = {
            'hidden_layer_sizes': [(100,), (150,), (200,)],
            'activation': ['logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
        }

        tune_list = []
        for i in range(3):
            print('Tuning for iteration: {}'.format(i))
            # Create an instance of  clf
            clf = MLPClassifier(random_state=random.randint(1, 1000))
            # grid search cv
            clf_cv = GridSearchCV(estimator=clf,
                                  param_grid=param_grid,
                                  scoring=['f1'],
                                  refit='f1',
                                  cv=3,
                                  n_jobs=-1)

            # Fit the grid search to the data
            print('Running grid search cv on training set...')
            start_time = time.time()
            clf_cv.fit(X_train, y_train)
            print("--- %s seconds to hyperparameter tune ---" % (time.time() - start_time))

            # Set model to the best estimator from grid search
            best_clf = clf_cv.best_estimator_

            # Prediction on validation set with best MLP
            tuned_probs = best_clf.predict_proba(X_val)
            accuracy = best_clf.score(X_val, y_val)
            f1 = f1_score(y_val, (tuned_probs[:, 1] >= 0.5) * 1)

            clf_dict = clf_cv.best_params_
            clf_dict['accuracy'] = accuracy
            clf_dict['f1'] = f1
            clf_dict['best_model'] = best_clf
            tune_list.append(clf_dict)

        # Get max F1 and use this as our best model
        max_dict = max(tune_list, key=lambda x: x['f1'])
        print('Max model stats after parameter tuning is: {}'.format(max_dict))
        # Save the model to file
        joblib.dump(max_dict['best_model'], '{}/mlp_model.joblib'.format(results))
        # saving as pickle too
        with open("{}/mlp_model.pkl".format(results), "wb") as file:
            pickle.dump(max_dict['best_model'], file)
        clf = max_dict['best_model']

    '''
    scoring = ['accuracy', 'f1']
    cv_scoring = cross_validate(forest, Xtrain, ytrain, scoring=scoring, cv=5)
    with open('{}/model_training_results.txt'.format(results_dir), 'w') as f:
        f.write('Accuracy scores for each fold are: {}'.format(cv_scoring['test_accuracy']))
        f.write('Average accuracy is: {}'.format(cv_scoring['test_accuracy'].mean()))
        f.write('F1 scores for each fold are: {}'.format(cv_scoring['test_f1']))
        f.write('Average F1 is: {}'.format(cv_scoring['test_f1'].mean()))
    '''

    # Measure model performance on test set
    print('Evaluating model...')
    probs = clf.predict_proba(Xtest)

    acc_test = clf.score(Xtest, ytest)
    f1_test = f1_score(ytest, (probs[:, 1] >= 0.5) * 1)

    # Re-indexing so we can put it all in a df
    test_info = test_info.reset_index().drop(columns=['index'])
    ytest = ytest.reset_index().drop(columns=['index'])

    accuracy = clf.score(Xtest, ytest)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    df_probs = pd.DataFrame()
    df_probs['model soft predictions'] = probs[:, 1]
    df_probs['groundtruth'] = ytest
    df_probs['date'] = test_info['date']
    df_probs['district'] = test_info['district']
    df_probs.to_csv('{}/predictions_and_groundtruth.csv'.format(results))

    # Plot roc_auc curve
    roc_auc(ytest, probs, results)

    # Calculate Model Performance
    calc_model_performance(clf, ytest, probs, Xtest, results, wandb_exp, model_type)

    # Logging results to wandb
    # --- Logging metrics
    wandb_exp.log({
        'accuracy': acc_test,
        'F1 at 0.5': f1_test,
        'roc': wandb.plot.roc_curve(ytest, probs, classes_to_plot=[1]),
        'pr': wandb.plot.pr_curve(ytest, probs, classes_to_plot=[1]),
        'Model State': model_state
    })

def run_trained_ukmo(root_directory, results_dir, wandb_exp, model_type, forecast_model_testing):
    """
    Run the ukmo model on the data from another forecast model from 2023 to test
    performance
    :param: forecast_model_testing: the model we are interested in querying for the forecast daat
    """
    rf_model = joblib.load('{}/Results/classic-vortex-111_ForecastModelukmo_EnsembleNum1/rf_model.joblib'.
                           format(root_directory))
    if forecast_model_testing == 'ecmwf':
        test_df = pd.read_csv('{}/LabelledData/ecmwf/2023_windowsize14_district.csv'.
                              format(root_directory))
        monsoon_test_list = daterange(date(2023, 4, 1), date(2023, 10, 31))
        monsoon_test = test_df[test_df['date'].isin(monsoon_test_list)]
        y_test = monsoon_test['label']
        X_test = monsoon_test.drop(columns=['label', 'Unnamed: 0'])

        # Rename ecmwf columns so that the model can recognize the correct features from the UKMO trained model
        X_test.columns = X_test.columns.str.replace('ecmwf', 'UKMO')
    if forecast_model_testing == 'ncep':
        test_df = pd.read_csv('{}/LabelledData/NCEP/ensemble_1/2023_windowsize14_district.csv'.
                              format(root_directory))
        monsoon_test_list = daterange(date(2023, 4, 1), date(2023, 10, 31))
        monsoon_test = test_df[test_df['date'].isin(monsoon_test_list)]
        y_test = monsoon_test['label']
        X_test = monsoon_test.drop(columns=['label', 'Unnamed: 0'])

        # Rename ecmwf columns so that the model can recognize the correct features from the UKMO trained model
        X_test.columns = X_test.columns.str.replace('NCEP', 'UKMO')

    info_cols = ['date', 'district']
    test_info = X_test[info_cols]
    X_test = X_test.drop(columns=info_cols)

    # Measure model performance on test set
    print('Evaluating model...')
    probs = rf_model.predict_proba(X_test)

    # Re-indexing so we can put it all in a df
    test_info = test_info.reset_index().drop(columns=['index'])
    y_test = y_test.reset_index().drop(columns=['index'])

    accuracy = rf_model.score(X_test, y_test)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    df_probs = pd.DataFrame()
    df_probs['model soft predictions'] = probs[:, 1]
    df_probs['groundtruth'] = y_test
    df_probs['date'] = test_info['date']
    df_probs['district'] = test_info['district']
    df_probs.to_csv('{}/predictions_and_groundtruth.csv'.format(results_dir))

    # Plot roc_auc curve
    roc_auc(y_test, probs, results_dir)

    calc_model_performance(rf_model, y_test, probs, X_test, results_dir, wandb_exp, model_type)


if __name__ == '__main__':
    args = get_args()
    root_dir = args.root_dir
    results_dir = args.results_dir
    model = args.model
    test_y = args.test_year
    forecast_model = args.forecast_model
    ensemble_num = args.ensemble_num
    hindcast_model = args.hindcast_model
    exp = args.experiment_type
    wandb_setting = args.wandb_setting
    test_forecast = args.test_forecast
    tuning = args.parameter_tuning

    #root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'

    # Set up wandb experiment for tracking
    experiment = wandb.init(project='landslide-prediction',
                           resume='allow', anonymous='must')
    experiment.config.update(dict(model=model, test_year=test_y))

    # Make results directory
    if wandb_setting == 'offline':
        results = '{}/{}_ForecastModel_{}_EnsembleNum{}'.format(results_dir, hindcast_model, wandb.run.id,
                                                                       forecast_model, ensemble_num)
    else:
        results = '{}/{}_ForecastModel_{}_EnsembleNum{}'.format(results_dir, hindcast_model, experiment.name,
                                                                       forecast_model, ensemble_num)
    os.mkdir(results)

    if exp == 'ukmo_trained':
        test_forecast_model = test_forecast
        run_trained_ukmo(root_dir, results, experiment, model, test_forecast_model)

    else:
        # Load data
        print('Loading data...')
        data_dir = '{}/LabelledData_{}/{}/ensemble_{}'.format(root_dir, hindcast_model, forecast_model, ensemble_num)
        X_train, y_train, X_test, y_test, X_val, y_val = load_data(test_y, data_dir, exp)

        if model == 'rf':
            run_rf(data_dir, X_train, y_train, X_test, y_test, X_val, y_val, results, experiment, model, test_y, tuning)

        if model == 'gb':
            run_gb(data_dir, X_train, y_train, X_test, y_test, X_val, y_val, results, experiment, model, test_y, tuning)

        if model == 'xgb':
            run_xgb(data_dir, X_train, y_train, X_test, y_test, X_val, y_val, results, experiment, model, test_y, tuning)

        if model == 'mlp':
            run_mlp(data_dir, X_train, y_train, X_test, y_test, X_val, y_val, results, experiment, model, test_y, tuning)