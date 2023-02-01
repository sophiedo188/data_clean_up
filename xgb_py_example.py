import warnings
warnings.filterwarnings('ignore')
from typing import List
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score, confusion_matrix,precision_score,recall_score 
import matplotlib.pyplot as plt
import getpass
import pandas as pd
# evaluate predictions

def format_(name, value) -> str:
    new = name + ": %.2f%%" % (value * 100.0)
    print(new)
    return new

def eval_pred(y_test, predictions) -> dict:
    result = {}
    accuracy = accuracy_score(y_test, predictions)
    result['accuracy'] = format_('accuracy', accuracy)
    f1 = f1_score(y_test, predictions)
    result['f1 score'] = format_('f1 score', f1)
    roc_auc = roc_auc_score(y_test, predictions)
    result['roc_auc score'] = format_('roc_auc score', roc_auc)
    confusion = confusion_matrix(y_test, predictions)
    print(confusion)
    result['confusion matrix'] = confusion
    precision = precision_score(y_test, predictions)
    result['precision score'] = format_('precision score', precision)
    recall = recall_score(y_test, predictions)
    result['recall score'] = format_('recall score', recall)
    
    return result

def load_model(model_name):
    model = pickle.load(open("../xgboost_models/" + model_name +".pickle.dat", "rb"))
    return model

def save_results(model, scores: List[dict], num_model) -> None:
	user_name = getpass.getuser()
	model_name = "{0:0=3d}".format(num_model)
	pickle.dump(model, open("../xgb_models/" + model_name + ".pickle.dat", "wb"))
	with open(f"{user_name}.txt", "a+") as f:
		head = "model " + model_name + ":" + "\n"
		f.writelines(head)
		for name, value in scores.items():
			f.writelines(str(value) + "\n")
			f.writelines("\n")
			print(f"Result written to {user_name}.txt")

if __name__ == "__main__":
    # load data
    X_train=pd.read_csv('X_train.csv')
    Y_train=pd.read_csv('Y_train.csv')
    Y_train = Y_train.loc[:, 'IMMUNNO_EFFECT_Induced']
    train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size=0.2, train_size=0.8, random_state=42)
    #train_x, train_y, test_x, test_y = None, None, None, None
    print(X_train.shape)
    print(Y_train.shape)
    # default paramters
    params = {'max_depth': 6, 
         'eta': 0.1, 
         'silent': 1, 
         'objective': 'binary:logistic',
         'subsample': 1.0,
         'colsample_bytree': 0.85,
         'learning_rate': 0.2,
         'eval_metric': 'auc'
        }

    model = xgb.XGBClassifier(**params)

    """ random search to tune the parameters """

    # base parameters
    base_params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # regression task
        'seed': 42, 'subsample': 1.0, 'eta': 0.1}

    # TODO: change following
    params = {
            'max_depth': [4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'colsample_bytree': [0.8, 0.85, 1],
    }
    folds = 3
    param_comb = 9

    # TODO: change above

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

    random_search = RandomizedSearchCV(model, 
                                param_distributions=params, 
                                n_iter=param_comb, 
                                scoring='f1', 
                                n_jobs=4, 
                                cv=skf.split(train_x, train_y), 
                                verbose=3, 
                                random_state=42)

    random_search.fit(train_x, train_y)
    print(random_search.best_params_)

    new_params = {**base_params, **random_search.best_params_}
    model = XGBClassifier(**new_params)

    """ fit the model """
    model.fit(train_x, train_y)

    """ make predictions """
    preds = model.predict_proba(test_x)
    preds = [val[1] for val in preds]
    pred_y = [round(val) for val in preds]

    # evaluate predictions
    scores = eval_pred(test_y, pred_y)

    # plot auc

    fpr, tpr, thresholds = roc_curve(test_y, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
    """ plot importance"""
    x = model.get_booster().get_score(importance_type='gain')
    import operator
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))
    print(sorted_x)
    plot_importance(model, max_num_features=10)
    #plt.show()
    plt.savefig('Importance.png')

