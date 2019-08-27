import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, OneHotEncoder, FunctionTransformer,RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV,KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import randint as sp_randint, skew,norm

__all__ = ['read_train_path','read_test_path','SelectItems','ImputeNa','FillbyType','FillbyGroup', 'MultiLabelEncoder', 'Modelling', 'Add6vars', 'Modiftype', 'AveragingModels', 'StackingAveragedModels', 'rmse']

def read_train_path():
    my_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(my_path, './data/train.csv')
    return(pd.read_csv(file_path, sep=','))

def read_test_path():
    my_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(my_path, './data/test.csv')
    return(pd.read_csv(file_path, sep=','))



class SelectItems(BaseEstimator, TransformerMixin):
    def __init__(self, type_select):
        self.type_select = type_select
        self.col_select = None
        
    def fit(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        self.col_select = list(X.columns[X.dtypes == self.type_select])
        return self
    
    def transform(self, X, y=None):
        assert(isinstance(X, pd.DataFrame))
        return X[self.col_select]
    
class ImputeNa(BaseEstimator, TransformerMixin):
    def __init__(self,word_fill = 'NA'):
        self.word_fill = word_fill
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.word_fill)    

    
class FillbyType(BaseEstimator, TransformerMixin):
    def __init__(self, vars_na, vars_0, vars_freq):
        self.vars_na = vars_na
        self.vars_0 = vars_0
        self.vars_freq = vars_freq
        self.val_freq = None
        
    def fit(self, X, y=None):
        self.val_freq = [X[var_i].value_counts().idxmax() for var_i in self.vars_freq]
        return self
    
    def transform(self, X, y=None):
        X[self.vars_na] = X[self.vars_na].fillna('None')
        X[self.vars_0] = X[self.vars_0].fillna(0)
        for index, var_i in enumerate(self.vars_freq):
            X[var_i] = X[var_i].fillna(self.val_freq[index])
        return X

class FillbyGroup(BaseEstimator, TransformerMixin):
    def __init__(self, var_gp,var_to_fill):
        self.var_gp = var_gp
        self.var_to_fill = var_to_fill
        self.var_fill = None
        
    def fit(self, X, y=None):
        self.var_fill = X[self.var_to_fill].groupby(X[self.var_gp]).median()
        return self
    
    def transform(self, X, y=None):
        row = X[self.var_to_fill].isnull()
        X[self.var_to_fill][row] = X[self.var_gp][row].map(lambda neighbor: self.var_fill[neighbor])
        return X

class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.var = []
        self.var_cat = []
        self.label = []
    
    def fit(self, X, y=None):
        for i in X.columns:
            label_i = LabelEncoder()
            label_i.fit(X[[i]])
            label_i.classes_ = np.append(label_i.classes_,'None')
            
            self.var.append(i)
            self.var_cat.append(label_i.classes_)
            self.label.append(label_i)
        return self
    
    def transform(self, X, y=None):
        res = []
        for index, var_i in enumerate(self.var):
            
            x = [i if i in self.var_cat[index] else 'None' for i in X[var_i]]
            res.append(pd.Series(self.label[index].transform(x)))   
        return pd.concat(res,axis=1)   
    

class Modelling(BaseEstimator, RegressorMixin):
    
    def __init__(self, model, **param):
        self.param = param
        self.model = model
    
    def fit(self, X, y):
        self.model.set_params(**self.param)
        self.model.fit(X, y)
        return self
    
    def predict(self, X, y=None):
        return pd.Series(self.model.predict(X), index = X.index)
    
    def score(self, X, y):
        p = self.predict(X)
        mae, rmse = abs(p - y).mean(), np.sqrt(np.power((p - y),2).mean())
        return mae, rmse
    
    def plot_obs_pred(self, X, y):
        p = self.predict(X)
        pd.DataFrame({'obs':y, 'pred':p}).plot()
        plt.show()
        

class Add6vars(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X, y=None):
        assert(isinstance(X, pd.DataFrame))
        X['total_sq_footage'] = X['GrLivArea'] + X['TotalBsmtSF'] 
        X['total_baths'] = X['BsmtFullBath'] + X['FullBath'] + (0.5 * (X['BsmtHalfBath'] + X['HalfBath']))
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        # How many years has remoded from built
        X['RemodYears'] = X['YearRemodAdd'] - X['YearBuilt']
        # Did a remodeling happened from built?
        X["HasRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]) * 1
        # Did a remodeling happen in the year the house was sold?
        X["HasRecentRemodel"] = (X["YearRemodAdd"] == X["YrSold"]) * 1
        return X

class Modiftype(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X, y=None):
        assert(isinstance(X, pd.DataFrame))
        X['MSSubClass'] = X['MSSubClass'].astype('object')
        X['OverallCond'] = X['OverallCond'].astype('object')
        return X
    
def rmse(predictions, targets):
    return np.sqrt(((predictions.values - targets.values) ** 2).mean())

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
    
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_index,:], y.iloc[train_index,:])
                y_pred = instance.predict(X.iloc[holdout_index,:])
                y_pred = y_pred.reshape(y_pred.shape[0],)
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)