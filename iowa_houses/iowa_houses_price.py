import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


iowa_file_path = os.path.abspath(os.curdir) + '/train.csv'

home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Decision Tree Model
def mae(max_leaf_nods, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nods, random_state=0)
    model.fit(train_X, train_y)
    model_predect = model.predict(val_X)
    mae = mean_absolute_error(val_y, model_predect)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
dt_mea = mae(best_tree_size, train_X,  val_X, train_y, val_y)

print("Iowa sleprice Validation MAE for Decision Tree Model: {:,.0f}".format(dt_mea))

#Random Forest Model
rf_iowa_model = RandomForestRegressor(random_state=1)
rf_iowa_model.fit(train_X, train_y)
rf_prediction = rf_iowa_model.predict(val_X)
rf_mea = mean_absolute_error(val_y, rf_prediction)
print("Iowa saleprice Validation MAE for Random Forest Model:  {:,.0f}".format(rf_mea))


#From above Random_forest_prediction is better

test_data_path = os.path.abspath(os.curdir)+'/test.csv'
test_data = pd.read_csv(test_data_path)

test_X = test_data[feature_names]

rf_model_on_full_data = RandomForestRegressor()
rf_model_on_full_data.fit(X, y)

test_predict = rf_model_on_full_data.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_predict})
output.to_csv('submission.csv', index=False)
