import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

class HousingPrices:

    df = pd.read_csv('HousingPrices.csv')

    def dataCheck(self):
        print(self.df.head())
        print('')
        print(self.df.tail())
        print('')
        print(self.df.describe())
        print('')
        print(self.df.info())
        print('')

    def Processing(self):
        self.df = self.df.dropna()
        self.df.drop(['HouseID'], axis=1, inplace=True)
        self.df['TypeOfLiving'] = self.df['TypeOfLiving'].str.capitalize()

        condition_mapping = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        self.df['Condition'] = self.df['Condition'].map(condition_mapping)

        self.df = pd.get_dummies(self.df, columns=['TypeOfLiving'], drop_first=True)

        scaler = StandardScaler()
        self.df[['squareMeters', 'useableArea']] = scaler.fit_transform(self.df[['squareMeters', 'useableArea']])

    def calculate_type_price_correlation(self):
        if 'TypeOfLiving_House' in self.df.columns:
            correlation = self.df['Price'].corr(self.df['TypeOfLiving_House'])
            print("Correlation between Price and Housing Type (House vs. Apartment):", correlation)
        else:
            print("TypeOfLiving_House column not found. Check if one-hot encoding was applied correctly.")

    def plotting(self):
        numeric_df = self.df.select_dtypes(include=[np.number])

        plt.figure(figsize=(10, 10))
        sns.heatmap(numeric_df.corr(), cbar=True, square=True, fmt='.2f', annot=True,
                    annot_kws={'size': 11}, cmap='coolwarm')
        plt.show()

    def testingSets(self):
        x = self.df.drop(['Price'], axis=1)
        y = self.df['Price']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

        # XGBoost Regressor
        xgb_model = XGBRegressor()
        xgb_model.fit(x_train, y_train)
        y_pred_test_xgb = xgb_model.predict(x_test)
        print("Prediction on training set: " + str(y_pred_test_xgb))

        '''
        dt_model = DecisionTreeRegressor(random_state=2)
        dt_model.fit(x_train, y_train)
        y_pred_test_dt = dt_model.predict(x_test)
        '''

        print("XGBoost Regressor Metrics:")
        self.evaluateModel(y_test, y_pred_test_xgb)

        '''
        print("Decision Tree Regressor Metrics:")
        self.evaluateModel(y_test, y_pred_test_dt)
        '''

    def evaluateModel(self, y_test, y_pred):
        score_1 = r2_score(y_test, y_pred)
        score_2 = mean_absolute_error(y_test, y_pred)
        score_3 = mean_squared_error(y_test, y_pred)

        print('R Squared Error:', format(score_1, '.2f'))
        print('Mean Absolute Error:', format(score_2, '.2f'))
        print('Mean Squared Error:', format(score_3, '.2f'))

        self.plotting2(y_test, y_pred)

    def plotting2(self, y_test, y_pred_test):
        y_test = y_test.reset_index(drop=True)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_test, color='blue', label='Actual Prices', alpha=0.6)  # Blue dots for actual prices
        plt.scatter(y_test, y_pred_test, color='red', marker='x', label='Predicted Prices', alpha=0.6)  # Red Xs for predicted prices

        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        plt.legend()
        plt.show()

        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
        pd.set_option('display.float_format', '{:,.2f}'.format)
        #display(results_df)
        print(results_df)


project = HousingPrices()
project.dataCheck()
project.Processing()
project.plotting()
project.calculate_type_price_correlation()
project.testingSets()
