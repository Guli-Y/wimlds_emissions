import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from emissions.trainer import Trainer
from emissions.data import load_data, clean_data
from sklearn.metrics import precision_score


class ImpSearch():
    """ 
    What ImSearch do:
    1. For each max_depth in a given list of max_depth values, 
       it trains the model, implement the solution for year 2019
       and then calculates "the average on-road Days Per Failing Vehicle in 2019 (DPFV)" .
    2. It saves the max_depth which gave the smallest DPFV as the best max_depth.
    3. Then it uses the best max_depth to train the RF on the data before 2020 
        and evaluate RF performance on 2020 data
    """
    # selecting features for modeling
    cols = ['VEHICLE_AGE', 'MILE_YEAR', 'MAKE', 
            'MODEL_YEAR', 'ENGINE_WEIGHT_RATIO']
    cat_col = ['MAKE']
    
    def __init__(self):
        """
        """
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.total_tests = None
        self.total_fails = None
        self.n_estimators = [50]
        self.max_depth = [2, 3]
        self.cum_df = None
        self.performances = {}
        self.best_max_depth = None
        
    def load_data(self):
        """
        1. loads clean data and save it as class attribute self.df
        2. adds counter columns
            count_fail: 1 if the test result is fail else 0
            count_test: 1 for each test
        """
        df = load_data()
        df = clean_data(df)
        df['count_test'] = 1
        df['count_fail'] = df.RESULT
        self.df = df
    
    def train_test_split(self, year):
        ''' 
        for a given year, 
        splits data to train (before that year) and test (that year) sets
        '''
        train = self.df[self.df.TEST_SDATE.dt.year < year].sort_values('TEST_SDATE')
        test = self.df[self.df.TEST_SDATE.dt.year == year].sort_values('TEST_SDATE')
        self.y_train = train.pop('RESULT')
        self.X_train = train
        self.y_test = test.pop('RESULT')
        self.X_test = test
        self.total_tests = self.X_test.shape[0]
        self.total_fails = self.y_test.sum()

    def get_estimator(self, depth):  
        '''
        uses Trainer class from trainer.py to get the trianed estimator
        prints the precision and recall scores 
        ''' 
        trainer = Trainer(self.X_train[self.cols],
                          self.y_train,
                          metric='precision',
                          n_estimators = self.n_estimators,
                          with_categorical=self.cat_col,
                          max_depth=depth
         )
        trainer.grid_search()
        tmp = trainer.evaluate(self.X_test[self.cols], self.y_test)
        #print('\nmax_depth:', trainer.search_result.best_params_['model__max_depth'])
        #print(tmp)
        # trainer.learning_curve()
        return trainer
    
    def get_counter_table(self):
        '''
        creates a counter table with TEST_SDATE as index and having columns:
            cum_tests: cumulative number of tests along the year
            cum_fails: cumulative number of failed tests along the year
        '''
        df = self.X_test[['TEST_SDATE', 'count_fail', 'count_test']].copy()
        df.set_index(pd.DatetimeIndex(df.TEST_SDATE), inplace=True)
        df['cum_tests'] = df.count_test.cumsum()
        df['cum_fails_reality'] = df.count_fail.cumsum()
        df['fails_left_reality'] = self.total_fails - df['cum_fails_reality']
        df['dayofyear'] = df.index.dayofyear
        df.drop(columns=['count_fail', 'count_test', 'TEST_SDATE'], inplace=True)
        return df
    
    def evaluate_baseline(self, year=2020, plot=True):
        """ evaluates the baseline model on 2020 data"""
        # get test data
        self.train_test_split(year)
        # get counter table
        df = self.get_counter_table()
        # get heuristic prediction
        y_pred = (self.X_test.VEHICLE_AGE > 16).astype('int')
        y_proba = (self.X_test.VEHICLE_AGE - self.X_test.VEHICLE_AGE.min())\
                    /(self.X_test.VEHICLE_AGE.max()-self.X_test.VEHICLE_AGE.min())
        # create df with prediction outcomes
        pred_df = pd.DataFrame.from_dict({'y_true': self.y_test, 
                                          'y_pred': y_pred,
                                          'y_proba': y_proba})
        pred_df = pred_df.sort_values('y_proba', ascending=False)
        # merge the prediction with counter table
        pred_df.index = df.cum_tests
        df = df.merge(pred_df, how='left', left_on='cum_tests', right_index=True)
        # add new columns storing cumulative number of fails
        df['cum_fails_heuristic'] = df.y_true.cumsum()
        df.drop(columns=['y_true', 'y_pred', 'y_proba'], inplace=True)
        df['fails_left_heuristic'] = self.total_fails - df['cum_fails_heuristic'] 
        # save this dataframe as class attribute
        self.cum_df = df
        # day of year 100
        t = df[df.dayofyear==100].index[0]
        fails_real_t = df[df.dayofyear==100].cum_fails_reality[0]
        fails_heuristic_t = df[df.dayofyear==100].cum_fails_heuristic[0]
        # calculate DPFV
        DPFV_reality = round(df.groupby('dayofyear')['fails_left_reality'].max().sum()/self.total_fails, 1)
        DPFV_heuristic = round(df.groupby('dayofyear')['fails_left_heuristic'].max().sum()/self.total_fails, 1)
        self.performances['reality'] = [DPFV_reality, fails_real_t]
        self.performances['heuristic'] = [DPFV_heuristic, fails_heuristic_t]
        if plot:
            # plot reality
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, 
                df.cum_fails_reality, 
                label=f'{DPFV_reality} DPFV, reality', 
                c='red')    
            # horizontal grey line - reality
            plt.plot(df[df.index < t].index, 
                    [fails_real_t for i in range(len(df[df.index < t]))], 
                    c='grey')
            # vertical grey line - reality
            plt.plot([t for i in range(100)], np.linspace(0, fails_heuristic_t, 100), c='grey')
            # plot heuristic
            plt.plot(df.index, 
                df.cum_fails_heuristic, 
                label=f'{DPFV_heuristic} DPFV, baseline',
                c='#2003fc')
            # horizontal grey line - heuristic
            plt.plot(df[df.index < t].index, 
                    [fails_heuristic_t for i in range(len(df[df.index < t]))], 
                    c='grey')
            # fill in the area corresponding to total pollution
            plt.fill_between(df.index,
                            df.cum_fails_heuristic,
                            [self.total_fails for i in range(df.shape[0])],
                            color='#abb0c7', alpha=0.2)
            plt.ylim(0, self.total_fails)
            plt.xlim(df.index.min(), df.index.max())
            plt.ylabel('Number of Vehicles Failed the Inspection')
            plt.xlabel('Date')
            plt.legend(loc='lower right', fontsize='large')
            plt.title(f'Performance for {year}')
            plt.show()
        # metrics table
        df = pd.DataFrame.from_dict(self.performances, 
                                orient='index',
                                columns=['DPFV', 'Number of Polluting Vehicles off the road by day 100'])
        return df

    def get_best_max_depth(self, year=2019, n_estimators=[50], max_depth=[4, 5], scores=False):
        """
        1. For each value in a list of given max_depth values:
                1. it trains the RF using that max_depth value
                2. implements the solution for 2019
                3. calculates the DPFV 
        2. Plots Cumulative number of failed vehicles vs. date of year 
        3. Saves the max_depth value which gave the smallest DPFV as the best max_depth
        """
        # train set split
        self.train_test_split(year)
        # update the class attributes         
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.performances = {}
        # get counter table
        df = self.get_counter_table()
        # calculate DPFV and plot for each max_depth value
        plt.figure(figsize=(10, 5))
        for depth in self.max_depth:
            trainer = self.get_estimator([depth])
            # get predictions
            y_pred = trainer.search_result.predict(self.X_test[self.cols])
            y_proba = trainer.search_result.predict_proba(self.X_test[self.cols])
            pred_df = pd.DataFrame.from_dict({'y_true':self.y_test, 
                                              'y_pred':y_pred, 
                                              'y_proba':y_proba[:,1]})
            pred_df = pred_df.sort_values('y_proba', ascending=False)
            # merge the predictions with self.cum_df
            _ = self.evaluate_baseline(year, plot=False)
            pred_df.index = self.cum_df.cum_tests
            df = self.cum_df.merge(pred_df, how='left', left_on='cum_tests', right_index=True)
            # add new columns storing cumulative number of fails
            df[f'cum_fails_{depth}'] = df.y_true.cumsum()
            df.drop(columns=['y_true', 'y_pred', 'y_proba'], inplace=True)
            df[f'fails_left_{depth}'] = self.total_fails - df[f'cum_fails_{depth}']
            # day 100 of the year
            t = df[df.dayofyear==100].index[0]
            fails_t = df[df.dayofyear==100][f'cum_fails_{depth}'][0]
            # collect performance
            DPFV = round(df.groupby('dayofyear')[f'fails_left_{depth}'].max().sum()/self.total_fails, 1)
            self.performances[depth] = [DPFV, fails_t]
            # plot cumulative number of fails vs. date of year
            plt.plot(df.index, 
                df[f'cum_fails_{depth}'], 
                label=f'{DPFV} DPFV, max_depth = {depth}')
        # plot heuristic
        DPFV_heuristic = self.performances['heuristic'][0]
        plt.plot(df.index, 
            df.cum_fails_heuristic, 
            label=f'{DPFV_heuristic} DPFV, baseline',
            c='#2003fc')
        # plot reality
        DPFV_reality = self.performances['reality'][0]
        plt.plot(df.index, 
            df.cum_fails_reality, 
            label=f'{DPFV_reality} DPFV, reality', 
            c='red')  
        # get the best max_depth
        perf_df = pd.DataFrame.from_dict(self.performances, 
                                            orient='index',
                                            columns=['DPFV', 
                                            'Number of Polluting Vehicles off the road by day 100'])
        perf_df.index.name = 'max_depth'
        perf_df.sort_values('DPFV', inplace=True)
        self.best_max_depth = int(perf_df.index[0])
        plt.ylim(0, self.total_fails)
        plt.xlim(df.index.min(), df.index.max())
        plt.ylabel('Number of Vehicles Failed the Inspection')
        plt.xlabel('Date')
        plt.legend(loc='lower right', fontsize='large')
        plt.title(f'Year {year}')
        plt.show()
        return perf_df
        
    def evaluate_RF(self, year):
        ''' 
        evaluates the Random Forest  on 2020 data using the best max_depth
        '''
        max_depth = [self.best_max_depth]
        n_estimators = self.n_estimators
        self.get_best_max_depth(year=year, 
                                n_estimators=n_estimators, 
                                max_depth=max_depth)
        
if __name__ == "__main__":
    imp = ImpSearch()
    imp.load_data()
    imp.evaluate_baseline(2020)