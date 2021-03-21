 # Technical Report
 
 The algorithm is divided in 4 parts : Data pre-processing, Model selection, Model assesment and Predictions.
 
 ## Technical choices
 
 #### Data pre-processing
 
* __Missing values__: 
I replaced the missing values with the median value for numerical features, and the one that appears most often for the categorical features. We could have deleted the lines with missing values but we would have lost too much data.
 
* __Categorical features encoding__:
For categorical features, we can turn them into numbers with label encoder or turn each category into a binary column with one hot encoder. I choose to use the label encoder because one hot encoder adds to many columns. If I wanted to use one hot encoder I should have then make a feature importance analysis for removing the ones with less impact.

 #### Model selection
 
 The problem is a supervised one as we have a training set. The target is a binary result (0 or 1) so we need to make a supervised binary classification. For this type of problem, some of the best methods are Logistic Regression, Naive Bayes, Support Vector Machine, XGBoost, Decision Tree and Random Forest. Decision tree and Random Forest are based on the same method but Random Forest can be more accurate when different features have similar importance and Decision Tree is more efficient when features have a hierarchical impact. As I didn't make a feature importance analysis and I don't have more information I decided to try both, along with the other methods.

* __Metrics__: 
For selecting the model, I separed the training dataset in two parts and kept 20% of the data for validation. I tried diferent machine leaning models and computed metrics for each of them using the validation dataset. For a classification problem, we can use different metrics like accuracy, precision, recall and f1 score. The choice of the right metric depends on tthe type of precision wanted on the target variable (for exemple if we want to optimize false negatives or false positives). As we have imbalance training data (more 0s than 1s) but I don't know what type of precision we want, I prefered to look at different metrics for each model and choose the best optimized one, which is Random Forest.

* __Model parameters__: 
For Random Forest, the most important parameter is the number of estimators. The accuracy grows with the number of estimators but the computation time grows too. So I wanted to choose the lower number of estimators when the accuracy stabilizes. I tried different ones and found that n_estimators=600 is sufficient, but I kept 800 estimators to be sure.
 

 ## Improvements
 According to the confusion matrix computed for Random Forest classification, the model is good at predicting 0s predicted 0s b but bad at predicting 1s. There are some methods that could help improving the algorithm reliability.
 
  #### Data pre-processing
  
 * __Missing values__: I decided to fill the missing values but another strategy could be to remove the lines with missing value or even consider missing values as a new category. 
 *  __Categorical features encoding__: Categorical features can also be encoded with one hot encoding, creating a new column for each category.
 * __Imbalance data__ : The trainaing data is imbalanced (more 0s than 1s for target flag). Different strategies can help overcome this issue. Undersampling : removing lines with 0s. Oversampling: adding ls by using existing ones.
 * __Feature selection__: It could be useful to make a feature importance analysis in order to get rid off of unrelevant features. Moreover, I could use a correlation matrix to remove correlated features.

 #### Model selection
 * __Metrics__: The choice of the metrics depends on the aim of the study so more insights about the general goals could help choose the most appropriate model.
 * __Cross-validation__: For model selection, I decided to separate 20% of the training data for evaluation, but I could have use cross validation with cross-val-score function wich gives more accurate results about performance as the performances of the different models are evaluated multiple times on different subsets of the data. I didn't use this function because it requires more computation time and it allows to compute only one metric whereas I wanted to compare the different metrics.
 * __Parameters optimization__: For better parameter optimization I could use grid searching.
