 # Technical Report
 
 The algorithm is divided in 4 parts : Data pre-processing, Model selection, Model assesment and Predictions.
 
 ## Technical choices justifications
 
 #### Data pre-processing
 
* Missing values
I replaced the missing values with the median value for numerical features, and the one that appears most often for the categorical features. We could have deleted the lines with missing values but we would have lost too much data.
 
* Categorical features encoding
 For categorical features, we can turn them into numbers with label encoder or turn each category into a binary column with one hot encoder. I choose to use the label encoder because one hot encoder adds to many columns. If I wanted to use one hot encoder I should have then make a feature importance analysis for removing the ones with less impact.

 #### Model selection
 The problem is a supervised one as we have a training set. The target is a binary result (0 or 1) so we need to make a supervised binary classification. For this type of problem, some of the best methods are Logistic Regression, Naive Bayes, Support Vector Machine, XGBoost, Decision Tree and Random Forest. Decision tree and Random Forest are based on the same method but Random Forest can be more accurate when different features have similar importance and Decision Tree is more efficient when features have a hierarchical impact. As I didn't make a feature importance analysis and I don't have more information I decided to try both, along with the other methods.
 * Metrics
 * random forest parameters
 

 ## Improvements
 
  #### Data pre-processing
 * standardization
 * feature selection
 * imbalance data : oversampling/undersampling

 #### Model selection
 * cross-validation for model selection
 * parameters optimization - grid searching
