import sklearn.metrics
def most_recent_model(model_name, wd, subfolder):
    """
    This function finds the name of the most recent model from the file directory.
    It's a companion function to the dunp_diff_model function.
    
    Example:
    If you had a model called 'log_reg' that was saved at this timestamp: '2019-05-21 12/04/54.815294'.
    The filename would be 'log_reg 2019-06-21 12/04/54.875304.joblib' and the root would be 'log_reg'.

    log_reg = load(py_utils.most_recent_model(model_name = 'log_reg', wd = "/Users/Daniel/Desktop/Projects/First_Project/"))
    
    Args:
    model_name (str): the name of the model you want to find the most recent version of.
    wd (str): your working directory, the function will look in the /Models subfolder of this working directory
    
    Returns:
    most_recent_model_path (str): the filepath of your most recent model
    """
    from os import listdir
    from os.path import isfile, join
    from datetime import datetime
    files_in_directory = [f for f in listdir(join(wd, 'Models', subfolder)) if isfile(join(wd, 'Models', subfolder, f))] #Find all the files 
    if len([x for x in files_in_directory if x.startswith(model_name)]) > 0:
        most_recent_model = sorted([x for x in files_in_directory if x.startswith(model_name)], reverse = True)[0]
        most_recent_model_path = join(wd, 'Models', subfolder, most_recent_model)
    else:
        most_recent_model_path = None
    return most_recent_model_path
def dump_diff_model(model_name, model_var, wd, subfolder=''):
    """
    This function will dump your model with a unique timestamp associated with it.
    First it will check if the most recent model is the same as the one you're trying to dump, 
    if this is case then the funciton will do nothing.
    
    Example:
    Let's say you want to dump a model called log_reg. You want to dump this file with the name 'master_model'.
    
    py_utils.dump_diff_model(name = 'master_model', log_reg, wd = "/Users/Daniel/Desktop/Projects/First_Project/")
    
    Args:
    model_mame (str): the name you want to give the model you're dumping.
    model_var: the actual model itself.
    wd (str): your working directory, the function will look in the /Models subfolder of this working directory.
    """
    from joblib import dump, load
    from datetime import datetime
    import filecmp 
    import os
    from os.path import isfile, join
    today = datetime.today()
    path_of_model_to_save = join(wd, 'Models', subfolder, f'{model_name}_{today}.joblib')
    most_recent_model_path = most_recent_model(model_name, wd = wd, subfolder = subfolder)
    holding_path = join(wd, 'Models', subfolder, 'holding.joblib')
    dump(model_var, holding_path)
    if most_recent_model_path == None:
        dump(model_var, path_of_model_to_save)
    elif not filecmp.cmp(most_recent_model_path, holding_path): #Only save model if it's different, this overwrite the file
        dump(model_var, path_of_model_to_save)
    os.remove(holding_path) #remove the holding file
def sensitivity_analysis(sens_voi, X_test, y_test, model, multiplier = 1.1, scoring_func = sklearn.metrics.roc_auc_score):
    """
    This function runs a sensitivity analysis by taking a model then modifying its feature vector in a univariate fashion.
    Following this, a scoring function is applied.
    When all features are ran through then a summary dataframe is returned.
    
    Example:
    Let's say you have a model called log_reg, you want to do sensitivity analysis on all features in a feature vector called X_train.
    You also want to multiply each feature by 1.1 and use roc_auc as your scoring metric.
    
    py_utils.sensitivity_analysis(
        sens_voi = X_train.columns, X_test = X_test, y_test = y_test, model = log_reg, multiplier = 1.1,  scoring_func = sklearn.metrics.roc_auc_score
        )
        
    Args:
    sens_voi (list): Sensitivitiy analysis features of interest.
    X_test (pd.DataFrame): the feature vector used for scoring.
    y_test (pd.Series): the target used for scoring.
    model (sklearn.estimator): the model you want to conduct sensitivity analysis on.
    multipler (float, int): the number that each feature is multiplied by.
    scoring_func (function): the sklearn scoring function you want to use e.g. sklearn.metrics.accuracy_score
    
    Returns:
    sens_df (pd.DataFrame): summary DataFrame showing sensitivity analysis results.
    """
    import pandas as pd
    # instantiate empty lists for later
    sens_vars, sens_score = [], []
    for feature in sens_voi: # for each feature in the features of interest
        X_test_altered = X_test.copy() # fresh copy after each loop to prevent changes carrying over for loop
        if feature not in X_test_altered.columns: # if feature is one-hot-encoded etc. let's skip
            print(f'Feature not in model: {feature}')
            pass
        else:
            X_test_altered.loc[:, feature] = X_test_altered.loc[:, feature] * multiplier # change by 10% (or user-defined multiplier)
            y_pred = model.predict(X_test_altered) # predict
            y_scores = model.predict_proba(X_test_altered) # score
            # now we will append the feature and score to lists
            sens_vars.append(feature) # add feature to list
            sens_score_dict = {'X_test': X_test_altered, 'y_true': y_test, 'y_score': y_scores[:, 1], 'y_pred': y_pred} # define dictionary for unpacking later
            # only select {key: value} pairs that appear in the scoring function
            sens_score_dict = {k: v for (k, v) in sens_score_dict.items() if k in scoring_func.__code__.co_varnames} 
            sens_score.append(scoring_func(**sens_score_dict)) # add score to list
    # now let's turn it into a dataframe
    sens_df = pd.DataFrame(zip(sens_vars, sens_score), columns = [f'Feature multiplied by {multiplier}', f'Sensitivity {scoring_func.__code__.co_name}'])
    return sens_df
def hide_code_cells():
    #Auto-hide code cells in jupyter
    from IPython.display import HTML, display
    
    hide_code = HTML('''<script>
    code_show=false; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    The code for this jupyter notebook has been hidden by default for easier reading.
    To toggle on/off the code, click <a href="javascript:code_toggle()">here</a>.''')
    
    display(hide_code)