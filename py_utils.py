def most_recent_model(model_name, wd, subfolder):
    """
    This function finds the name of the most recent model from the file directory.
    It's a companion function to the dunp_diff_model function.
    
    Example:
    If you had a model called 'log_reg' that was saved at this timestamp: '2019-05-21 12/04/54.815294'.
    The filename would be 'log_reg 2019-06-21 12/04/54.875304.joblib' and the root would be 'log_reg'.

    log_reg = load(dans_utils.most_recent_model(model_name = 'log_reg', wd = "/Users/Daniel/Desktop/Projects/First_Project/"))
    
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
    
    dans_utils.dump_diff_model(name = 'master_model', log_reg, wd = "/Users/Daniel/Desktop/Projects/First_Project/")
    
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
def sensitivity_analysis(sens_voi, sens_x_test, sens_y_test, sens_model, multiplier = 1.1):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    sens_x_test = sens_x_test.copy() # take a copy of x_test dataframe
    # instantiate empty lists for later
    sens_vars, sens_roc_auc = [], []
    for variable in sens_voi: # for each variable in the variables of interest
        if variable not in sens_x_test.columns: # if variable is one-hot-encoded etc. let's skip
            print(f'{variable} variable not in model')
            pass
        else:
            sens_x_test[variable] = sens_x_test[variable] * multiplier # change by 10% (or user-defined multiplier)
            sens_y_pred = sens_model.predict(sens_x_test) # predict
            sens_y_scores = sens_model.predict_proba(sens_x_test) # score
            # now we will append the variable and roc_auc to lists
            sens_vars.append(variable) # add variable to list
            sens_roc_auc.append(roc_auc_score(sens_y_test, sens_y_scores[:,1])) # add roc auc score to list
    # now let's turn it into a dataframe
    sens_df = pd.DataFrame(zip(sens_vars, sens_roc_auc), columns = [f'Variable multiplied by {multiplier}', 'Sensitivity ROC_AUC'])
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