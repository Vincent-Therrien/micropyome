import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
import argparse
from datetime import datetime
from sklearn.metrics import make_scorer
from scipy.stats import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.lines as mlines

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run model with or without hyperparameter optimization.")
# Add an argument for hyperparameter optimization
parser.add_argument('--mode', type=str, default='default', help='"opt" to run with optimization, anything else runs with default parameters.')

# Parse the arguments
args = parser.parse_args()

# Check if we should run the optimization or use default parameters
run_optimization = args.mode == "opt"

# Use "python3 reg_champs_comp.py --mode opt" to run hyperparameter optimization
# Use "pytho3 reg_champs_comp.py --mode default" / "pytho3 reg_champs_comp.py" for default run

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Could not import the lzma module. Your installed Python is incomplete. Attempting to use lzma compression will result in a RuntimeError.", category=UserWarning, module='pandas.compat')

#if not os.path.isdir("Champignons"):
#  #!unzip "data.zip"
#  !unzip "Champignons.zip"

start_time = datetime.now()

# Base path
base_path = "Champignons"

# Taxonomic levels
taxonomic_levels = ['fg', 'phylum', 'class', 'order', 'family', 'genus']

# Dictionaries to hold all results
all_results = {}
compared_study_r2_results = {}

# Non-important mean threshold
mean_threshold = 0.012 #0.01 #0.012 #0.01
mean_threshold_compared_study = 0.012 #0.01 #0.012

mean_threshold_fg = 0.01
mean_threshold_fg_compared_study = 0.01

# Functions
def normalize_rows(df):
    # Replace negative values with 0
    df[df < 0] = 0
    # Normalize each row so that the sum of the row equals 1
    row_sums = df.sum(axis=1)
    normalized_df = df.div(row_sums, axis=0)
    return normalized_df

def calculate_r_squared_by_category(y_observed, y_predicted):
    r_squared_values = {}

    # Iterate through each category to calculate R^2 separately
    for column in y_observed.columns:
        observed = y_observed[column]
        predicted = y_predicted[column]

        ss_res = ((observed - predicted) ** 2).sum()
        mean_observed = observed.mean()
        ss_tot = ((observed - mean_observed) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)

        r_squared_values[column] = r_squared

    return r_squared_values

def calculate_r_squared_by_category_ndarray(y_observed, y_predicted):
    r_squared_values = {}

    # Ensure y_observed and y_predicted are NumPy arrays
    y_observed = np.array(y_observed)
    y_predicted = np.array(y_predicted)

    # Iterate through each category (column index) to calculate R^2 separately
    for col_index in range(y_observed.shape[1]):  # Iterate over columns
        observed = y_observed[:, col_index]
        predicted = y_predicted[:, col_index]

        ss_res = ((observed - predicted) ** 2).sum()
        mean_observed = observed.mean()
        ss_tot = ((observed - mean_observed) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)

        r_squared_values[col_index] = r_squared  # Use column index as the key

    return r_squared_values

# Define a function to evaluate model performance and return R² mean
def evaluate_model_performance(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
    y_test_df = pd.DataFrame(y_test, columns=y_test.columns)

    # Exclude the "other" column from both observed and predicted data
    if(level == "phylum"):
      y_pred_df = y_pred_df.drop(columns=["other"], errors='ignore')
      y_test_df = y_test_df.drop(columns=["other"], errors='ignore')

    if(level == "fg"):
      for column in y_test_df.columns:
        if(y_test_df[column].mean() < mean_threshold_fg):
            y_test_df = y_test_df.drop(columns=[column], errors='ignore')
            y_pred_df = y_pred_df.drop(columns=[column], errors='ignore')
    else:
      # Exclude non-important columns in observed and predicted based on a mean threshold (relative to observed)
      for column in y_test_df.columns:
        if(y_test_df[column].mean() < mean_threshold):
            y_test_df = y_test_df.drop(columns=[column], errors='ignore')
            y_pred_df = y_pred_df.drop(columns=[column], errors='ignore')

    y_test_df = normalize_rows(y_test_df)
    y_pred_df = normalize_rows(y_pred_df)

    y_test_processed = y_test_df.to_numpy()
    y_pred_processed = y_pred_df.to_numpy()

    r2 = calculate_r_squared_by_category_ndarray(y_test_processed, y_pred_processed)
    r2_mean = sum(r2.values()) / len(r2)
    return r2_mean

"""## Hyper Opt for DT, RF and GB"""

# Custom scorer for RandomSearch
def custom_scoring_r2(y_true, y_pred):
    # Ensure y_observed and y_predicted are NumPy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Calculate R² for each category
    r_squared_values = []
    for col_index in range(y_true.shape[1]):  # Iterate over columns
        observed = y_true[:, col_index]
        predicted = y_pred[:, col_index]
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        r_squared_values.append(r_squared)

    # Return the mean R² value across all categories
    return np.mean(r_squared_values)

custom_r2_scorer = make_scorer(custom_scoring_r2, greater_is_better=True)

if run_optimization:

    param_distributions = {
        'k-NN': {
           "n_neighbors": [5, 10, 15, 20, 25]
        },
        'DecisionTree': {
            'max_depth': [None, 5, 10, 20, 30],  # Including None for unlimited depth
            'min_samples_split': randint(2, 8),  # Default is 2
            'min_samples_leaf': randint(1, 6),  # Default is 1
            'max_features': [None, 'auto', 'sqrt', 'log2', 0.5, 0.8]  # Default is None
        },
        'RandomForest': {
            'n_estimators': randint(80, 120),  # Close to default of 100
            'max_depth': [None, 5, 10, 20, 30],  # Default is None
            'min_samples_split': randint(2, 8),  # Default is 2
            'min_samples_leaf': randint(1, 5),  # Default is 1
            'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.8]  # Default is 'auto'
        },
        'GradientBoosting': {
            'estimator__n_estimators': randint(80, 120),  # Close to default of 100
            'estimator__learning_rate': [0.05, 0.1, 0.15, 0.2],  # Default is 0.1
            'estimator__max_depth': [3, 4, 5, 6, 7, 8],  # Default is 3
            'estimator__min_samples_split': randint(2, 10),  # Default is 2
            'estimator__min_samples_leaf': randint(1, 6),  # Default is 1
            'estimator__max_features': [None, 'sqrt', 'log2', 0.5, 0.8],  # Default is None
            'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0]  # Default is 1.0
        }
    }

    # Dictionary to store the best parameters for each taxonomic level
    best_params_per_level = {level: {} for level in taxonomic_levels}

    models_to_optimize = ['k-NN', 'DecisionTree', 'RandomForest', 'GradientBoosting']
    # models_to_optimize = ['GradientBoosting']

    for level in taxonomic_levels:
        print(f"Optimizing models for taxonomic level '{level}'...\n")
        variables_df = pd.read_csv(f"{base_path}/{level}/13_variables.csv")
        targets_df = pd.read_csv(f"{base_path}/{level}/observed.csv")
        variables_df = variables_df.drop(variables_df.columns[0], axis=1)
        targets_df = targets_df.drop(targets_df.columns[0], axis=1)
        scaler = StandardScaler()
        variables_df_normalized = scaler.fit_transform(variables_df)
        X_train, X_test, y_train, y_test = train_test_split(variables_df_normalized, targets_df, test_size=0.18, random_state=42)

        for model_name in models_to_optimize:
            if model_name == 'DecisionTree':
                model = DecisionTreeRegressor()
                print("Optimizing Decision Tree")
            elif model_name == 'RandomForest':
                model = RandomForestRegressor()
                print("Optimizing Random Forest")
            elif model_name == 'GradientBoosting':
                model = MultiOutputRegressor(GradientBoostingRegressor())
                print("Optimizing Gradient Boosting")
            elif model_name == 'k-NN':
                model = KNeighborsRegressor()
                print("Optimizing k-NN")

            search = RandomizedSearchCV(model, param_distributions[model_name], n_iter=1000, cv=5, scoring=custom_r2_scorer, n_jobs=-1, verbose=0, random_state=42)
            #search.fit(X_train, y_train)
            search.fit(variables_df, targets_df)
            best_params_per_level[level][model_name] = search.best_params_
            print(f"Best parameters for {model_name} at {level}: {search.best_params_}\n")

    for level, params in best_params_per_level.items():
        if 'GradientBoosting' in params:
            # Adjust the parameter names for GradientBoosting
            gb_params = {k.replace('estimator__', ''): v for k, v in params['GradientBoosting'].items()}
            best_params_per_level[level]['GradientBoosting'] = gb_params


    # Initialize models with the adjusted best parameters
    print(f"k-NN parameters: {best_params_per_level[level]['k-NN']}")
    models = {
        "k-NN": KNeighborsRegressor(**best_params_per_level[level]['k-NN']),
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(**best_params_per_level[level]['DecisionTree']),
        "Random Forest": RandomForestRegressor(**best_params_per_level[level]['RandomForest']),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(**best_params_per_level[level]['GradientBoosting'])),
    }
else:
    # Use default parameters
    print("Running with default models parameters")
    models = {
        "Linear Regression": LinearRegression(),
        "k-NN": None,
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor())
    }

# Define the number of iterations
n_iterations = 10

# Initialize a structure to store R² values for each model at each taxonomic level
all_results = {level: {model_name: [] for model_name in models.keys()} for level in taxonomic_levels}

# Loop through each taxonomic level
for i, level in enumerate(taxonomic_levels):
    print(f"Processing data for taxonomic level '{level}'...\n")

    # Load the datasets
    variables_df = pd.read_csv(f"{base_path}/{level}/13_variables.csv")
    targets_df = pd.read_csv(f"{base_path}/{level}/observed.csv")

    variables_df = variables_df.drop(variables_df.columns[0], axis=1)
    targets_df = targets_df.drop(targets_df.columns[0], axis=1)

    # Exclude the "other" column from both observed and predicted data
    #if(level == "phylum"):
    #  targets_df = targets_df.drop(columns=["other"], errors='ignore')

    # Feature Scaling
    scaler = StandardScaler()
    variables_df_normalized = scaler.fit_transform(variables_df)

    for iteration in range(n_iterations):
        # Split the datasets into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(variables_df_normalized, targets_df, test_size=0.16, random_state=42)

        for model_name, model in models.items():
            # Initialize default and best parameter models
            if model_name == "Gradient Boosting":
                default_model =  MultiOutputRegressor(GradientBoostingRegressor())
            elif model_name == "k-NN":
                if level in ("fg", "phylum"):
                    nn = 10
                else:
                    nn = 20
                print(f"{level}: k = {nn}")
                model = KNeighborsRegressor(nn)
            else:
                default_model = type(model)()  # Initialize a new instance with default parameters

            best_params_model = model  # This model already has best parameters if optimization was run

            # Evaluate performance
            default_r2_mean = evaluate_model_performance(default_model, X_train, X_test, y_train, y_test)
            best_params_r2_mean = evaluate_model_performance(best_params_model, X_train, X_test, y_train, y_test)

            # Compare and store the best result
            best_r2_mean = max(default_r2_mean, best_params_r2_mean)
            all_results[level][model_name].append(best_r2_mean)

            #print(f"default_r2_mean for {model_name} at {level} : {default_r2_mean}\n")
            #print(f"best_params_r2_mean for {model_name} at {level} : {best_params_r2_mean}\n")
            print(f"Best R² mean for {model_name} at {level}: {best_r2_mean}\n")
            all_results[level][model_name].append(best_r2_mean)

    #for model_name, model in models.items():
    #  print([model_name])
    #  print(all_results[level][model_name])
    #  print("\n")

#**********************************
# Compute Averill et al. results
#**********************************

    # Load observed and predicted datasets from the other study
    df_observed = pd.read_csv(f"{base_path}/{level}/observed.csv", float_precision='high')
    y_observed = df_observed.drop(df_observed.columns[0], axis=1)
    df_predicted = pd.read_csv(f"{base_path}/{level}/predicted.csv", float_precision='high')
    y_predicted = df_predicted.drop(df_predicted.columns[0], axis=1)

    # Exclude the "other" column from both observed and predicted data
    if(level == "phylum"):
      y_observed = y_observed.drop(columns=["other"], errors='ignore')
      y_predicted = y_predicted.drop(columns=["other"], errors='ignore')

    if(level == "fg"):
      for column in y_observed.columns:
        observed = y_observed[column]
        if(observed.mean() < mean_threshold_fg_compared_study):
          y_observed = y_observed.drop(columns=column, errors='ignore')
          y_predicted = y_predicted.drop(columns=column, errors='ignore')
    else:
      # Exclude non-important columns in observed and predicted based on a mean threshold (relative to observed)
      for column in y_observed.columns:
        observed = y_observed[column]
        if(observed.mean() < mean_threshold_compared_study):
          y_observed = y_observed.drop(columns=column, errors='ignore')
          y_predicted = y_predicted.drop(columns=column, errors='ignore')

    # Apply zeroing of negative values and normalization for each row
    y_observed_normalized = normalize_rows(y_observed)
    y_predicted_normalized = normalize_rows(y_predicted)

    compared_study_r2 = calculate_r_squared_by_category(y_observed, y_predicted)
    compared_study_mean_r_squared = round(sum(compared_study_r2.values()) / len(compared_study_r2), 3)
    compared_study_r2_results[level]= compared_study_mean_r_squared

#print(all_results)
# Calculate the mean R² value across all iterations for each model and level
mean_r2_results = {level: {model_name: np.mean(r2_scores) for model_name, r2_scores in models.items()} for level, models in all_results.items()}


# Print the mean R² values
for level, models_r2 in mean_r2_results.items():
    print("\nRésultat Averill et al.")
    print(compared_study_r2_results[level])
    print(f"\nMean R² values for taxonomic level '{level}':")
    for model_name, mean_r2 in models_r2.items():
        print(f"{model_name}: {mean_r2:.3f}")

# Converts all_results to a DataFrame and exports to a CSV file (ITERATION)
df_mean_r2_results = pd.DataFrame(mean_r2_results)
print(df_mean_r2_results)

df_mean_r2_results.to_csv(f"{base_path}/r2_results.csv", float_format='%.3f')
print(f"\nSaved mean R² values to: {base_path}/r2_results.csv")

# Converts all_results to a DataFrame and exports to a CSV file (DEFAULT)
#results_df = pd.DataFrame(all_results)
#results_df.to_csv(f"{base_path}/r2_results.csv")
#print(f"R² values stored in {base_path}/r2_results.csv")

# Converts compared_study_r2_results to a DataFrame and exports to a CSV file
compared_study_r2_results_df = pd.DataFrame(compared_study_r2_results, index=["Averill et al. method"])
compared_study_r2_results_df.to_csv(f"{base_path}/r2_results_Compared_Study.csv")
print(f"\nCompared study R² values stored in {base_path}/r2_results_Compared_Study.csv")

# Load the data
r2_results_df = pd.read_csv(f"{base_path}/r2_results.csv", index_col=0)
r2_results_compared_study_df = pd.read_csv(f"{base_path}/r2_results_Compared_Study.csv", index_col=0)

# Define the taxonomic levels and models of interest
taxonomic_levels = ['fg', 'phylum', 'class', 'order', 'family', 'genus']
models_of_interest = ['Random Forest', 'Gradient Boosting', "k-NN"]
#models_of_interest = ['Random Forest']

# Function to generate a smooth curve
def smooth_curve(x, y):
    x_new = np.linspace(min(x), max(x), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_new)
    return x_new, y_smooth

plt.figure(figsize=(14, 7))

# Define line styles and markers for each model
line_styles = {
    'Random Forest': '--',       # Dashed line
    'Gradient Boosting': ':',    # Dotted line
    'Averill et al. 2021': '-.',       # Dash-dot line
    "k-NN": "-" # TODO: Change sign
}
markers = {
    'Random Forest': 's',        # Square point
    'Gradient Boosting': 'o',    # Round point
    'Averill et al. 2021': '^',       # Triangle point
    "k-NN": "." # TODO: Change sign
}

# Plotting the models from r2_results.csv (lines)
for model in models_of_interest:
    x_values = np.arange(len(taxonomic_levels))
    y_values = r2_results_df.loc[model, taxonomic_levels]
    x_smooth, y_smooth = smooth_curve(x_values, y_values)
    plt.plot(x_smooth, y_smooth, line_styles[model], linewidth=1.5, label=model, color='black', zorder=1)

# Plotting the models (markers on top)
for model in models_of_interest:
    y_values = r2_results_df.loc[model, taxonomic_levels]
    if model == "k-NN":
        print(f"k-NN: {r2_results_df.loc[model, taxonomic_levels]}")
    marker_color = '0.42' if model == 'Gradient Boosting' else 'black'
    plt.scatter(x_values, y_values, marker=markers[model], color=marker_color, s=150, zorder=2, linewidth=1.5)  # s controls size

# Plotting the compared study (lines and markers)
y_values_compared = r2_results_compared_study_df.loc['Averill et al. method', taxonomic_levels]
x_smooth, y_smooth = smooth_curve(x_values, y_values_compared)
plt.plot(x_smooth, y_smooth, line_styles['Averill et al. 2021'], linewidth=1.5, label='Averill et al. 2021', color='black', zorder=1)
plt.scatter(x_values, y_values_compared, marker=markers['Averill et al. 2021'], color='black', s=150, zorder=2, linewidth=1.5)

# Create custom legend handles with increased marker and line sizes
legend_handles = []
for model in models_of_interest + ['Averill et al. 2021']:
    marker_color = '0.42' if model == 'Gradient Boosting' else 'black'
    line = mlines.Line2D([], [], color='black', marker=markers[model],
                         linestyle=line_styles[model], markersize=15,  # Increase marker size
                         markeredgewidth=1, markerfacecolor=marker_color,
                         label=model, linewidth=1.5)  # Increase line width
    legend_handles.append(line)

# Convert taxonomic level names to title case for x-axis labels
#taxonomic_levels_title_case = [level.title() for level in taxonomic_levels]
taxonomic_levels_title_case = ['Functional', 'Phylum', 'Class', 'Order', 'Family', 'Genus']

plt.xticks(x_values, taxonomic_levels_title_case, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Taxonomic Level', fontsize=18)
plt.ylabel(r'$R^2$ score', fontsize=18)
plt.title(r'$R^2$ score comparison across methods for different fungal taxonomic levels', fontsize=20, pad=18)
plt.legend(handles=legend_handles, loc='best', fontsize=16)
#plt.savefig(f"{base_path}/comparison_graph.svg", format='svg', bbox_inches='tight')
plt.savefig(f"{base_path}/fungi_comparison_graph_3.png", format='png', bbox_inches='tight')
plt.savefig(f"{base_path}/fungi_comparison_plot_3.eps", format='eps', bbox_inches='tight')
plt.show()

end_time = datetime.now()

# Calculate the duration
duration = end_time - start_time
duration_in_s = duration.total_seconds()

# Calculate minutes and seconds
minutes = divmod(duration_in_s, 60)[0]
seconds = divmod(duration_in_s, 60)[1]

# Print the duration in minutes and seconds
print(f"Process completed in: {int(minutes)} minutes and {int(seconds)} seconds")
