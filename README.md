# Software-Impact
A ML-MCDM-based decision support system for optimal route selection in autonomous urban delivery
In this paper, we have integrated two recently developed MCDM methods, called the Alternatives Ranking with
Elected Nominee (ARWEN) and the Win-Loss-Draw (WLD) methods. ARWEN serves as a ranking method to
prioritize options, while WLD is employed as a subjective weighting approach for evaluating the importance of
different criteria. The proposed DSS is designed to identify the most efficient and safest route for autonomous delivery
vehicles in urban settings. The structure of the paper is as follows: Section two provides a theoretical background on
the DSS; section three outlines the proposed theoretical framework of the DSS; section four applies the DSS to a case
study; section five focuses on validating the DSS's performance; section six presents the Python codes used for
constructing the DSS; and finally, section seven offers conclusions and directions for future research.

1- Theoretical background
In this section, ARWEN and WLD methods are introduced.

1.1- ARWEN
the ARWEN method is developed to identify the most suitable option based on the smallest rate of
change, as opposed to selecting an elected nominee. The method has been further elaborated into four variants, each based on the type and extent of information that decision-makers have at their disposal. The ARWEN’s algorithm’s fundamental selection process is based on the larger value of Γi

1.2. WLD method
The WLD method is developed based on the assumption that decision-makers have complete information regarding the criteria [26]. This straightforward weighting approach assigns two distinct importance weights for criteria evaluation, mimicking human behavioral patterns in assessing these criteria. The process of the WLD method is outlined in the following steps: 

Step 1. The first step is to evaluate the criteria in terms of their importance by employing a     scale, in which the upper and lower bounds are 1,10, and the center is 5. Decision-makers can     choose   any rational numbers between the mentioned numbers.
  
Step 2. The second step is to establish the pairwise comparison matrix.

Step 3. Calculation of the final weights is the third step of the WLD algorithm.

2. Theoretical Framework
The proposed model is architected on three main variables, the weights computed by the WLD methods, the weights computed by the ML random forest employed to analyze the trends and ranks of the routes using ARWEN to generate.

3. The DSS Codes
3.1. Computing criteria weights
The following codes display the implementation of the aggregation of the weights using python:

import numpy as np
# Assume w_j^ML and w_j^WLD are numpy arrays containing the weights from ML
and MCDM methods
# Also, assume theta and vartheta are the coefficients for ML and MCDM weights
respectively
def aggregate_weights(w_j_ML, w_j_WLD, theta, vartheta):
# Ensure that theta + vartheta = 1
if not np.isclose(theta + vartheta, 1):
raise ValueError("The coefficients theta and vartheta must sum up to
1.")
# Compute the aggregated weights according to the equation
W_j = (vartheta * w_j_ML + theta * w_j_WLD) / (vartheta * w_j_ML / 2 +
theta * w_j_WLD / 2)
return W_j
# Example usage with hypothetical weights and coefficients
w_j_ML = np.array([0.2, 0.3, 0.5]) # replace with actual ML weights
w_j_WLD = np.array([0.4, 0.4, 0.2]) # replace with actual MCDM weights
theta = 0.6 # coefficient for ML weights
vartheta = 0.4 # coefficient for MCDM weights
# Calculate the aggregated weights
W_j = aggregate_weights(w_j_ML, w_j_WLD, theta, vartheta)
print('Aggregated Weights:', W_j)

The “aggregate_weights” function combines the ML weights, WLD weights, and their corresponding coefficients, θ and θ, to determine the combined weights for each criterion.
Moving forward, we will proceed with the implementation of the ML component, which involves computing the weights through a Random Forest classifier using a ‘imaginary database A.’ Typically, historical data with route criteria measurements and corresponding success or quality labels would be used for this purpose. However, since we do not have access to the data, we will simulate this process using random numbers.

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# Simulate a dataset (Imaginary Database A)
# Let's create a dataset with 1000 samples, each sample with 5 features
(criteria)
# The target variable is binary, indicating the success (1) or failure (0) of
the route
X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
n_redundant=2, random_state=42)
# Instantiate the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# Fit the model on the entire dataset (since this is an imaginary dataset)
rf_classifier.fit(X, y)
# Extract feature importances which act as weights from ML
ml_weights = rf_classifier.feature_importances_
# Normalize the ML weights to sum to 1 for further processing
ml_weights /= np.sum(ml_weights)
print("ML Computed Weights:", ml_weights)

The Random Forest classifier is employed on the simulated dataset. In the above example, the criteria used to evaluate the routes are Traffic Conditions, Road Quality, Safety Rating, Environmental Impact, and Scenic Value.

Table 1. the criteria and their random weights
Citeria                 Weights
Traffic Conditions      0.2481
Road Quality            0.3912
Safety Rating           0.0801
Environmental Impact    0.1683
Scenic Value            0.1123

The weights shown in Table 2 reflect the relative importance of each criterion as learned from the simulated historical data. In an actual implementation, the real historical data ought to be used to determine these weights. Now ML computed weights are available, the process proceeds to integrate them with the MCDM computed weights using the aggregation formula and codes provided earlier.
In the code provided above, we simulated an "imaginary database A" by creating a synthetic dataset using the “make_classification” function from scikit-learn1. This function generates a random n-class classification problem, which in this context, we used to represent different routes with associated criteria and outcomes. This synthetic dataset stands for real historical data might have be in an actual "database A." In a real-world application, "database A" must be an actual database containing historical records of route selections and their success or failure outcomes. The features (criteria like Traffic Conditions, Road Quality, etc.) and labels (outcomes of the routes) in "database A" would be used to train the ML model, such as the Random Forest classifier used in the example.
The make_classification function is a placeholder to demonstrate how you would use historical data to extract criteria weights with ML. In practice, the synthetic data must be replaced with the actual dataset, which might be similar to the following codes:

# Placeholder for loading your actual database A
# X_db_A would be the features matrix from your database A
# y_db_A would be the outcomes (labels) from your database A
X_db_A, y_db_A = load_your_database() # This function would be defined by you
to load your data
# Then, you would train the Random Forest classifier on your actual data
rf_classifier.fit(X_db_A, y_db_A)

It is worth noting that the “load_your_database()” function is hypothetical and would need to be implemented according to how the actual data is stored and needs to be processed.

3.2 Ranking the routes Computing criteria weights 
As displayed in Figure 1 and explained in the theoretical framework section, ARWEN algorithm is employed to rank the routes. Below is the Python code for implementing the ARWEN method:

import numpy as np
def arwen_method(performance_matrix, aggregated_weights):
"""
Apply the ARWEN MCDM method to select the optimal route.
:param performance_matrix: Matrix where each row represents a route and
each column a criterion performance.
:param aggregated_weights: Aggregated weights for each criterion.
:return: The index of the optimal route and the Gamma values for all
routes.
"""
n_criteria = performance_matrix.shape[1]
n_routes = performance_matrix.shape[0]
# Initialize the Γ values for each route
gamma_values = np.zeros(n_routes)
# Calculate the Γ value for each route
for i in range(n_routes):
# Calculate the sum of weighted performance for each criterion
sum_weighted_performance = 0
for j in range(n_criteria):
max_performance = np.max(performance_matrix[:, j])

r_ij = performance_matrix[i, j]

sum_weighted_performance += aggregated_weights[j] *
(max_performance / r_ij)
gamma_values[i] = (2 * n_criteria) - sum_weighted_performance
# The optimal route has the largest Γ value
optimal_route_index = np.argmax(gamma_values)
return optimal_route_index, gamma_values
# Example usage:
# Define the performance matrix for each route and criteria
performance_matrix = np.array([
[0.8, 0.9, 0.7, 0.6, 0.9], # Route 1 performances for each criterion
[0.9, 0.8, 0.8, 0.7, 0.8], # Route 2 performances for each criterion
[0.7, 0.9, 0.6, 0.8, 0.7] # Route 3 performances for each criterion
])
# Define the aggregated weights for each criterion
aggregated_weights = np.array([0.2, 0.3, 0.1, 0.2, 0.2]) # Example aggregated
weights
# Calculate the optimal route using ARWEN
optimal_route_index, gamma_values = arwen_method(performance_matrix,
aggregated_weights)
# Output the optimal route index and the Gamma values for all routes
print(f"The optimal route index is: {optimal_route_index}")
print(f"Gamma values for each route: {gamma_values}")

This code defines the “arwen_method” function, which can be utilized with the data embedded in the decision matrix to select the optimal route. The above codes also include an example of how to call this function using a hypothetical “performance_matrix” and “aggregated_weights”.



  

