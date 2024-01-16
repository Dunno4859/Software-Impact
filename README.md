# Software-Impact
A ML-MCDM-based decision support system for optimal route selection in autonomous urban delivery
In this paper, we have integrated two recently developed MCDM methods, called the Alternatives Ranking with Elected Nominee (ARWEN) and the Win-Loss-Draw (WLD) methods. ARWEN serves as a ranking method to prioritize options, while WLD is employed as a subjective weighting approach for evaluating the importance of different criteria. The proposed DSS is designed to identify the most efficient and safest route for autonomous delivery vehicles in urban settings. The structure of the paper is as follows: Section two provides a theoretical background on the DSS; section three outlines the proposed theoretical framework of the DSS; section four applies the DSS to a case
study; section five focuses on validating the DSS's performance; section six presents the Python codes used for constructing the DSS; and finally, section seven offers conclusions and directions for future research.

#1- Theoretical background
In this section, ARWEN and WLD methods are introduced.

1.1- ARWEN
the ARWEN method is developed to identify the most suitable option based on the smallest rate of
change, as opposed to selecting an elected nominee. The method has been further elaborated into four variants, each based on the type and extent of information that decision-makers have at their disposal. The ARWEN’s algorithm’s fundamental selection process is based on the larger value of Γi

#1.2. WLD method
The WLD method is developed based on the assumption that decision-makers have complete information regarding the criteria [26]. This straightforward weighting approach assigns two distinct importance weights for criteria evaluation, mimicking human behavioral patterns in assessing these criteria. The process of the WLD method is outlined in the following steps: 

Step 1. The first step is to evaluate the criteria in terms of their importance by employing a     scale, in which the upper and lower bounds are 1,10, and the center is 5. Decision-makers can     choose   any rational numbers between the mentioned numbers.
  
Step 2. The second step is to establish the pairwise comparison matrix.

Step 3. Calculation of the final weights is the third step of the WLD algorithm.

#2. Theoretical Framework
The proposed model is architected on three main variables, the weights computed by the WLD methods, the weights computed by the ML random forest employed to analyze the trends and ranks of the routes using ARWEN to generate.



  

