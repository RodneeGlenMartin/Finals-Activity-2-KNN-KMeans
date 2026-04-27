# k-NN Classification: Manual Computation Guidelines

## Dataset and New Point
* Identify the existing dataset with its features and known class labels.
* Identify the new data point that needs to be classified.
* The algorithm uses the Euclidean distance formula (expandable for $n$ features): $\sqrt{(X_{2}-X_{1})^{2}+(Y_{2}-Y_{1})^{2}}$
    * $X_{2}$ and $Y_{2}$ are the features of the new entry.
    * $X_{1}$ and $Y_{1}$ are the corresponding features of the existing entry.

## Compute for the Distance 1
* Calculate the exact Euclidean distance between the new data point and the very first row in the dataset.

## Compute for the Distance 2
* Calculate the distance between the new data point and the second row in the dataset. 
* Repeat this calculation iteratively for every single row remaining in the existing dataset.

## Sort in Ascending Order
* Once the distances for all rows have been computed, sort the entire dataset based on the newly calculated distance values from lowest to highest.

## Choose K?
* Identify the predetermined integer value for $k$.
* Select the top $k$ rows from the sorted dataset (the ones with the shortest distances).
* Count the class labels among these $k$-nearest neighbors to determine which class holds the majority vote.

## Updated Dataset
* Assign the new data point to the class that won the majority vote.
* Append the newly classified data point to the final dataset.