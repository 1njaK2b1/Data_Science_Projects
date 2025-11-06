machine learning (ML)

1. Purpose of Evaluation
	•	Measure performance: Quantify how accurately a model predicts or classifies unseen data.
	•	Prevent overfitting: Ensure the model generalizes beyond the training set.
	•	Guide model selection: Compare models, hyperparameters, or feature representations.
	•	Monitor improvement: Track progress during iterative development.

Evaluation Techniques

Cross-Validation
	•	Split data into k folds.
	•	Train on k–1 folds, test on the remaining one.
	•	Repeat for all folds and average performance.
	•	Reduces variance and bias from data partitioning.

Confusion Matrix

A 2×2 (or multi-class) table comparing predicted vs. actual labels:

Helps analyze specific types of errors.

ROC Curve and AUC
	•	ROC Curve: Plots True Positive Rate vs. False Positive Rate.
	•	AUC: Scalar summary of overall classification quality (1 = perfect, 0.5 = random).

⸻

Common Pitfalls
	1.	Data leakage — using test data in training or tuning.
	2.	Imbalanced datasets — misleading accuracy; use precision/recall or AUC instead.
	3.	Overfitting to validation data — excessive hyperparameter tuning.
	4.	Ignoring uncertainty — lack of confidence intervals or variance estimates.

Measure agent performance (e.g., Pacman search agents, reinforcement learning).
	•	Compute accuracy for classifiers (e.g., Naive Bayes, Perceptron).
	•	Assess expected utility in decision-making problems.
	•	Compare policies in reinforcement learning through average reward or discounted return.

