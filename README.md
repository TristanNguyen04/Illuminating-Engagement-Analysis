# Executive Summary
Lighting conditions play a crucial role in influencing user engagement levels. However, the existing
lighting control systems (at our site) often utilize generic, one-size-fits-all approaches, failing
to account for the nuances of individual user preferences and environmental factors. This project
tackles this challenge by leveraging a data-driven approach, aiming to develop an intelligent system
capable of accurately predicting user engagement levels based on prevailing lighting conditions. The
ultimate goal is to enable strategic lighting adjustment and optimization, maximizing user engagement
through customized lighting setups.

Several key assumptions underpin the proposed solution. First, the collected data on lighting
conditions (light intensity, colour temperature, and ambient noise levels) and user engagement metrics
must be representative of the target user population and environment. Second, the data must
be free from significant errors, outliers, and biases that could skew the analysis. Furthermore, it is
assumed that the multilayer perceptron (MLP) neural network architecture can effectively model the
complex, nonlinear relationships between the independent lighting variables and the dependent user
engagement variable. Additionally, it is assumed that the network architecture, hyperparameters,
and training process have been optimized to achieve robust and accurate predictions, and that the
trained model can generalize well to new, unseen data, providing reliable predictions for enhancing
user experiences.

The chosen solution method employs an MLP neural network model with an input layer, two
hidden layers (first layer within 10 neurons and second layer within 5 neurons) and an output layer.
The model learns a nonlinear mapping function, parametrized by weight matrices and bias vectors,
to map the input features (light intensity, colour temperature, and noise level) to the target user
engagement level. The model parameters are optimized using a mean squared error loss function and
gradient-based optimization methods, such as forward-propagation and stochastic gradient descent.
Once trained, this MLP model can make predictions on new, unseen data by forward-propagating
the input features through the network to obtain the estimated user engagement levels.
The trained MLP model achieved promising results on both the training round (Mean Squared Error :
0.015, R2 : 0.828) and testing round (Mean Squared Error : 0.016, R2 : 0.825) of datasets, demonstrating
its ability to capture the intricate relationships between lighting conditions and user engagement
levels. Various visualizations, including 3D plots, contour plots, and partial dependence
plots, were generated to assess the model’s performance and interpretability. These visualizations
provide insights into the model’s predictions and the influence of individual input features on the
predicted engagement levels.

The proposed MLP model exhibits several strengths, including the ability to model complex,
nonlinear relationships, a flexible and adaptive architecture facilitated by multiple hidden layers and
nonlinear activation functions, and a data-driven approach that uncovers insights directly from the
collected dataset. However, a key weakness is the model’s dependence on the quality and representativeness
of the collected data, as well as the potential for overfitting or poor generalization if
not properly regularized or trained with diverse data. Future improvements could involve exploring
ensemble methods by combining multiple MLP models, implementing advanced regularization
techniques, tuning hyperparameters through techniques like grid search or Bayesian optimization,
collecting more real-world data across diverse lighting scenarios and user demographics, and comparing
the MLP’s performance against other model architectures like decision trees or gradient boosting
machines to identify the most comprehensive and scalable approach.
