
    The Coefficient of Determination
    
The coefficient of determination is also known as
the r-squared metric which is used to test the 
goodness of fit for a regression based algorithm.

The value of r-squared is bounded in the limit of
[0 to 1]

A value closer to 1 signifies goodness of fit or 
simply a better model.

A value closer to 0 signifies underfitting or poor
model.

The score() function from sklearn library computes
the r-squared value in case of regression based
algorithms.

                Problems with R-squared
        
Input                       Output              R-squared
        
YOE                         Salary                 0.85
YOE, EQ                     Salary                 0.90
YOE, EQ, CS                 Salary                 0.92
YOE, EQ, CS, No of Hairs    Salary                 0.9202
YOE, EQ, CS, Hairs, Teeth   Salary                 0.93

The value of r-squared is bound to increase if there is
an increase in the number of predictors (features, columns).

If the model adds a strong predictor the value of r-squared
would increase exponentially or linearly.

But if the model adds a weak predictor the value of r-squared
would witness a miniscule increase.

To deal with this problem, mathematicians have devised a 
new metric very smartly known as the Adjusted R-squared
value.

                Adjusted R - Squared
                
Adjusted R-squared is a carefully crafted metrics which
punishes the model for adding irrelevant features by lowering
its value and rewards the model for adding strong predictors
by increasing its value.



































