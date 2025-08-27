# iris-ml-gravitation-solver
This endeavors to use the concepts of classical Newtonian gravitation to classify samples from the Iris machine learning dataset. This aims to be more of an educational endeavor rather than a practical one. When we look at machine learning packages e.g. sklearn, they may be able to solve the problems quicker, but it is generally not easy to see what goes behind the scenes. This endeavor does not optimize for speed, but it aims to show what goes behind the scenes in this self-proposed machine learning classification technique.

# Foundation
According to [Newton's Law of Universal Gravitation](https://en.wikipedia.org/wiki/Newton%27s_law_of_universal_gravitation), two point masses exert an attractive force on each other proportional to their masses, and inversely proportional to the squared distance in between them.

$$F\propto\frac{m_{1}m_{2}}{r^{2}}$$

This endeavor intends to apply the same to training and testing data points in a machine learning model. Since the mass (or in more machine learning and less physics-accurate terms, weight) of the testing point is not consequential to the classification of the testing point, we can take the mass of the testing point $m_{2} = 1$, and so the mass of the training point can be denoted as $m_{1} = m$

It might actually be more computational more efficient and non-consequential to the final classification task if we drop the square and consider only the inverse of the distance between the sample points. This actually brings about another concept used in physics, and that is the gravitational potential $V(r)$. Another added benefit is that as gravitational potential is a scalar quantity, the individual gravitational potentials may be summed up more simply compared to gravitational force.

Gravitational field strength is related to the gravitational force by

$$g(r):=\frac{F}{m_{1}}=\frac{Gm}{r^{2}}$$

and is related to gravitational potential by being its derivative over the distance between the masses

$$g(r)=\frac{dV(r)}{dr}$$

$$V(r)=\int g(r) dr=\int \frac{Gm}{r^{2}} dr = -\frac{Gm}{r}+C$$

The gravitational potential is defined at zero when the masses are infinitely far apart from each other. This is useful as in the case of machine learning, if the training and testing points are infinitely far apart from each other, the probability that the test point belongs to the class of the training point is zero (which is an extreme case).

$$\lim_{r\rightarrow\infty}V(r):=0$$

$$\lim_{r\rightarrow\infty}\left\[-\frac{Gm}{r}+C\right\]=0$$

The antiderivative constant will then be zero

$$C=0$$

$$V(r)=-\frac{Gm}{r}\propto\frac{m}{r}$$

We can model a probability function discounting [the second Kolmogorov axiom](https://en.wikipedia.org/wiki/Probability_axioms#Second_axiom):

$$p(X)=\sum_{\text{for all }i}\frac{m_{i}}{r_{X,i}}$$

Applying the second axiom $P(\Omega)=1$, we can model the probability as

$$P(X)=\frac{1}{\sum_{\text{for all }x}p(X=x)}p(X)$$
