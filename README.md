# Easy21-RL
solutions to [David Silver's RL course project Easy21](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/Easy21-Johannes.pdf)

Class notes and some writeup on [my website](http://kvfrans.com/model-free-prediction-and-control/)

# Writeup:

**What are the pros and cons of bootstrap?**

Bootstrapping has less variance, since it depends on the current reward and intermediate steps more than only the end result.
However, there may be some bias, while Monte-Carlo is guaranteed to converge given enough examples.

**Would you expect bootstrapping to help more in blackjack or Easy21?**

Easy21 has red cards that can move your total sum backwards, so its episodes will last longer overall than blackjack. Therefore, bootstrapping will help more than in blackjack, where Monte-Carlo methods will work well since the episodes are short.

**What are the pros and cons of function approximation in Easy21?**

Function approximation reduces the learning time since we have less variables we need to learn. However, we lose some of the precision that was possible in a simple table lookup since the features are generalized.

**How would you modify the function approximator suggested in this section to get better results in Easy21?**

The overlapping regions led to bad results, as a certain state could trigger multiple features, and the sum of their weights could lead to values greater than one, which is impossible to achieve in the game itself. The overlapping regions should be removed, as they do not bring any significant advantage.
