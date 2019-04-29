Dear organizers, my name is Xiaowen. I'm from Shanghai and you can call me logos. I work as a senior data mining engineer at ctrip.com which is the second largest Online Travel Agency platform all over the world. I'm experienced in building machine learning data products and I have great passion for football data analysis. No matter what the outcome of the challenge, I will study football data as my long-term career.

I have uploaded my code to GitHub. Could you please open the link: <https://github.com/Logosxxw/challenge_PSG>

there are detail project scripts and feature descriptions there.

###I think the main highlights of my algorithm are: 

1. Construct some statistical features that describe the style of the player or team 
2. Collect as many samples as possible by sliding window on each game data 
3. Find the rnn model with football event sequence can significantly improve the prediction accuracy of which team will the next event belongs to.



Here is my modeling process:

### The first thing is to analyze the problem and select the math tool. 

The question is predicting a player or predicting the team and coordinates of the next event based on a 15-minute opta game event xml. I think machine learning is a suitable tool because football is so complicated that it is hard to calculate an analytical solution. However, there are many football games, and a large number of learning samples could be constructed for training a good machine learning model.
So, we can use the multi-classification model to predict players, use binary classification model to predict which team the next event will belongs to and use regression model to predict the xy-coordinate of the next event. 

Besides, I think we can also add intermediate models. For example, to predict players, one can firstly predict which team the player belongs to and what position the player play and finally predict the specific player. The intermediate models can also help predict the team of next event. For example, if the home team is predicted as a team willing to control the ball and has a high pass success rate. The last five events are mainly the home team's possession. Then the team which the next event belongs to is most likely to be the home team.

Based on the above analysis, I divided the models into two categories. 

First category includes models of predicting players or teams based on the performance within the 15 minutes. This kind of model should mainly focus on how to construct useful statistical features that describe the style of the player or team well and traditional machine learning model could be used to  enhances the generalization ability.

Second category includes predicting team of next event and xy of next event. This kind of model are characterized by a sequence of events, It is natural to think of using a sequence-based RNN model such as GRU or LSTM to solve this problem.



### Determine the solutions of the 3 problems, then data exploration and feature construction is the key process. 

By the way, I think the core of data mining is amost always the feature engineering.

I read the opta explanation documentation first, try to understand the structure of the xml file and what events opta has recorded. Then I builded up a event.tsv and a qualifier.tsv which contains event id and event explanation, then I join the meaning of events and qualifiers to the original game xml file so that converting the original xml file to a user-friendly file to browse. 

At the same time, I picked a match in the 17th round of Paris vs. Nice. while watching the game, I check the opta xml file and look for inspiration for constructing features. Throughout the project, more than 80% of my time and energy is spent on data exploration and feature construction.

I mainly construct features from three categories which is passing, shooting and other events.

####Football is a passing game,

I found that players can be distinguished from passing stat. For example, the midfielder's passing direction is comprehensive, such as Verratti, he would pass forward or backward, pass to the left side or to the right side. While a wing player are restricted by the field, there are few opportunities for a right back player to pass the ball to the right. The passing length is also very important. The central defender often needs bigfoot clearance while a technical players tend to make short passes. Therefore, I converted the qualifier 213 to the pass direction and converted qulifier 212 to the passing length classification including short pass, medium distance pass and long distance pass. In addition, whether the pass is a chipped or a cross or a through are all in consideration. I would calculate the passing statistics for every players and teams within each 15 minutes. You could check all my features in my GitHub repo.

####Shoot is the key point for wining the game. 

On the one hand, the shooting statistics of the two teams can reflect which team is dominat. On the other hand, strikers has more chances to shoot than the defender. While centre back like Thiago Silva may often win a head shot chance from the corner. So I also constructed a series of shooting features, such as the number of shots, number of shots on target, number of shoot on head and so on.

Other features include number of corner kicks, free kicks, offside counts, tackles,  interceptions, saves, fouls and so on, all of which help the model find a distinctive player or team.

All of the above features apply to the first type of model I mentioned before, predicting players or teams through technical statistics within 15 minutes.

####For the second type of sequence-based rnn model, 

I need to do some transformation to the last ten events so as to encode them. For example, one encoded event would be 1 dash Pass dash middle dash 2. The first number one represent it is a home event. next the Pass is the event type. next the middle is the area where the event occurred. I personally divided the field into 9 areas which is back left, back right, back box, back out of box, front left, front right, front box, front out of box and the middle field. And the last number 2 means the game period which is divided into 4 segments, the performance of teams will be different in different periods due to fatigue. After encoding each last 10 event, I could use the similar text classification method to train rnn for the next event.

###Because I have constructed a lot of features, and I want to use the deep learning model,

I hope that the larger the sample size, the better. I used the window sliding method to construct the sample. I slided a 15 minutes window on each game. Every time, only move forward one event time to generate a new sample. For example, the first samle might start from 0 second and end at 15 minutes 0 second and the next sample will start from 3 seconds and end at 15 minutes 3 seconds. In this way, I constructed more than 350,000 samples from all 19 rounds matches.
Importantly, collecting samples in this way will result in strong correlation between neighbour samples, especially when predicting players, it is likely that the technical statistics of one player on two neighbour windows are exactly the same, because not every players touch ball often, so I did some sampling on the player samples. The player samples were taken every 2 minutes instead of few seconds.

At the same time, when selecting model parameters through gridsearch, I did not use the conventional scheme to randomly select the training set and the validation set. Instead, I selected all the matches in the 19th round as the validation set, and the first 18 rounds were used as the training set. I need to avoid the neighbour window samples fall into training set and validation set respectively which would lead to a over-fitting.

###I didn't know about this challenge until April, 

so my time was very tight, and I finished the sample construction work and started the model tunning at the day before the game deadline. In terms of training speed, I chose to use GBDT as the model. There is a package called xgboost which performing parallel computing good. The training speed is very fast, and the tree-based model can automatically find the appropriate segmentation points for continuous variables. It saves a lot of feature transformation work. 

Finally, I submitted the algorithm before the deadline, but many optimizing ideas are too late to implement, such as training the team model and the player's on-site position model first, then using the output of these two models as the features of the player predicting model. 

However, after the deadline for submission, I continued to compare the effects of gbdt and rnn. I was pleasantly surprised to find that when predicting the team of the next event, the effect of training rnn with sequence features is better than gbdt model with statistics features. The accuracy of the validation set is 82% vs 78%, and the improvement is very significant. In addition, I also tried to add statistics features to concat in the hidden layer of rnn, but the effect has not been significantly improved. That is the event sequence itself already contains a wealth of information.

###In conclusion

I use two types of features, technical statistics  and event sequence. The accuracy of predicting players is 9%. The accuracy of predicting next team is 82%. The MAE of next x is 16.8 of next y is 24.66. And there are lot of optimizing strategies to implement.

###Finally, inspired by this game, divergent thinking

Maybe we could predict the probability of scoring with different offensive tactics:
I find that rnn has achieved good results in the prediction of the next event, we can make the offensive tactics into different sequence of events, and meanwhile we could put the opponent's playing style into the model, if the model fit the probability of a goal well, then this model should be able to assist the coach to design different offensive strategies against different styles of teams.