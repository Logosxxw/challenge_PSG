### **Your mission** 

Train an algorithm (Python or R) on the first part of the season that we will test on the second part. 15 minutes of a game will be randomly selected in the second half of the season. Your algorithm will have to return the identity of a player who has performed certain actions as well as predictions about the next event to take place on that game.
 

### **At your disposal**

You will have in your hands a learning database consisting of the F24 Opta files accurately describing all the ball events in all the games of the first part of the Ligue 1 season, 2016-2017.



### For the prediction

**In the resources of your participation area, you will find a file with an explanation of the databases. The following procedure will be applied several thousand times to your algorithm:**

- We randomly choose a match from the test database *(F24 Opta files of all matches in the second half of the 2016-2017 Ligue 1 season).*
- We randomly choose the first or second half for this match.
- We randomly pick 15 minutes of the selected halftime (all events between *t* and *t+15* minutes with *t*randomly selected).
- We replace all the names of the teams by "**1**" (home) or "**0**" (away). In the files, it means we replace "**team_id**" by "**1**" or "**0**".
- We delete all the players IDs and write "**0**" instead, except for one randomly chosen player *(who played more than 800 minutes on the learning dataset and did not change team in January)*. We write "**1**" for the ID of this specific player. In the files, it means we replace "**player_id**" by "**1**" or "**0**". In addition, when: Type ID=140, Type ID=141, qualifier_id=140 or qualifier_id=141 appears, we replace the values by " ".
- We delete all position information (y, x) for all events except the last 10 Opta events within 15 minutes. In the files, this means that we replace "**y**" with "**0**" or "**x**" with "**0**".
- We will also remove everything that is written in the F24 files before the first OPTA event. We also replace the values of Event timestamp, Event id, Q id and version with " ".
- We reduce some of the information of the last 10 events. In the files it means:
  \- For the last 10 events we replace "**outcome**" by " "
  \- For the last 10 events we get rid of information about qualifier_ID 
  So : we replace the "**value**" by " " and we replace all the "qualifier_id" by " ".



### **Your deliverable**  

 

**Your algorithm (python or R) applied to the test base should allow you to:**  

1. Find out the identity of player 1.
2. For the next Opta event (first Opta event after t+15), find out if the team will be 1 (home) or 0 (outdoors).
3. For the next Opta event, find the associated position (y,x).  

 

**The algorithm must return a CSV file without a header whose components are four real numbers:**

1. Player ID.

2. 1 or 0 for the team at home or away.

3. Y.

4. X.

   

### **Evaluation criteria**

50% of the assessment will be based on the first question and 25% on each of the other two. The success score for the first 2 tasks will be the percentage of correct answers. For the last task an average error in standard L2 will be calculated. The final ranking will be based on the weighted average of the rankings for the 3 tasks.