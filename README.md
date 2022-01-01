# ReinforcmentLarningPU

DqnModel.py is the actuall, reinforcment learning part.

You can see here:
https://www.youtube.com/watch?v=iUfu2p9vwkQ

game.py
is the definition of the "enviroment", for the agent,
i use maze generation algorithm in order to create a maze, and then destroy 30% of the walls by random chance in order to get the enviroment seen in the video.

Game goal:
in a 150 action period get as many "food" as you can, the game ends if you collide with an enemy or time ends.
the enemies always make a valid move but if an enemy can continue going forward it will go forward with about 0.8 probability, and about 5% chance to change for any other direction.

at least as of now:
the models saved in models/dqn are either 50k trained or 150k trained,
where trained on a 9x9 enviroment, but would work on any size ( it works well with 11x11, bigger enviroments would probably wont work with the trained models ).
the number of "food" sources can be increased and the number of enemies, the model was trained on 5 food 2 enemies scenarios.
