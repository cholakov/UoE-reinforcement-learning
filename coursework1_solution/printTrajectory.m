%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STUDENT SUPPORT FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRINT TRAJECTORY (Exercise 3) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Visually inspect that policy works as expected. Simulates agent behavior 
%% when following policy defined by $policy$. The code in this function has 
%% been borrowed from the original exercise.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function PrintTrajectory(...
  MDP, policy, episodeLength, probabilityOfUniformlyRandomDirectionTaken)

  currentTimeStep = 0 ;
  currentMap = MDP ;
  agentLocation = currentMap.Start ;
  % Keeping record of initial location.
  startingLocation = agentLocation ;
  % Keep track of agent movement history:
  agentMovementHistory = zeros(episodeLength+1, 2) ;
  agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;

  realAgentLocation = agentLocation ; % The location on the full test map.
  Return = 0;

  for i = 1:episodeLength
      actionTaken = policy( realAgentLocation(1), realAgentLocation(2) );
      % $actionMoveAgent$ is used to simulate agent (the car) behaviour.
      [ agentRewardSignal, realAgentLocation, currentTimeStep, ...
          agentMovementHistory ] = ...
          actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
          currentTimeStep, agentMovementHistory, ...
          probabilityOfUniformlyRandomDirectionTaken ) ;
      Return = Return + agentRewardSignal;  
  end
  currentMap = MDP ;
  agentLocation = realAgentLocation ;
  Return
  printAgentTrajectory
end