%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STUDENT SUPPORT FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POLICY ITERATION (Exercise 3) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A function which find an optimal policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [stateValues, betterPolicy] = policyIteration( ...
  blockSize, episodeLength, MDP, policy, num_value_iterations, num_policy_iterations)

  % Initialize parameters %
  num_states = length(policy); % Number of states in the MDP
  betterPolicy = policy;

  % Number of times to run policy iteration
  for i=1:num_policy_iterations

    % Evaluate the current policy
    stateValues = policyEvaluation(blockSize, episodeLength, MDP, ...
      betterPolicy, num_value_iterations);
    stateValuesMatrix = reshape(stateValues, [blockSize, length(stateValues)/blockSize])'

    % Number of states
    for j=1:num_states
      currState = j;
      bestNextStateValue = -Inf;
      bestAction = -Inf;

      % Get possible transitions from currState
      [ possibleTransitions, ~ ] = ...
        MDP.getTransitions(currState, 0);

      % Iterate over possibleTransitions
      for k=1:length(possibleTransitions)
        possibleNextState = possibleTransitions(k);
        
        % Determine what action gets us to possibleNextState
        if currState - possibleNextState == blockSize + 1
          action = 1; % UP_LEFT
        elseif currState - possibleNextState == blockSize - 1
          action = 3; % UP_RIGHT
        else
          action = 2; % UP
          % This captures when currState - bestNextState == blockSize,
          % as well as the last absorbing state
        end
      
        valueOfNextState = stateValues(possibleNextState);
        immediateReward = MDP.getReward(currState, possibleNextState, action);
      
        % Greedy policy to best next state
        if bestNextStateValue < valueOfNextState + immediateReward
          bestNextStateValue = valueOfNextState + immediateReward;
          bestAction = action;
        end
      end

      betterPolicy(j) = bestAction;

    end % i=1:5
  end % j=1:num_states
end % function betterPolicy