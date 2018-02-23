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

    % Number of states
    for j=1:num_states
      currState = j;
      bestNextState = -Inf;
      bestNextStateValue = -Inf;
      bestAction = -Inf;

      % Get possible transitions from currState
      [ possibleTransitions, probabilityForEachTransition ] = ...
        MDP.getTransitions(currState, 0);

      % Iterate over possibleTransitions
      for k=1:length(possibleTransitions)
        possibleNextState = possibleTransitions(k);
        % Greedy policy to best next state
        if bestNextStateValue < stateValues(possibleNextState)
          bestNextState = possibleNextState;
          bestNextStateValue = stateValues(possibleNextState);
        end
      end

      % Now we know the best next state
      % Determine what action (UP, UP_LEFT or UP_RIGHT) gets us there
      if currState - bestNextState == blockSize + 1
        bestAction = 1; % UP_LEFT
      elseif currState - bestNextState == blockSize - 1
        bestAction = 3; % UP_RIGHT
      else
        bestAction = 2; % UP
        % This captures when currState - bestNextState == blockSize,
        % as well as the last absorbing state
      end

      betterPolicy(j) = bestAction;

    end % i=1:5
  end % j=1:num_states
end % function betterPolicy