%% STUDENT SUPPORT FUNCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% POLICY EVALUATION (Exercise 1) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% A function for iterative value update
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function stateValues = policyEvaluation( ...
  blockSize, episodeLength, MDP, policy, num_value_iterations)

  % Initialize parameters %
  num_states = length(policy); % Number of states in the MDP
  stateValues = zeros(1, num_states); % Initialize to 0 the values of all 
  % states. We will iteratively converge onto the true values under the policy

  % Run iterative updates %
  for i=1:num_value_iterations
    % Temporary store for values of current iteration
    stateValues_temp = zeros(1, num_states);
    for j=1:num_states
      % Coordinates of the current state
      currState = j;
      actionTaken = policy(j);
      % Get possible transitions from currState
      [ possibleTransitions, probabilityForEachTransition ] = ...
        MDP.getTransitions(currState, actionTaken);
      % Iterate over possibleTransitions
      for k=1:length(possibleTransitions)
        nextState = possibleTransitions(k);
        probForTransition = probabilityForEachTransition(k);

        immediateReward = MDP.getReward(currState, nextState, actionTaken);
        valueOfNextState = stateValues(nextState);
        
        % Increment the value of the current state
        stateValues_temp(j) = stateValues_temp(j) + ...
          probForTransition * (immediateReward + valueOfNextState);

      end % k=1:size(possibleTransitions,1)
    end % j=1:num_states
    stateValues = stateValues_temp;
  end % i=1:num_value_iterations
end % function stateValues