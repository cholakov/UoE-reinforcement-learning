%% Student code: BEGIN | Settings
%
ALGORITHM 		= 0; % 0 for Monte Carlo, 1 for TD-Learning
FEATURES 		= 0; % 0 for Original, 	  1 for Custom

EVAL 			= false; % Activate Evaluation mode. 
EVAL_MODE 		= 1; % use only when EVAL = true
					 % 0 evaluate policy implied by Q_test1, 
					 % 1 evaluate policy implied by trained weights

% Notes on how to use the settings above:
% Run program at least once with EVAL = false to learn theta, else the
% program does not have trained weights.

% Note that if you train on custom features, FEATURES = 1, then you cannot 
% evaluate Q_test1 with EVAL_MODE = 0, because Q_test1 has a set of 4x5 features,
% while the custom features are only 3.


NUM_EPISODES 	= 2000;
DECR_FACTOR 	= 0.9; % Decreasing factor for Alpha

%
%% Student code: END


%% ACTION CONSTANTS:
UP_LEFT = 1 ;
UP = 2 ;
UP_RIGHT = 3 ;


%% PROBLEM SPECIFICATION:

blockSize = 5 ; % This will function as the dimension of the road basis 
% images (blockSize x blockSize), as well as the view range, in rows of
% your car (including the current row).

n_MiniMapBlocksPerMap = 5 ; % determines the size of the test instance. 
% Test instances are essentially road bases stacked one on top of the
% other.

basisEpsisodeLength = blockSize - 1 ; % The agent moves forward at constant speed and
% the upper row of the map functions as a set of terminal states. So 5 rows
% -> 4 actions.

episodeLength = blockSize*n_MiniMapBlocksPerMap - 1 ;% Similarly for a complete
% scenario created from joining road basis grid maps in a line.

%discountFactor_gamma = 1 ; % if needed

rewards = [ 1, -1, -20 ] ; % the rewards are state-based. In order: paved 
% square, non-paved square, and car collision. Agents can occupy the same
% square as another car, and the collision does not end the instance, but
% there is a significant reward penalty.

probabilityOfUniformlyRandomDirectionTaken = 0.0 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly 
% transitions into one of the above 3 states. If one of those states would 
% be outside the map, the next state will be the one above the current one.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid 
% maps, complete with an initial location for your agent. (Also see the 
% GridMap class).

noCarOnRowProbability = 0.8; % the probability that there is no car 
% spawned for each row

seed = 1234;
rng(seed); % setting the seed for the random nunber generator

% Call this whenever starting a new episode:
MDP = generateMap(roadBasisGridMaps, n_MiniMapBlocksPerMap, blockSize, ...
	noCarOnRowProbability, probabilityOfUniformlyRandomDirectionTaken, ...
	rewards);


%% Initialising the state observation (state features) and setting up the 
% exercise approximate Q-function:
stateFeatures = ones(4, 5);
action_values = zeros(1, 3);

Q_test1 = ones(4, 5, 3);
Q_test1(:,:,1) = 100;
Q_test1(:,:,3) = 100; % obviously this is not a correctly computed Q-function; 
% it does imply a policy however: Always go Up! (though on a clear road it will 
% default to the first indexed action: go left)


%% Student code: BEGIN | Hyperparameters 
%

if EVAL
	avgReturn = 0; % average return over NUM_EPISODES 
else
	if FEATURES == 0
		num_features = 20;
	elseif FEATURES == 1
		num_features = 3;
	end

	EPSILON 		= 0.2;
	ALPHA 			= 0.005;
	theta 			= ones(num_features,3); % Weight vector, size: (num_features, num_actions)
	episodeFeatures = zeros(24, num_features);
	episodeRewards 	= zeros(24,1);
	episodeActions 	= zeros(24,1);
end
%
%% Student code: END

for episode = 1:NUM_EPISODES

	currentTimeStep = 0 ;
	MDP = generateMap( roadBasisGridMaps, n_MiniMapBlocksPerMap, ...
		blockSize, noCarOnRowProbability, ...
		probabilityOfUniformlyRandomDirectionTaken, rewards );
	currentMap = MDP ;
	agentLocation = currentMap.Start ;
	startingLocation = agentLocation ; % Keeping record of initial location.
	
	% If you need to keep track of agent movement history:
	agentMovementHistory = zeros(episodeLength+1, 2) ;
	agentMovementHistory(currentTimeStep + 1, :) = agentLocation ;
		
	realAgentLocation = agentLocation ; % The location on the full test map.
	Return = 0;
	
	for i = 1:episodeLength
		
		stateFeatures = MDP.getStateFeatures(realAgentLocation); % Dimensions, 4 rows x 5 columns

		if FEATURES == 1 % Transform features
			% new_all = sum(stateFeatures); % vector of new features, summed vertically

			if realAgentLocation(2) == 1
				new_left = -5;
			else
				% new_left = sum(new_all(1:realAgentLocation(2)-1));
				new_left = stateFeatures(4,realAgentLocation(2)-1);
			end
			if realAgentLocation(2) == 5
				new_right = -5;
			else
				% new_right = sum(new_all(realAgentLocation(2)+1: 5));
				new_right = stateFeatures(4,realAgentLocation(2)+1);
			end

			% new_up = new_all(realAgentLocation(2));
			new_up = stateFeatures(4,realAgentLocation(2));
			
			
			stateFeatures = [new_left, new_up, new_right];
		end
		
		%% Student code: BEGIN | Implement Epsilon-Greedy Exploration

		if EVAL % Are we in evaluation mode? If yes, go greedy. Else, explore a bit.
			
			if EVAL_MODE == 0 % Use policy implied by Q_test1
				for action = 1:3
            		action_values(action) = sum (sum( Q_test1(:,:,action) .* stateFeatures ));
       			end % for each possible action
				[~, actionTaken] = max(action_values);

			else % Use policy implied by trained weights
				for action = 1:3
					action_values(action) = sum(theta(:, action) .* stateFeatures(:));
				end % for each possible action
				[~, actionTaken] = max(action_values);
			end

		else
			for action = 1:3
				action_values(action) = sum(theta(:, action) .* stateFeatures(:));
			end % for each possible action
			set_a_max = find(action_values == max(action_values)); % Set of optimal actions, if more than one
			set_a_other = setdiff([1 2 3], set_a_max); % Set of all other actions
			if rand < (EPSILON / 3 + 1 - EPSILON) | length(set_a_other) == 0 % Exploration or exploitation?
				idx = randsample(length(set_a_max), 1); % Randomly pick one of the max actions 
				actionTaken = set_a_max(idx);
			else	
				idx = randsample(length(set_a_other), 1); % Randomly pick one of the other actions 
				actionTaken = set_a_other(idx);
			end
		end
		%% Student code: END
			   		
		[ agentRewardSignal, realAgentLocation, currentTimeStep, ...
			agentMovementHistory ] = ...
			actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
			currentTimeStep, agentMovementHistory, ...
			probabilityOfUniformlyRandomDirectionTaken ) ;

		%% Student code: BEGIN | Record episode variables
		episodeFeatures(i,:) = stateFeatures(:);
		episodeRewards(i) = agentRewardSignal;
		episodeActions(i) = actionTaken;
		Return = Return + agentRewardSignal;
		%% Student code: END
		
	end % episodeLength

	%% Student code: BEGIN | Update Gradient

	% theta (20,3)
	% episodeFeatures (24, 20)
	% episodeRewards (24,1)
	% episodeActions (24,1)

	if EVAL
		% printAgentTrajectory;
		avgReturn = [avgReturn, Return];
	else
		for i = 1:episodeLength - 1
			a = episodeActions(i); 		% Action taken @i
			phi = episodeFeatures(i,:); % Feature representation @i
			Q = phi * theta(:, a); 		% Estimated Action-Value @i
			if ALGORITHM == 0 % Monte Carlo
				% G = sum(episodeRewards(i:end)); 	% Actual reward accumulated till end of episode
				G = episodeRewards(i); 			% Actual reward accumulated for one step ???> better results
				grad = (ALPHA * (G - Q)) .* phi'; 	% Gradient
			elseif ALGORITHM == 1 % TD(0)-Learning
				R_next = episodeRewards(i + 1); 	% Actual reward @i+1
				Q_next = episodeFeatures(i + 1,:) * theta(:, episodeActions(i + 1)); % Estimated Q @i+1
				grad = ALPHA * (R_next + Q_next - Q) .* phi'; % Gradient
			end
			theta(:,a) = theta(:,a) + grad;
		end % episodeLength

		ALPHA = DECR_FACTOR * ALPHA; % decreasing alpha
	end
	%% Student code: END
	
	currentMap = MDP;
	agentLocation = realAgentLocation;
	
	
end % NUM_EPISODES


if EVAL
	mean(avgReturn)
else
	% printAgentTrajectory;
	% theta
end
    