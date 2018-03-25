ALGORITHM = 1; % 0 for Monte Carlo, 1 for TD-Learning

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

probabilityOfUniformlyRandomDirectionTaken = 0.15 ; % Noisy driver actions.
% An action will not always have the desired effect. This is the
% probability that the selected action is ignored and the car uniformly 
% transitions into one of the above 3 states. If one of those states would 
% be outside the map, the next state will be the one above the current one.

roadBasisGridMaps = generateMiniMaps ; % Generates the 8 road basis grid 
% maps, complete with an initial location for your agent. (Also see the 
% GridMap class).

noCarOnRowProbability = 0.8 ; % the probability that there is no car 
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


%% TEST ACTION TAKING, MOVING WINDOW AND TRAJECTORY PRINTING:
% Simulating agent behaviour when following the policy defined by 
% $pi_test1$.
%
% Commented lines also have examples of use for $GridMap$'s $getReward$ and
% $getTransitions$ functions, which act as our reward and transition
% functions respectively.


%% Student code: BEGIN
ALPHA 			= 0.005;
NUM_EPISODES 	= 1000;
theta 			= ones(20,3); % Weight vector, size: (num_features, num_actions)
episodeFeatures = zeros(24, 20);
episodeRewards 	= zeros(24,1);
episodeActions 	= zeros(24,1);
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
		
		% Use the $getStateFeatures$ function as below, in order to get the
		% feature description of a state:
		stateFeatures = MDP.getStateFeatures(realAgentLocation); % Dimensions, 4 rows x 5 columns
		
		for action = 1:3
			action_values(action) = sum(sum(Q_test1(:,:,action) .* stateFeatures));
		end % for each possible action

		%% Student code: BEGIN | Break ties between optimal actions
		set_a_max = find(action_values == max(action_values)); % Set of optimal actions, if more than one
		idx = randsample(length(set_a_max), 1); % Randomly pick one of the max actions
		actionTaken = set_a_max(idx);
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

	for i = 1:episodeLength - 1
		a = episodeActions(i+1); 	% Action taken @i
		phi = episodeFeatures(i,:); % Feature representation @i
		Q = phi * theta(:, a); 		% Estimated Action-Value @i
		if ALGORITHM == 0 % Monte Carlo
			G = sum(episodeRewards(i:end)); 	% Actual reward accumulated till end of episode
			grad = (ALPHA * (G - Q)) .* phi'; 	% Gradient
		elseif ALGORITHM == 1 % TD(0)-Learning
			R_next = episodeRewards(i + 1); 	% Actual reward @i+1
			Q_next = episodeFeatures(i + 1,:) * theta(:, episodeActions(i + 1)); % Estimated Q @i+1
			grad = ALPHA * (R_next + Q_next - Q) .* phi'; % Gradient
		end % if
		theta(:,a) = theta(:,a) + grad;
	end % episodeLength

	ALPHA = (1/sqrt(episode)) * ALPHA; % decreasing alpha
	%% Student code: END
	
	currentMap = MDP;
	agentLocation = realAgentLocation;
	
	% Return;
	% printAgentTrajectory;
end % NUM_EPISODES

theta