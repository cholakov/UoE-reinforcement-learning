
ALGORITHM = 0 % 0 for Monte Carlo, 1 for TD-Learning

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



num_episodes = 5
total_returns = containers.Map;

EPSILON = 0.3
GAMMA = 0.8;
ALPHA = 0.9;
ANNEALING = 1.5; % decreasing factor for ALPHA


for episode = 1:num_episodes

	%% Student code: BEGIN
	visited_in_episode = containers.Map;
	%% Student code: END


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
		stateFeatures = MDP.getStateFeatures(realAgentLocation); % dimensions are 4 rows x 5 columns
		
		for action = 1:3
			action_values(action) = ...
				sum ( sum( Q_test1(:,:,action) .* stateFeatures ) );
		end % for each possible action

		%% Student code: BEGIN
		% Implement Epsilon-Greedy exploration on all actions
		set_a_max = find(action_values == max(action_values));  % indices of actions with max value
		set_a_other = setdiff([1 2 3], set_a_max); % indices of all other actions

		if rand > EPSILON/3+1-EPSILON | length(set_a_other) == 0 % epsilon: exploration or exploitation?
			actionTaken = randsample(length(set_a_max), 1); % randomly pick one of the max actions 
			% if more than one in set_a_max
		else	
			actionTaken = randsample(length(set_a_other), 1); % randomly pick one of the max actions 
			% if more than one in set_a_other
		end
		%% Student code: END
			   		
		[ agentRewardSignal, realAgentLocation, currentTimeStep, ...
			agentMovementHistory ] = ...
			actionMoveAgent( actionTaken, realAgentLocation, MDP, ...
			currentTimeStep, agentMovementHistory, ...
			probabilityOfUniformlyRandomDirectionTaken ) ;
		

		%% Student code: BEGIN
		% Implements Every-Visit Monte Carlo
		% Increments visited_in_episode
		%
		%%

		stateFeatures_str = sprintf('%s', stateFeatures);
		% stateFeatures_str = sprintf('%s', i);
		% Increment all previously visited states with the reward from the
		% current state 
		k = keys(visited_in_episode);
		for j = 1:length(visited_in_episode)
			d = visited_in_episode(k{j});
			r = d(1) + agentRewardSignal;
			visited_in_episode(k{j}) = [r d(2)];
 		end
 		% If a state not visited in the current episode (determined by state's 
 		% representation), initialize entry
		if ~isKey(visited_in_episode, stateFeatures_str) 
			visited_in_episode(stateFeatures_str) = [0 actionTaken]; % [reward action]
		end		
		%% Student code: END

		Return = Return + agentRewardSignal;
		
	end % for state in episode


	%% Student code: BEGIN
	% Increments total_returns
	%
	%%

	k = keys(visited_in_episode);
	for j = 1:length(visited_in_episode)
		s = k{j}; % state identified by its feature space
		last = visited_in_episode(s);
		r = last(1); % reward accumulated (in last episode)
		a = last(2); % action taken (in last episode)
		if ~isKey(total_returns, s) % initialize
			new = [0 0; 0 0; 0 0]; % 3 states, each characterized
			% by its total return and counter over many episodes
			total_returns(s) = new;
		end
		% increment
		incr = total_returns(s);
		incr(a, 1) = incr(a, 1) + r;
		incr(a, 2) = incr(a, 2) + 1;
		total_returns(s) = incr;
	end
	%% Student code: END
	
	currentMap = MDP;
	agentLocation = realAgentLocation;
	
	Return;
	printAgentTrajectory;
	
end % for each episode