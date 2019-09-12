import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hash import *


class simulation:
    def __init__(self, length=12096, mu=0, sigma=0.001117728,
                 b_target=10, block_reward=12.5, hash_ubd=55,
                 hash_slope=3, hash_center=1.5, prev_data=pd.DataFrame(),
                 T_BCH=144, T_BTC=2016, init_price=5400,
                 init_winning_rate=0.00003):
        '''
        Parameters
        ----------
            length: time length of simulation
                length = the number of blocks generated in one simulation.
                A new block is generated in 10 minutes in expection;
                12096 blocks are generated in three months in expectation.

            mu: average of the brownian motion

            sigma: standard deviation of the brownian motion

            b_target: target block time (min) (default: 10 min)
                \bar{B}

            block_reward:
                the amount of cryptocurrency the miner receives when he
                adds a block. (default: 12.5)

            hash_ubd: the upper bound of global hash rate.

            hash_slope, hash_center:
                the parameters that affects the shape of hash supply function

            prev_data:
                a pandas dataframe containing (i) prices, (ii) winning rates,
                (iii) hash rates, and (iv) block times.
                The number of rows should coincides with T_BCH.

            T_BCH: the length of the time window used for DAA of BCH.

            T_BTC: the length of the time window used for DAA of BTC.

            init_price: the initial price.

            init_winning-rate: the inirial winning rate.

        Attributes
        ----------
            block_times
            prices
            winning_rates
            hash_rates
            optimal_winning_rates
            expected_returns

        Notes
        -----
        * As for BTC and BCH, b_target is set to be 10 minutes.

        '''

        # params
        self.mu = mu
        self.sigma = sigma
        self.b_target = b_target
        self.length = length
        self.block_reward = block_reward
        self.hash_ubd = hash_ubd
        self.hash_slope = hash_slope
        self.hash_center = hash_center
        self.T_BCH = T_BCH
        self.T_BTC = T_BTC

        if prev_data.empty == True:
            self.prev_prices = np.ones(T_BCH) * init_price
            self.prev_block_times = np.ones(T_BCH) * b_target
            self.prev_winning_rates = np.ones(T_BCH) * init_winning_rate
        else:
            self.prev_prices = prev_data['prices']
            self.prev_block_times = prev_data['block_times']
            self.prev_winning_rates = prev_data['winning_rates']


    def sim_DAA_1(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                  df_opt_w=pd.DataFrame(),
                  init_height=551443, presim_length=2016, ubd_param=3):
        '''
        Conduct a simulation using DAA-1 as its DAA.
        DAA-1 is based on the DAA used by BTC.

        Parameters
        ----------
            prices : exogenously given. price[t] is the price at time 10*t

            exprvs : exogenously given; used for computing block times.

            opt_w : the oprimal winning rates, computed in advance.

            init_height :
                the height of the block that is created first
                in the simulation. (default: 551443)

            presim_length :
                the length of periods contained in prev_data.
                (Real data used for the pre-simulation period.)
                See also __init__.

            ubd_param :
                determine the maximum number of iterations
                See also _initialization.

        Returns
        -------
            None

        Notes
        -----
            Difficulty, or winning_rate W(t), is adjusted
            every self.T_BTC periods. In reality, BTC lets T_BTC = 2016.
        '''
        if prices.empty == True:
            prices = self.generate_prices()
        if exprvs.empty == True:
            exprvs = self.generate_exprvs()

        # initialization
        ## period 0 to period (presim_length - 1): pre-simulation period
        self._initialization(ubd_param)

        # main loop
        ## See what happens within self.length*self.b_target minutes
        ## default: 12096*10 min = 12 weeks = 3 month
        time_ubd = self.length * self.b_target
        time = 0
        period = presim_length-1

        for t in range(presim_length-1, self.length*ubd_param+presim_length-1):
            # S(t), W(t) is given

            # R(t) = S(t) * M * W(t)
            self.expected_rewards[t] =\
                self.winning_rates[t] * self.block_reward * self.prices[t]

            # W^*(t)
            price_truncated = self.prices[t]
            price_truncated = (price_truncated//50)*50 # grid size = 50
            price_truncated = int(np.max([np.min([price_truncated, 11000]), 100])) # max 11000
            self.optimal_winning_rates[t] =\
                df_opt_w.loc[price_truncated, 'opt_w']

            # hash rate H(t) <- W(t), S(t)
            self.hash_rates[t] = self.hash_supply(t)

            # block time B(t) <- H(t), W(t)
            # multiply 60 to rescale time unit from second to minute
            self.block_times[t] = \
                exprvs[t]/ \
                (self.hash_rates[t] * self.winning_rates[t] * 60)

            time += self.block_times[t]
            period += 1

            if time < time_ubd:
                # S(t+1)
                self.compute_price(current_period=t, current_time=time,
                                   prices=prices)

                # W(t+1)
                if (init_height + t)%self.T_BTC == 0:
                    self.diff_adjust_BTC(current_period=t)
            else:
                break

        self._postprocessing(period)

        return None


    def sim_DAA_2(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                  df_opt_w=pd.DataFrame(),
                  presim_length=2016, ubd_param=3):
        '''
        Conduct a simulation using DAA-2 as its DAA.
        DAA-2 is based on the DAA used by BCH.

        Parameters
        ----------
            prices: see sim_BTC.

            exprvs: see sim_BTC.

            presim_length: see sim_BTC.

            ubd_param: see sim_BTC.

        Returns
        -------
            None

        Notes
        -----
            Difficulty, or winning_rate W(t), is adjusted every period.
            At each adjustment, the last T_BCH blocks are taken into account.
        '''
        if prices.empty == True:
            prices = self.generate_prices()
        if exprvs.empty == True:
            exprvs = self.generate_exprvs()

        # initialization
        ## period 0 to period (presim_length - 1): pre-simulation period
        self._initialization(ubd_param)

        # main loop
        ## See what happens within self.length*self.b_target minutes
        ## default: 12096*10 min = 12 weeks = 3 month
        time_ubd = self.length * self.b_target
        time = 0
        period = presim_length-1

        for t in range(presim_length-1, self.length*ubd_param+presim_length-1):
            # S(t), W(t) is given

            # R(t) = S(t) * M * W(t)
            self.expected_rewards[t] =\
                self.winning_rates[t] * self.block_reward * self.prices[t]


            # W^*(t)
            price_truncated = self.prices[t]
            price_truncated = (price_truncated//50)*50 # grid size = 50
            price_truncated = int(np.max([np.min([price_truncated, 11000]), 100])) # max 11000
            self.optimal_winning_rates[t] =\
                df_opt_w.loc[price_truncated, 'opt_w']

            # hash rate H(t) <- W(t), S(t)
            self.hash_rates[t] = self.hash_supply(t)

            # block time B(t) <- H(t), W(t)
            # multiply 60 to rescale time unit from second to minute
            self.block_times[t] = \
                exprvs[t]/ \
                (self.hash_rates[t] * self.winning_rates[t] * 60)

            time += self.block_times[t]
            period += 1

            if time < time_ubd:
                # S(t+1)
                self.compute_price(current_period=t, current_time=time,
                                   prices=prices)

                # W(t+1)
                ## different from that of BTC in that
                ## difficulty adjustment is conducted every period.
                self.diff_adjust_BCH(current_period=t)
            else:
                break

        self._postprocessing(period)

        return None


    def sim_DAA_0(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                  df_opt_w=pd.DataFrame(),
                  init_height=551443, presim_length=2016, ubd_param=3):
        '''
        Conduct a simulation where the difficulty is always adjusted
        to the optimal level. (imaginary DAA)

        Parameters
        ----------
            prices : exogenously given. price[t] is the price at time 10*t

            exprvs : exogenously given; used for computing block times.

            opt_w :

            init_height :
                the height of the block that is created first
                in the simulation. (default: 551443)

            presim_length :
                the length of periods contained in prev_data.
                (Real data used for the pre-simulation period.)
                See also __init__.

            ubd_param :
                determine the maximum number of iterations
                See also _initialization.

        Returns
        -------
            None

        Notes
        -----
            Difficulty, or winning_rate W(t), is adjusted
            every self.T_BTC periods. In reality, BTC lets T_BTC = 2016.
        '''
        if prices.empty == True:
            prices = self.generate_prices()
        if exprvs.empty == True:
            exprvs = self.generate_exprvs()

        # initialization
        ## period 0 to period (presim_length - 1): pre-simulation period
        self._initialization(ubd_param)

        # main loop
        ## See what happens within self.length*self.b_target minutes
        ## default: 12096*10 min = 12 weeks = 3 month
        time_ubd = self.length * self.b_target
        time = 0
        period = presim_length-1

        for t in range(presim_length-1, self.length*ubd_param+presim_length-1):
            # S(t), W(t) is given

            # W^*(t)
            ## W(t) = W^*(t)
            price_truncated = self.prices[t]
            price_truncated = (price_truncated//50)*50 # grid size = 50
            price_truncated = int(np.max([np.min([price_truncated, 11000]), 100])) # max 11000
            self.optimal_winning_rates[t] =\
                df_opt_w.loc[price_truncated, 'opt_w']
            self.winning_rates[t] = self.optimal_winning_rates[t]

            # R(t) = S(t) * M * W(t)
            self.expected_rewards[t] =\
                self.winning_rates[t] * self.block_reward * self.prices[t]

            # hash rate H(t) <- W(t), S(t)
            self.hash_rates[t] = self.hash_supply(t)

            # block time B(t) <- H(t), W(t)
            # multiply 60 to rescale time unit from second to minute
            self.block_times[t] = \
                exprvs[t]/ \
                (self.hash_rates[t] * self.winning_rates[t] * 60)

            time += self.block_times[t]
            period += 1

            if time < time_ubd:
                # S(t+1)
                self.compute_price(current_period=t, current_time=time,
                                   prices=prices)
            else:
                break

        self._postprocessing(period)

        return None


    def compute_price(self, current_period, current_time, prices):
        '''
        Compute the price at the time when the (t+1)-th block is created:
        compute S(t+1) using price data via linear interpolation.
        prices contains the price date recorded every 10 minutes.
        '''
        time_left = int(current_time//self.b_target)
        time_right = time_left + 1

        self.prices[current_period+1] = \
            prices[time_left] + (prices[time_right] - prices[time_left]) * \
            ((current_time - time_left*self.b_target)/self.b_target)

        return None


    def diff_adjust_BTC(self, current_period):
        '''
        Used by sim_DAA-1.
        Modify self.winning_rates in place.
        '''
        multiplier = \
            (self.block_times[current_period-self.T_BTC+1:\
                current_period+1].sum() / (self.T_BTC * self.b_target))
        self.winning_rates[current_period+1:current_period+self.T_BTC+1] = \
            self.winning_rates[current_period] * multiplier

        return None


    def diff_adjust_BCH(self, current_period):
        '''
        Used by sim_DAA_2.
        Modify self.winning_rates in place.
        '''
        # the term related to B(t)
        block_term = \
            (self.block_times[current_period-self.T_BCH+1: \
                current_period+1].sum() / self.b_target)

        # the term related to W(t)
        temp = np.ones(self.T_BCH)
        w_inverses = temp / (self.winning_rates[current_period-self.T_BCH+1: \
                             current_period+1])
        winning_prob_term = 1 / w_inverses.sum()

        # update W(t)
        self.winning_rates[current_period+1] = \
            block_term * winning_prob_term

        return None


    def hash_supply(self, current_period):
        '''
        Compute hash supply in current period (EH)
        '''
        current_exp_reward = \
            (self.prices[current_period] * self.winning_rates[current_period]
             * self.block_reward)

        return self.hash_ubd * \
            self._sigmoid(self.hash_slope *
                          (current_exp_reward - self.hash_center))


    def _sigmoid(self, x):
        sigmoid_range = 34.538776394910684

        if x <= -sigmoid_range:
            return 1e-15
        if x >= sigmoid_range:
            return 1.0 - 1e-15

        return 1.0 / (1.0 + np.exp(-x))


    def _initialization(self, ubd_param, presim_length=2016):
        # the number of iteration cannot exceeds self.length * self.ubd_param
        sim_length_ubd = self.length * ubd_param

        self.prices = np.zeros((sim_length_ubd,)) # S(t)
        self.winning_rates = np.zeros((sim_length_ubd,)) # W(t)
        self.block_times = np.zeros((sim_length_ubd,)) # B(t)
        self.hash_rates = np.zeros((sim_length_ubd,)) #H(t)
        self.optimal_winning_rates = np.zeros((sim_length_ubd,)) #W^*(t)
        self.expected_rewards = np.zeros((sim_length_ubd,)) #R(t)

        # add pre-simulation periods
        self.prices = np.hstack([self.prev_prices, self.prices])
        self.block_times = \
            np.hstack([self.prev_block_times, self.block_times])
        self.winning_rates = \
            np.hstack([self.prev_winning_rates, self.winning_rates])
        ## for BTC, set the winning rates
        self.winning_rates[presim_length:presim_length+self.T_BTC] = \
            self.winning_rates[presim_length-1]

        ## hash rates in pre-simulation periods will not be used
        ## The same is true of opt_win_rate and exp_returns
        _ = np.zeros(presim_length) + self.hash_supply(presim_length-1) # may be redundant
        self.hash_rates = np.hstack([_, self.hash_rates])
        _ = np.zeros(presim_length)
        self.optimal_winning_rates = np.hstack([_, self.optimal_winning_rates])
        self.expected_rewards = np.hstack([_, self.expected_rewards])

        return None


    def _postprocessing(self, period, presim_length=2016):
        self.block_times = self.block_times[presim_length:period]
        self.prices = self.prices[presim_length:period]
        self.winning_rates = self.winning_rates[presim_length:period]
        self.hash_rates = self.hash_rates[presim_length:period]
        self.optimal_winning_rates =\
            self.optimal_winning_rates[presim_length:period]
        self.expected_rewards = self.expected_rewards[presim_length:period]
        return None


# Functions
def generate_simulation_data(num_iter=3, price_shock=0,
                             opt_w=pd.DataFrame(), prev_data=pd.DataFrame(),
                             dir_sim='/Volumes/Data/research/BDA/simulation/'):
    '''
    Notes
    -----
    num_iter is a number of observations.
    The price data 'sim_prices_ps={}_5000obs.csv'.format(price_shock) should
    be created in advance.
    '''
    df_exprvs = pd.read_csv(dir_sim+'sim_exprvs_5000obs.csv')
    df_price = pd.read_csv(dir_sim+'sim_prices_ps={}_5000obs.csv'\
                           .format(price_shock))
    df_opt_w = pd.read_csv(dir_sim + 'opt_w.csv', index_col=0)

    path = '../data/BTCdata_presim.csv'
    prev_data = pd.read_csv(path)
    prev_data['time'] = pd.to_datetime(prev_data['time'])
    prev_data = prev_data.rename(columns={'blocktime': 'block_times', 'price': 'prices', 'probability of success /Eh': 'winning_rates'})

    df_DAA_1_blocktime = pd.DataFrame()
    df_DAA_1_hashrate = pd.DataFrame()
    df_DAA_1_winrate = pd.DataFrame()
    df_DAA_1_optwinrate = pd.DataFrame()
    df_DAA_1_expreward = pd.DataFrame()

    df_DAA_2_blocktime = pd.DataFrame()
    df_DAA_2_hashrate = pd.DataFrame()
    df_DAA_2_winrate = pd.DataFrame()
    df_DAA_2_optwinrate = pd.DataFrame()
    df_DAA_2_expreward = pd.DataFrame()

    df_DAA_0_blocktime = pd.DataFrame()
    df_DAA_0_hashrate = pd.DataFrame()
    df_DAA_0_winrate = pd.DataFrame()
    df_DAA_0_optwinrate = pd.DataFrame()
    df_DAA_0_expreward = pd.DataFrame()

    sim = simulation(prev_data=prev_data)

    for iter in range(num_iter):
        prices = df_price.loc[:, 'iter_{}'.format(iter)]
        exprvs = df_exprvs.loc[:, 'iter_{}'.format(iter)]

        # DAA-1
        _blocktime = pd.DataFrame()
        _hashrate = pd.DataFrame()
        _winrate = pd.DataFrame()
        _optwinrate = pd.DataFrame()
        _expreward = pd.DataFrame()

        sim.sim_DAA_1(prices=prices, exprvs=exprvs, df_opt_w=df_opt_w)
        _blocktime['iter_{}'.format(iter)] = sim.block_times
        _hashrate['iter_{}'.format(iter)] = sim.hash_rates
        _winrate['iter_{}'.format(iter)] = sim.winning_rates
        _optwinrate['iter_{}'.format(iter)] = sim.optimal_winning_rates
        _expreward['iter_{}'.format(iter)] = sim.expected_rewards

        df_DAA_1_blocktime = pd.concat([df_DAA_1_blocktime, _blocktime], axis=1)
        df_DAA_1_hashrate = pd.concat([df_DAA_1_hashrate, _hashrate], axis=1)
        df_DAA_1_winrate = pd.concat([df_DAA_1_winrate, _winrate], axis=1)
        df_DAA_1_optwinrate = pd.concat([df_DAA_1_optwinrate, _optwinrate], axis=1)
        df_DAA_1_expreward = pd.concat([df_DAA_1_expreward, _expreward], axis=1)

        # DAA-2
        _blocktime = pd.DataFrame()
        _hashrate = pd.DataFrame()
        _winrate = pd.DataFrame()
        _optwinrate = pd.DataFrame()
        _expreward = pd.DataFrame()

        sim.sim_DAA_2(prices=prices, exprvs=exprvs, df_opt_w=df_opt_w)
        _blocktime['iter_{}'.format(iter)] = sim.block_times
        _hashrate['iter_{}'.format(iter)] = sim.hash_rates
        _winrate['iter_{}'.format(iter)] = sim.winning_rates
        _optwinrate['iter_{}'.format(iter)] = sim.optimal_winning_rates
        _expreward['iter_{}'.format(iter)] = sim.expected_rewards

        df_DAA_2_blocktime = pd.concat([df_DAA_2_blocktime, _blocktime], axis=1)
        df_DAA_2_hashrate = pd.concat([df_DAA_2_hashrate, _hashrate], axis=1)
        df_DAA_2_winrate = pd.concat([df_DAA_2_winrate, _winrate], axis=1)
        df_DAA_2_optwinrate = pd.concat([df_DAA_2_optwinrate, _optwinrate], axis=1)
        df_DAA_2_expreward = pd.concat([df_DAA_2_expreward, _expreward], axis=1)

        # DAA-0
        _blocktime = pd.DataFrame()
        _hashrate = pd.DataFrame()
        _winrate = pd.DataFrame()
        _optwinrate = pd.DataFrame()
        _expreward = pd.DataFrame()

        sim.sim_DAA_0(prices=prices, exprvs=exprvs, df_opt_w=df_opt_w)
        _blocktime['iter_{}'.format(iter)] = sim.block_times
        _hashrate['iter_{}'.format(iter)] = sim.hash_rates
        _winrate['iter_{}'.format(iter)] = sim.winning_rates
        _optwinrate['iter_{}'.format(iter)] = sim.optimal_winning_rates
        _expreward['iter_{}'.format(iter)] = sim.expected_rewards

        df_DAA_0_blocktime = pd.concat([df_DAA_0_blocktime, _blocktime], axis=1)
        df_DAA_0_hashrate = pd.concat([df_DAA_0_hashrate, _hashrate], axis=1)
        df_DAA_0_winrate = pd.concat([df_DAA_0_winrate, _winrate], axis=1)
        df_DAA_0_optwinrate = pd.concat([df_DAA_0_optwinrate, _optwinrate], axis=1)
        df_DAA_0_expreward = pd.concat([df_DAA_0_expreward, _expreward], axis=1)

    df_DAA_1_blocktime.to_csv(dir_sim+'DAA-1_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_1_hashrate.to_csv(dir_sim+'DAA-1_hashrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_1_winrate.to_csv(dir_sim+'DAA-1_winrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_1_optwinrate.to_csv(dir_sim+'DAA-1_optwinrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_1_expreward.to_csv(dir_sim+'DAA-1_expreward_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')

    df_DAA_2_blocktime.to_csv(dir_sim+'DAA-2_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_2_hashrate.to_csv(dir_sim+'DAA-2_hashrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_2_winrate.to_csv(dir_sim+'DAA-2_winrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_2_optwinrate.to_csv(dir_sim+'DAA-2_optwinrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_2_expreward.to_csv(dir_sim+'DAA-2_expreward_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')

    df_DAA_0_blocktime.to_csv(dir_sim+'DAA-0_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_hashrate.to_csv(dir_sim+'DAA-0_hashrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_winrate.to_csv(dir_sim+'DAA-0_winrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_optwinrate.to_csv(dir_sim+'DAA-0_optwinrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_expreward.to_csv(dir_sim+'DAA-0_expreward_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')

    return None


def generate_simulation_data_DAA0(num_iter=3, price_shock=0,
                             opt_w=pd.DataFrame(), prev_data=pd.DataFrame(),
                             dir_sim='/Volumes/Data/research/BDA/simulation/'):
    '''
    Notes
    -----
    num_iter is a number of observations.
    The price data 'sim_prices_ps={}_5000obs.csv'.format(price_shock) should
    be created in advance.
    '''
    df_exprvs = pd.read_csv(dir_sim+'sim_exprvs_5000obs.csv')
    df_price = pd.read_csv(dir_sim+'sim_prices_ps={}_5000obs.csv'\
                           .format(price_shock))
    df_opt_w = pd.read_csv(dir_sim + 'opt_w.csv', index_col=0)

    path = '../data/BTCdata_presim.csv'
    prev_data = pd.read_csv(path)
    prev_data['time'] = pd.to_datetime(prev_data['time'])
    prev_data = prev_data.rename(columns={'blocktime': 'block_times', 'price': 'prices', 'probability of success /Eh': 'winning_rates'})

    df_DAA_0_blocktime = pd.DataFrame()
    df_DAA_0_hashrate = pd.DataFrame()
    df_DAA_0_winrate = pd.DataFrame()
    df_DAA_0_optwinrate = pd.DataFrame()
    df_DAA_0_expreward = pd.DataFrame()

    sim = simulation(prev_data=prev_data)

    for iter in range(num_iter):
        prices = df_price.loc[:, 'iter_{}'.format(iter)]
        exprvs = df_exprvs.loc[:, 'iter_{}'.format(iter)]

        # DAA-0
        _blocktime = pd.DataFrame()
        _hashrate = pd.DataFrame()
        _winrate = pd.DataFrame()
        _optwinrate = pd.DataFrame()
        _expreward = pd.DataFrame()

        sim.sim_DAA_0(prices=prices, exprvs=exprvs, df_opt_w=df_opt_w)
        _blocktime['iter_{}'.format(iter)] = sim.block_times
        _hashrate['iter_{}'.format(iter)] = sim.hash_rates
        _winrate['iter_{}'.format(iter)] = sim.winning_rates
        _optwinrate['iter_{}'.format(iter)] = sim.optimal_winning_rates
        _expreward['iter_{}'.format(iter)] = sim.expected_rewards

        df_DAA_0_blocktime = pd.concat([df_DAA_0_blocktime, _blocktime], axis=1)
        df_DAA_0_hashrate = pd.concat([df_DAA_0_hashrate, _hashrate], axis=1)
        df_DAA_0_winrate = pd.concat([df_DAA_0_winrate, _winrate], axis=1)
        df_DAA_0_optwinrate = pd.concat([df_DAA_0_optwinrate, _optwinrate], axis=1)
        df_DAA_0_expreward = pd.concat([df_DAA_0_expreward, _expreward], axis=1)

    df_DAA_0_blocktime.to_csv(dir_sim+'DAA-0_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_hashrate.to_csv(dir_sim+'DAA-0_hashrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_winrate.to_csv(dir_sim+'DAA-0_winrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_optwinrate.to_csv(dir_sim+'DAA-0_optwinrate_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_0_expreward.to_csv(dir_sim+'DAA-0_expreward_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')

    return None
