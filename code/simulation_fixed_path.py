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


    def sim_BTC(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                init_height=551443, presim_length=2016, ubd_param=3):
        '''
        Conduct a simulation using DAA-1 as its DAA.

        Parameters
        ----------
            prices: exogenously given. price[t] is the price at time 10*t

            exprvs: exogenously given; used for computing block times.

            init_height:
                the height of the block that is created first
                in the simulation. (default: 551443)

            presim_length:
                the length of periods contained in prev_data.
                (Real data used for the pre-simulation period.)
                See also __init__.

            ubd_param:
                determine the maximum number of iterations
                See also _initialization.

        Returns
        -------
            self.block_times:
                the realized block generation times B(t).

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

        return self.block_times


    def sim_BTC_bdd(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                    init_height=551443, presim_length=2016, ubd_param=3):
        '''
        Parameters
        ----------
            prices: see sim_BTC.

            exprvs: see sim_BTC.

            init_height: see sim_BTC.

            presim_length: see sim_BTC.

            ubd_param: see sim_BTC.

        Returns
        -------
            self.block_times: see sim_BTC.
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
                    self.diff_adjust_BTC_bdd(current_period=t)
            else:
                break

        self._postprocessing(period)

        return self.block_times


    def sim_BCH(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                presim_length=2016, ubd_param=3):
        '''
        Parameters
        ----------
            prices: see sim_BTC.

            exprvs: see sim_BTC.

            presim_length: see sim_BTC.

            ubd_param: see sim_BTC.

        Returns
        -------
            self.block_times: see sim_BTC.

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

        return self.block_times


    def sim_BCH_bdd(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                    presim_length=2016, ubd_param=3):
        '''
        Parameters
        ----------
            prices: see sim_BTC.

            exprvs: see sim_BTC.

            presim_length: see sim_BTC.

            ubd_param: see sim_BTC.

        Returns
        -------
            self.block_times: see sim_BTC.
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
                self.diff_adjust_BCH_bdd(current_period=t)
            else:
                break

        self._postprocessing(period)

        return self.block_times


    def sim_pseudoBCH(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                      presim_length=2016, ubd_param=3):
        '''
        Parameters
        ----------
            prices: see sim_BTC.

            exprvs: see sim_BTC.

            presim_length: see sim_BTC.

            ubd_param: see sim_BTC.

        Returns
        -------
            self.block_times: see sim_BTC.
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
                self.diff_adjust_pseudoBCH(current_period=t)
            else:
                break

        self._postprocessing(period)

        return self.block_times


    def sim_DAA_4(self, prices=pd.DataFrame(), exprvs=pd.DataFrame(),
                  presim_length=2016, ubd_param=3):
        '''
        Parameters
        ----------
            prices: see sim_BTC.

            exprvs: see sim_BTC.

            presim_length: see sim_BTC.

            ubd_param: see sim_BTC.

        Returns
        -------
            self.block_times: see sim_BTC.

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

                self.diff_adjust_DAA_4(current_period=t)
            else:
                break

        self._postprocessing(period)

        return self.block_times


    def generate_prices(self, init_price=5400, grid=10, ubd_param=3, \
                        seed=None):
        '''
        Generate a price path that follows the price motion.
        Each price is observed every grid(default: 10) minutes.
        '''
        if seed != None:
            np.random.seed(seed)

        # initialization
        self.prices = np.zeros(self.length*ubd_param)
        self.prices[0] = init_price

        # fix a path of epsilons
        epsilons = np.random.normal(size=self.length*ubd_param)

        # generate prices according to epsilons
        for period in range(self.length*ubd_param-1):
            drift = (self.mu * self.prices[period] * grid)
            disturbance = (self.sigma * np.sqrt(grid)
                           * epsilons[period] * self.prices[period])
            self.prices[period + 1] = (self.prices[period]
                                           + drift + disturbance)

        return self.prices


    def generate_exprvs(self, ubd_param=3, seed=None):
        '''
        Generate a series of random variables that follows Exp(1).
        The series are used for calculate block times.
        '''
        if seed != None:
            np.random.seed(seed)

        exprvs = np.random.exponential(scale=1.0, \
                                            size=self.length*ubd_param)
        return exprvs


    def price_shock(self, price_shock):
        self.prices[self.T_BCH-1] = self.prices[self.T_BCH-2] - price_shock
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
        Used by sim_BTC.
        Modify self.winning_rates in place.
        '''
        multiplier = \
            (self.block_times[current_period-self.T_BTC+1:\
                current_period+1].sum() / (self.T_BTC * self.b_target))
        self.winning_rates[current_period+1:current_period+self.T_BTC+1] = \
            self.winning_rates[current_period] * multiplier

        return None


    def diff_adjust_BTC_bdd(self, current_period):
        '''
        Used by sim_BTC_bounded.
        Modify self.winning_rates in place.
        '''
        multiplier = \
            (self.block_times[current_period-self.T_BTC+1: \
                current_period+1].sum() / (self.T_BTC * self.b_target))

        multiplier = np.min((np.max((0.25, multiplier)), 4))

        self.winning_rates[current_period+1:current_period+self.T_BTC+1] = \
            self.winning_rates[current_period] * multiplier

        return None


    def diff_adjust_pseudoBCH(self, current_period):
        '''
        Used by sim_pesudoBCH.
        Modify self.winning_rates in place.
        '''
        multiplier = \
            (self.block_times[current_period-self.T_BCH+1: \
                current_period+1].sum() / (self.T_BCH * self.b_target))
        diff_avg = \
            self.winning_rates[current_period-self.T_BCH+1: \
                                 current_period+1].sum() / self.T_BCH
        self.winning_rates[current_period+1] = diff_avg * multiplier

        return None


    def diff_adjust_DAA_4(self, current_period):
        '''
        Used by sim_DAA_4.
        Modify self.winning_rates in place.
        '''
        weights = np.arange(self.T_BCH)+1.0
        terms = weights * self.block_times[current_period-self.T_BCH+1: \
                current_period+1] * \
                self.winning_rates[current_period-self.T_BCH+1: \
                                   current_period+1]

        self.winning_rates[current_period+1] = \
            (2.0 * terms.sum()) / (self.T_BCH * (self.T_BCH+1) * self.b_target)

        return None


    def diff_adjust_BCH(self, current_period):
        '''
        Used by sim_BCH.
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


    def diff_adjust_BCH_bdd(self, current_period):
        '''
        Used by sim_BCH_bdd.
        Modify self.winning_rates in place.
        '''
        # the term related to B(t)
        block_term = \
            (self.block_times[current_period-self.T_BCH+1: \
                current_period+1].sum() / self.b_target)
        ## bound
        block_term = np.max((np.min((block_term, 288)), 72))

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
        _ = np.zeros(presim_length) + self.hash_supply(presim_length-1)
        self.hash_rates = np.hstack([_, self.hash_rates])

        return None


    def _postprocessing(self, period, presim_length=2016):
        self.block_times = self.block_times[presim_length:period]
        self.prices = self.prices[presim_length:period]
        self.winning_rates = self.winning_rates[presim_length:period]
        self.hash_rates = self.hash_rates[presim_length:period]

        return None


# Functions
## For data generation.
def generate_price_paths(price_shock=0, num_iter=1000, filename='',
                         dir='/Volumes/Data/research/BDA/simulation/',
                         df_presim=pd.DataFrame(), T_BCH=144):
    '''
    Generate price paths. Save them as a csv file.
    '''
    if df_presim.empty == True:
        init_price = 3604.59 # height 551442
    else:
        init_price = df_presim.loc[T_BCH-1, 'prices']
    sim = simulation()

    df_price = pd.DataFrame()
    for iter in range(num_iter):
        df_price['iter_{}'.format(iter)] = \
            sim.generate_prices(init_price=init_price - price_shock)
    df_price.to_csv(dir + 'sim_prices_ps={}'\
        .format(price_shock) + filename + '.csv', index=False)

    return None


def generate_exprvs(num_iter=1000, filename='',
                    dir='/Volumes/Data/research/BDA/simulation/'):
    '''
    Generate random variables following Exp(1). Save them as a csv file.
    '''
    sim = simulation()

    df_exprvs = pd.DataFrame()
    for iter in range(num_iter):
        df_exprvs['iter_{}'.format(iter)] = \
            sim.generate_exprvs()
    df_exprvs.to_csv(dir + 'sim_exprvs' + filename + '.csv', index=False)

    return None


def generate_simulation_data(num_iter=1000, price_shock=0,
                             opt_w=pd.DataFrame(), prev_data=pd.DataFrame(),
                             dir_sim='/Volumes/Data/research/BDA/simulation/'):
    '''
    num_iter is a number of observations.
    Record (i) the realized winning rates and (ii) the optimal winning rates.
    The price data 'sim_prices_ps={}_5000obs.csv'.format(price_shock) should
    be created in advance.

    Notes
    -----
    This program requires much time and memory; the code may need to be
    rewritten in a more efficient manner.
    '''
    sim = simulation(prev_data=prev_data)
    df_exprvs = pd.read_csv(dir_sim+'sim_exprvs_5000obs.csv')
    df_price = pd.read_csv(dir_sim+'sim_prices_ps={}_5000obs.csv'\
                           .format(price_shock))

    df_BTC = pd.DataFrame()
    df_BCH = pd.DataFrame()
    df_BTC_bdd = pd.DataFrame()
    df_BCH_bdd = pd.DataFrame()
    df_pseudoBCH = pd.DataFrame()
    df_DAA_4 = pd.DataFrame()

    df_BTC_opt = pd.DataFrame()
    df_BCH_opt = pd.DataFrame()
    df_BTC_bdd_opt = pd.DataFrame()
    df_BCH_bdd_opt = pd.DataFrame()
    df_pseudoBCH_opt = pd.DataFrame()
    df_DAA_4_opt = pd.DataFrame()

    for iter in range(num_iter):
        prices = df_price.loc[:, 'iter_{}'.format(iter)]
        exprvs = df_exprvs.loc[:, 'iter_{}'.format(iter)]

        _ = pd.DataFrame()
        _opt = pd.DataFrame()
        w = sim.sim_BTC(prices=prices, exprvs=exprvs)
        _['iter_{}'.format(iter)] = w
        # w^*を計算するステップをどこかで入れる
        # minerの利益も
        df_BTC = pd.concat([df_BTC, _], axis=1)
        df_BTC_opt = pd.concat([df_BTC_opt, _opt], axis=1)

        _ = pd.DataFrame()
        _opt = pd.DataFrame()
        w = sim.sim_BCH(prices=prices, exprvs=exprvs)
        _['iter_{}'.format(iter)] = w
        # w^*を計算するステップをどこかで入れる
        # minerの利益も
        df_BCH = pd.concat([df_BCH, _], axis=1)
        df_BCH_opt = pd.concat([df_BCH_opt, _opt], axis=1)

        _ = pd.DataFrame()
        _opt = pd.DataFrame()
        w = sim.sim_BTC_bdd(prices=prices, exprvs=exprvs)
        _['iter_{}'.format(iter)] = w
        # w^*を計算するステップをどこかで入れる
        # minerの利益も
        df_BTC_bdd = pd.concat([df_BTC_bdd, _], axis=1)
        df_BTC_bdd_opt = pd.concat([df_BTC_bdd_opt, _opt], axis=1)

        _ = pd.DataFrame()
        _opt = pd.DataFrame()
        w = sim.sim_BCH_bdd(prices=prices, exprvs=exprvs)
        _['iter_{}'.format(iter)] = w
        # w^*を計算するステップをどこかで入れる
        # minerの利益も
        df_BCH_bdd = pd.concat([df_BCH_bdd, _], axis=1)
        df_BCH_bdd_opt = pd.concat([df_BCH_bdd_opt, _opt], axis=1)

        _ = pd.DataFrame()
        _opt = pd.DataFrame()
        w = sim.sim_pseudoBCH(prices=prices, exprvs=exprvs)
        _['iter_{}'.format(iter)] = w
        # w^*を計算するステップをどこかで入れる
        # minerの利益も
        df_pseudoBCH = pd.concat([df_pseudoBCH, _], axis=1)
        df_pseudoBCH_opt = pd.concat([df_pseudoBCH_opt, _opt], axis=1)

        _ = pd.DataFrame()
        _opt = pd.DataFrame()
        w = sim.sim_DAA_4(prices=prices, exprvs=exprvs)
        _['iter_{}'.format(iter)] = w
        # w^*を計算するステップをどこかで入れる
        # minerの利益も
        df_DAA_4 = pd.concat([df_DAA_4, _], axis=1)
        df_DAA_4_opt = pd.concat([df_DAA_4_opt, _opt], axis=1)

    df_BTC.to_csv(dir_sim+'BTC_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_BCH.to_csv(dir_sim+'BCH_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_BTC_bdd.to_csv(dir_sim+'BTC_bdd_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_BCH_bdd.to_csv(dir_sim+'BCH_bdd_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_pseudoBCH.to_csv(dir_sim+'pseudoBCH_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_4.to_csv(dir_sim+'DAA_4_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')

    df_BTC_opt.to_csv(dir_sim+'BTC_opt_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_BCH_opt.to_csv(dir_sim+'BCH_opt_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_BTC_bdd_opt.to_csv(dir_sim+'BTC_bdd_opt_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_BCH_bdd_opt.to_csv(dir_sim+'BCH_bdd_opt_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_pseudoBCH_opt.to_csv(dir_sim+'pseudoBCH_opt_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')
    df_DAA_4_opt.to_csv(dir_sim+'DAA_4_opt_blocktime_ps{}_{}obs'\
        .format(price_shock, num_iter)+'.csv')

    return None


## For analysis.
def compute_opt_w_array(prices, M=12.5, hash_slope=3, hash_ubd=55,
                        hash_center=1.5, b_target_sec=600,
                        W_init_low=1e-6, W_init_high=1e-3, W_grid=1e-8,
                        tol=1e-7, num_iter=1000):
    '''
    Given prices, compute the optimal winning rates.
    '''
    opt_winning_rates = []
    for t in range(prices.shape[0]):
        opt_w = optimal_winning_rate(price=prices[t], tol=tol,
            W_init_low=W_init_low, W_init_high=W_init_high, W_grid=W_grid)
        opt_winning_rates.append(opt_w)
    opt_winning_rates = np.array(opt_winning_rates)
    return opt_winning_rates


def optimal_winning_rate(price=3500, M=12.5, hash_slope=3, hash_ubd=55,
                         hash_center=1.5, b_target_sec=600,
                         W_init_low=1e-6, W_init_high=1e-3, W_grid=1e-8,
                         tol=1e-7, num_iter=1000):
    '''
    Compute the optimal winning rate given a price and
    the hash supply function.
    Solve the equasion numerically using the contraction mapping.
    '''
    W_init_list = np.arange(W_init_low, W_init_high, W_grid)
    for W_prev in W_init_list:
        for _ in range(num_iter):
            W_next = _contraction(W=W_prev, S=price, M=M, s=hash_slope,
                                  u=hash_ubd, B=b_target_sec, c=hash_center)
            if np.isnan(W_next):
                break
            if np.abs(W_next - W_prev) < tol:
                return W_next
            else:
                W_prev = W_next

    print('price = {}'.format(price))
    raise Exception('No solution found.')


def _contraction(W, S=3500, M=12.5, s=3, u=55, B=600, c=1.5):
    return (-(1/s)*np.log(W*u*B-1)+c)/(S*M)


def stats_fixed_path(df):
    '''
    over X is the fraction of paths that experience at least one over X minutes blocktime
    '''
    mean = df.mean().mean()
    std = df.std().mean()
    over60 = (((df>60).sum()/(df>60).sum()).fillna(0)).mean()
    over120 = (((df>120).sum()/(df>120).sum()).fillna(0)).mean()
    over180 = (((df>180).sum()/(df>180).sum()).fillna(0)).mean()

    return mean, std, over60, over120, over180


def make_stats(filelist, dir_sim):
    df_stats = pd.DataFrame()
    for simfile in filelist:
        df = pd.read_csv(simfile)
        mean, std, over60, over120, over180 = stats_fixed_path(df)
        df_stats.loc['mean', simfile[len(dir_sim):]] = mean
        df_stats.loc['std', simfile[len(dir_sim):]] = std
        df_stats.loc['over60', simfile[len(dir_sim):]] = over60
        df_stats.loc['over120', simfile[len(dir_sim):]] = over120
        df_stats.loc['over180', simfile[len(dir_sim):]] = over180

    return df_stats
