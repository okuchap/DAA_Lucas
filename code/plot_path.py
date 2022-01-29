import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hash import *
from simulation_fixed_path import compute_opt_w_array


def coloring_BTC(block_times, y, init_adjust=2016-942, ylabel='', title=''):
    length = block_times.shape[0]
    cum_blocktimes = block_times.cumsum()
    time_shifts = []
    num_shifts = (length+init_adjust)//2016
    for i in range(num_shifts):
        time_shifts.append(cum_blocktimes[(i+1)*2016 - init_adjust - 1])
    y_low = y.min()
    y_high = y.max()

    for i in range(len(time_shifts)):
        if i < len(time_shifts)-1:
            x_low = time_shifts[i]/1440
            x_high = time_shifts[i+1]/1440
        else:
            x_low = time_shifts[i]/1440
            x_high = cum_blocktimes.max()/1440
        testx = [x_low, x_high, x_high, x_low]
        testy = [y_low, y_low, y_high, y_high]
        if i % 2 == 0:
            plt.fill(testx, testy, color="y", alpha=0.3)
        # else:
            # plt.fill(testx,testy,color="blue",alpha=0.3)

    plt.plot(cum_blocktimes/1440, y)
    plt.xticks(rotation=30)
    plt.xlabel('days')
    if len(ylabel) > 0:
        plt.ylabel(ylabel)
    if len(title) > 0:
        plt.title(title)
    plt.show()

    return None


def plot_paths(sim_list=[],
               title_list=['DAA-1', 'DAA-1 with bound',
                           'DAA-2', 'DAA-2 with bound']):
    '''
    sim_list is a list containing instances of simulation class.
    Plot the path of the variables that sim contains.
    default: sim_list = [sim_BTC, sim_BTC_bdd, sim_BCH, sim_BCH_bdd]
    '''
    init_adjust = 2016-942
    fig = plt.figure()
    for i in range(len(sim_list)):
        sim = sim_list[i]
        x = sim.block_times.cumsum()/1440  # minute -> day

        ax1 = fig.add_subplot(4, 4, 1+i)
        y = sim.winning_rates
        ax1.plot(x, y)
        #plt.setp(ax1.get_xticklabels(), rotation=30)
        ax1.set_xlabel('time (day)')
        #ax1.set_ylim(0.00003, 0.00007)

        # fill
        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])
        y_low = y.min()
        y_high = y.max()

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax1.fill(testx, testy, color="y", alpha=0.3)
        if i == 0:
            ax1.set_ylabel('Winning Rate $W(t)$\n(Pr(success)/Ehash)')
        ax1.set_title(title_list[i])

        ax2 = fig.add_subplot(4, 4, 5+i)
        y = sim.prices*sim.winning_rates*12.5
        ax2.plot(x, y)
        #plt.setp(ax3.get_xticklabels(), rotation=30)
        ax2.set_xlabel('time (day)')
        #ax2.set_ylim(0.5, 3.0)

        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])
        y_low = y.min()
        y_high = y.max()

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax2.fill(testx, testy, color="y", alpha=0.3)

        if i == 0:
            ax2.set_ylabel('Reward $W(t)M(t)S(t)$\n(USD/Ehash)')
        # plt.title(title_list[i])

        ax3 = fig.add_subplot(4, 4, 9+i)
        y = sim.hash_rates
        ax3.plot(x, y)
        #plt.setp(ax4.get_xticklabels(), rotation=30)
        ax3.set_xlabel('time (day)')
        #ax3.set_ylim(0, 55)

        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])
        y_low = y.min()
        y_high = y.max()

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax3.fill(testx, testy, color="y", alpha=0.3)

        if i == 0:
            ax3.set_ylabel('Hash Rate $H(t)$\n(Ehash/s)')
        # plt.title(title_list[i])

        ax4 = fig.add_subplot(4, 4, 13+i)
        y = sim.block_times
        ax4.plot(x, y)
        #plt.setp(ax2.get_xticklabels(), rotation=30)
        ax4.set_xlabel('time (day)')
        #ax4.set_ylim(0, 500)

        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])
        y_low = y.min()
        y_high = y.max()

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax4.fill(testx, testy, color="y", alpha=0.3)

        if i == 0:
            ax4.set_ylabel('Block Time $B(t)$\n(min.)')
        # plt.title(title_list[i])

    plt.tight_layout()
    fig.align_labels()

    plt.show()
    return None


def plot_paths_2(exprvs=pd.DataFrame(), sim_list=[],
                 W_init_low=1e-6, W_init_high=1e-4, W_grid=1e-8, tol=1e-10,
                 title_list=['DAA-1(2016)', 'DAA-2(144)']):
    '''
    sim_list is a list containing instances of simulation class.
    Plot the path of the variables that sim contains.
    default: sim_list = [sim_BTC, sim_BTC_bdd, sim_BCH, sim_BCH_bdd]

    Parameters
    ----------
        exprvs: numpy array containing block shocks delta(t) ~ Exp(1).

        sim_list:
            list containing instances of simulation class
            default: sim_list = [sim1, sim2] where sim1 contains the data
            about DAA-1 and sim2 contains the data abount DAA-2.

        title_list:
            list containing titles used when graphs are plotted.
    '''
    # assuming the height of the first block to be created is 551443
    init_adjust = 2016-942

    fig = plt.figure()
    for i in range(len(sim_list)):
        sim = sim_list[i]
        x = sim.block_times.cumsum()/1440  # minute -> day

        # winning rate
        ax1 = fig.add_subplot(4, 2, 1+i)
        y = sim.winning_rates
        opt_w = compute_opt_w_array(sim.prices, W_init_low=W_init_low,
                                    W_init_high=W_init_high, W_grid=W_grid, tol=tol)
        ax1.plot(x, y, label='real')
        ax1.plot(x, opt_w, label='first-best')
        ax1.legend(loc='upper right')
        #plt.setp(ax1.get_xticklabels(), rotation=30)
        ax1.set_xlabel('time (day)')

        # y_low = y.min()
        # y_high = y.max()
        y_low = 0.000025
        y_high = 0.000095
        ax1.set_ylim(y_low, y_high)

        # fill
        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax1.fill(testx, testy, color="y", alpha=0.3)
        if i == 0:
            ax1.set_ylabel('Winning Rate $W(t)$\n(Pr(success)/Ehash)')
        ax1.set_title(title_list[i])

        # reward
        ax2 = fig.add_subplot(4, 2, 3+i)
        opt_reward = opt_w*12.5*sim.prices
        y = sim.prices*sim.winning_rates*12.5
        ax2.plot(x, y, label='real')
        ax2.plot(x, opt_reward, label='first-best')
        ax2.legend(loc='upper right')
        #plt.setp(ax3.get_xticklabels(), rotation=30)
        ax2.set_xlabel('time (day)')

        # y_low = y.min()
        # y_high = y.max()
        y_low = 0.9
        y_high = 3.7
        ax2.set_ylim(y_low, y_high)

        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax2.fill(testx, testy, color="y", alpha=0.3)

        if i == 0:
            ax2.set_ylabel('Reward $R(t)$\n(USD/Ehash)')
        # plt.title(title_list[i])

        # hash rate
        ax3 = fig.add_subplot(4, 2, 5+i)
        opt_hash = hash(opt_reward)
        y = sim.hash_rates
        ax3.plot(x, y, label='real')
        ax3.plot(x, opt_hash, label='first-best')
        ax3.legend(loc='upper right')
        #plt.setp(ax4.get_xticklabels(), rotation=30)
        ax3.set_xlabel('time (day)')

        # y_low = y.min()
        # y_high = y.max()
        y_low = 10
        y_high = 60
        ax3.set_ylim(y_low, y_high)

        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax3.fill(testx, testy, color="y", alpha=0.3)

        if i == 0:
            ax3.set_ylabel('Hash Rate $H(t)$\n(Ehash/s)')
        # plt.title(title_list[i])

        # block time
        ax4 = fig.add_subplot(4, 2, 7+i)
        y = sim.block_times
        opt_blocktime = 10 * exprvs[: x.shape[0]]
        # The following two lines should be fixed: The graph should be bar graphs and I should have used ax4.bar
        # As the horizontal line is very short, there is little problem...? (But it seems to me that the graphs in the paper are bar graphs.)
        ax4.plot(x, y, label='real', linewidth=1)
        ax4.plot(x, opt_blocktime, label='first-best', alpha=0.5,
                 linewidth=1)
        ax4.legend(loc='upper right')
        #plt.setp(ax2.get_xticklabels(), rotation=30)
        ax4.set_xlabel('time (day)')

        y_low = y.min()
        y_high = y.max()
        y_low = 0
        y_high = 400
        ax4.set_ylim(y_low, y_high)

        length = sim.block_times.shape[0]
        cum_blocktimes = sim.block_times.cumsum()
        time_shifts = []
        num_shifts = (length+init_adjust)//2016
        for j in range(num_shifts):
            time_shifts.append(cum_blocktimes
                               [(j+1)*2016 - init_adjust - 1])

        for j in range(len(time_shifts)):
            if j < len(time_shifts)-1:
                x_low = time_shifts[j]/1440
                x_high = time_shifts[j+1]/1440
            else:
                x_low = time_shifts[j]/1440
                x_high = cum_blocktimes.max()/1440
            testx = [x_low, x_high, x_high, x_low]
            testy = [y_low, y_low, y_high, y_high]
            if j % 2 == 0:
                ax4.fill(testx, testy, color="y", alpha=0.3)

        if i == 0:
            ax4.set_ylabel('Block Time $B(t)$\n(min.)')
        # plt.title(title_list[i])

    plt.tight_layout()
    fig.align_labels()

    plt.show()
    return None
