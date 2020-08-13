## An Economic Analysis of Difficulty Adjustment Algorithms in Proof-of-Work Blockchain Systems

![DAA_Lucas](https://user-images.githubusercontent.com/12281235/64770339-f50bc000-d587-11e9-9bcc-3fccd2bfaa93.png)

This repository contains the codes and notebooks used for the following paper:

> **An Economic Analysis of Difficulty Adjustment Algorithms in Proof-of-Work Blockchain Systems**<br>
> Shunya Noda (University of British Columbia), Kyohei Okumura (University of Tokyo), Yoshinori Hashimoto (BUIDL, Ltd.)<br>
> https://ssrn.com/abstract=3410460
>
> **Abstract:** *The design of the difficulty adjustment algorithm (DAA) of the Bitcoin system is vulnerable as it dismisses miners' responses to policy changes. We develop an economic model of the Proof-of-Work based blockchain system. Our model allows miners to pause operation when the expected reward is below the shutdown point. Hence, the supply of aggregate hash power can be elastic in the cryptocurrency price and the difficulty target of the mining puzzle. We prove that, when the hash supply is elastic, the Bitcoin DAA fails to adjust the block arrival rate to the targeted level. In contrast, the DAA of another blockchain system, Bitcoin Cash, is shown to be stable even when the cryptocurrency price is volatile and the supply of hash power is highly elastic. We also provide empirical evidence and simulation results supporting the model's prediction. Our results indicate that the current Bitcoin system might collapse if a sharp price reduction lowers the reward for mining denominated in fiat money. While this crisis can be prevented through the upgrading of DAA, we also discuss that miners may disagree on upgrading the DAA because many of them obtain a larger expected profit from an unstable DAA.*


## Resources

* [Notebook for impulse response](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/impulse_response.ipynb)
    - Section 4.4

* [Notebook for empirical analysis](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/empirical_analysis.ipynb)
    - Section 5

* [Notebook for simulation](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/simulation.ipynb)
    - Section 6.1, 6.2, 6.3

* [Notebook for simulation about miners' profit](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/miner_profit.ipynb)
    - Section 6.4
