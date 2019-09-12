This repository contains the codes and notebooks used for the following paper:

> **A Lucas Critique to the Difficulty Adjustment Algorithm of the Bitcoin System**<br>
> Shunya Noda (University of British Columbia), Kyohei Okumura (University of Tokyo), Yoshinori Hashimoto (BUIDL, Ltd.)<br>
> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3410460
>
> **Abstract:** *The design of the difficulty adjustment algorithm (DAA) of the Bitcoin system is vulnerable as it dismisses miners' response to the difficulty adjustment. We develop an economic model of the Proof-of-Work based blockchain system. Our model allows miners to pause operation when the expected reward is below the shutdown point. Hence, the supply of aggregate hash power can be elastic in the cryptocurrency price and the difficulty target of the mining puzzle. We prove that, when the hash supply is elastic, the Bitcoin DAA fails to generate a new block at a constant rate. In contrast, the DAA of another blockchain system, Bitcoin Cash, is shown to be stable even when the cryptocurrency price is volatile and the supply of hash power is highly elastic. We also provide empirical evidence and simulation results supporting the model's prediction. Our results indicate that the current Bitcoin system might collapse once a sharp price fall lowers the reward for mining denominated in fiat money. However, such a crisis can be prevented through upgrading.*


## Resources

* [Notebook for impulse response](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/impulse_response.ipynb)
    - The effect of one-time persistent price shock (Section 3.4)

* [Notebook for MLE](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/MLE.ipynb)
    - Estimate the hash supply function. (Section 4.3)


* [Notebook for simulation](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/simulation.ipynb)
    - Section 5.1, 5.2, 5.3

* [Notebook for simulation about miners' profit](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/miner_profit.ipynb)
    - Section 5.4