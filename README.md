## An Economic Analysis of Difficulty Adjustment Algorithms in Proof-of-Work Blockchain Systems

![DAA_Lucas](https://user-images.githubusercontent.com/12281235/64770339-f50bc000-d587-11e9-9bcc-3fccd2bfaa93.png)

This repository contains the codes and notebooks used for the following paper:

> **An Economic Analysis of Difficulty Adjustment Algorithms in Proof-of-Work Blockchain Systems**<br>
> Shunya Noda (University of British Columbia), Kyohei Okumura (Northwestern Univerisity), Yoshinori Hashimoto (Turingum K.K.,)<br>
> https://ssrn.com/abstract=3410460
>
> **Abstract:** *The design of the Bitcoin \textit{difficulty adjustment algorithm} (DAA) is vulnerable as it dismisses miners’ responses to policy changes. We develop an economic model of the Proof-of-Work blockchain system. Our model allows miners to pause operation when it is not profitable. Hence, the supply of aggregate hash power can be elastic in the cryptocurrency price and the difficulty of the mining puzzle. We prove that, when the hash supply is elastic, the Bitcoin DAA fails to stabilize the block arrival rate. In contrast, the DAA of another cryptocurrency, Bitcoin Cash, is stable under a general condition. We also provide empirical evidence and simulation results supporting the model’s prediction. Our results indicate that the current Bitcoin system might collapse once a sharp price reduction lowers the mining reward. While this crisis can be prevented through upgrading, miners may disagree because many of them obtain a larger profit from an unstable DAA.*


## Resources

* [Notebook for impulse response](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/impulse_response.ipynb)
    - Section 4.4

* [Notebook for empirical analysis](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/empirical_analysis.ipynb)
    - Section 5

* [Notebook for simulation](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/simulation.ipynb)
    - Section 6.1, 6.2, 6.3

* [Notebook for simulation about miners' profit](https://github.com/okuchap/DAA_Lucas/blob/master/notebook/miner_profit.ipynb)
    - Section 6.4
