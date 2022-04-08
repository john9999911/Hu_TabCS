import scipy.stats as stats

DeepCS_MRR_top1_10times_bootstrap = [0.2409, 0.2361, 0.2312, 0.2349, 0.2346, 0.2401, 0.2289, 0.2321, 0.2369, 0.2283]

TSACS_TASF_MRR_10times_bootstrap = [0.5735977777777758, 0.567866785714284, 0.5669624999999976, 0.5727877777777753,
                                    0.5653392063492049, 0.5681766269841253, 0.5569228968253946, 0.5615689682539661,
                                    0.5645849603174583, 0.563360317460315]

print(stats.wilcoxon(DeepCS_MRR_top1_10times_bootstrap, TSACS_TASF_MRR_10times_bootstrap))
