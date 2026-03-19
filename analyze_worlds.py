from experiment_configs import CONFIGS
from runner import simulate_many

def summarize(name, results):
    print(f"\nCONFIG: {name}")
    print(f"Investor total mean: {results['investor_totals'].mean():.3f}")
    print(f"Investor total std : {results['investor_totals'].std():.3f}")
    print(f"Trustee total mean : {results['trustee_totals'].mean():.3f}")
    print(f"Trustee total std  : {results['trustee_totals'].std():.3f}")
    print(f"Mean investment    : {results['mean_investments'].mean():.3f}")
    print(f"Mean repay prop    : {results['mean_repay_props'].mean():.3f}")

if __name__ == "__main__":
    for name in ["baseline_independent", "baseline_coordinated", "strong_coordination", "harsh_max"]:
        results = simulate_many(CONFIGS[name], n_episodes=500, seed=0)
        summarize(name, results)