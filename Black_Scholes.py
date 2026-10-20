import numpy as np
from scipy.stats import norm

def black_scholes_merton(S, K, T, r, sigma, q=0, option_type='call'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega / 100,
        'theta': theta / 365,
        'rho': rho / 100
    }


def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


if __name__ == "__main__":
    print("=== Black-Scholes-Merton Option Pricing ===\n")

    S = get_float("Current stock price (S): ")
    K = get_float("Strike price (K): ")
    T = get_float("Time to expiration in years (T): ")
    r = get_float("Risk-free interest rate, e.g. 0.05 for 5% (r): ")
    sigma = get_float("Volatility, e.g. 0.20 for 20% (sigma): ")
    q = get_float("Dividend yield, e.g. 0.02 for 2% (enter 0 if none) (q): ")

    option_type = input("Option type ('call' or 'put'): ").strip().lower()
    while option_type not in ('call', 'put'):
        option_type = input("Please enter 'call' or 'put': ").strip().lower()

    result = black_scholes_merton(S, K, T, r, sigma, q, option_type)

    print(f"\n=== Results ({option_type.upper()}) ===")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
