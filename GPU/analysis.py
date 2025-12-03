import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def binder_cumulant(m):
    # m: array of magnetizations per spin (can be + or -)
    m2 = np.mean(m*m)
    m4 = np.mean(m*m*m*m)
    U = 1.0 - (m4 / (3.0 * m2 * m2))
    return U

def integrated_autocorr_time(x, maxlag=None):
    # return integrated autocorrelation time using normalized autocorrelation
    x = np.asarray(x)
    n = len(x)
    x = x - np.mean(x)
    if maxlag is None:
        maxlag = min(n-1, 1000)
    # compute autocorrelation via FFT for efficiency
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf /= acf[0]
    aclen = min(maxlag, n-1)
    # integrated time (sum of acf excluding lag 0)
    tau_int = 0.5 + np.sum(acf[1:aclen+1])
    return tau_int, acf[:aclen+1]

def analyze(csvfile):
    df = pd.read_csv(csvfile)
    # group by temperature
    grouped = df.groupby('T')
    Ts = []
    mean_abs_m = []
    binder = []
    tau_est = []
    for T, g in grouped:
        # g contains many runs and samples; we will first average across runs & samples
        # But to estimate errors properly, collapse all samples from all runs at same T
        m = g['m'].values
        # magnetization absolute
        mean_abs_m.append(np.mean(np.abs(m)))
        binder.append(binder_cumulant(m))
        # estimate autocorr per run and average tau across runs
        taus = []
        for run_id, sub in g.groupby('run_id'):
            arr = sub.sort_values('sample_idx')['m'].values
            if len(arr) < 10:
                continue
            tau, acf = integrated_autocorr_time(arr, maxlag=min(200,len(arr)-1))
            taus.append(tau)
        tau_m = np.mean(taus) if len(taus)>0 else np.nan
        tau_est.append(tau_m)
        Ts.append(T)

    Ts = np.array(Ts)
    mean_abs_m = np.array(mean_abs_m)
    binder = np.array(binder)
    tau_est = np.array(tau_est)

    # Sort by T
    order = np.argsort(Ts)
    Ts = Ts[order]
    mean_abs_m = mean_abs_m[order]
    binder = binder[order]
    tau_est = tau_est[order]

    # Plot results
    plt.figure()
    plt.plot(Ts, mean_abs_m, marker='o')
    plt.xlabel('T')
    plt.ylabel('<|m|>')
    plt.title('Average absolute magnetization')
    plt.grid(True)
    plt.savefig('magnetization_vs_T.png', dpi=200)

    plt.figure()
    plt.plot(Ts, binder, marker='o')
    plt.xlabel('T')
    plt.ylabel('Binder cumulant U')
    plt.title('Binder cumulant vs T')
    plt.grid(True)
    plt.savefig('binder_vs_T.png', dpi=200)

    plt.figure()
    plt.plot(Ts, tau_est, marker='o')
    plt.xlabel('T')
    plt.ylabel('Estimated integrated autocorrelation time (sweeps)')
    plt.title('Estimated tau_int vs T')
    plt.grid(True)
    plt.savefig('tau_vs_T.png', dpi=200)

    # print table
    out = pd.DataFrame({'T':Ts, 'mean_abs_m':mean_abs_m, 'binder':binder, 'tau_int_est':tau_est})
    out.to_csv('analysis_summary.csv', index=False)
    print(out)
    print("Plots saved: magnetization_vs_T.png, binder_vs_T.png, tau_vs_T.png")
    print("Summary saved: analysis_summary.csv")
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analysis.py magnetizations.csv")
        sys.exit(1)
    analyze(sys.argv[1])
